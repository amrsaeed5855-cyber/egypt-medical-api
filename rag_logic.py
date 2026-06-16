"""
rag_logic.py — Egyptian Pharmacy Subagent for ElevenLabs Workflow Orchestration
==============================================================================
Changed: startup data cleaning, LLM query extraction before search, multi-drug
lookups, conversation follow-up resolution, strict retrieval gates, price/form/usage
card fixes, substitute strength disambiguation, max 3 cards, removed unsafe fallbacks.

This service is a **pharmacy subagent**, not the primary medical assistant.

Architecture
------------
* ElevenLabs orchestrates the overall conversation (triage, booking, referrals).
* This subagent receives **delegated tasks** for medication guidance only.
* It returns **structured workflow responses** so ElevenLabs can resume control.

Scope: pharmacy only — drug info, prices, ingredients, OTC suggestions from dataset.
No diagnosis. Out of scope → `RETURN_TO_ORCHESTRATOR: true` to ElevenLabs.
Every drug shown includes dataset row number, price, and active ingredient.

Startup architecture (Railway-compatible — no OOM)
---------------------------------------------------
Embeddings and the FAISS index are generated OFFLINE (once) using
build_index.py, then committed to the repository as:

    faiss.index      — binary FAISS IndexFlatIP file

At startup this service does:
  1. Read egypt_drugs_cleaned_utf8.csv              (~instant)
  2. Load faiss.index with faiss.read_index()       (~instant, only if file exists)

It does NOT run build_index.py, call embed_model.encode() on the dataset, or
load SentenceTransformer at startup. The embed model loads lazily on the first
semantic query when ENABLE_SEMANTIC_SEARCH=true.

Run with:
    uvicorn app:app --host 0.0.0.0 --port $PORT

Required files next to app.py (generate with build_index.py):
    egypt_drugs_cleaned_utf8.csv
    faiss.index
"""

# ──────────────────────────────────────────────────────────────────────────────
# STANDARD IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import json
import os
import re
import time
from threading import Lock
import requests
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set

# ──────────────────────────────────────────────────────────────────────────────
# AI / RAG IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import faiss

from data_cleaning import clean_dataframe, display_form, is_generic_ingredient
from retrieval import DrugRetrievalEngine, extract_strengths
from medication_context import (
    FORM_NOT_FOUND_MSG,
    NO_APPROPRIATE_RESULT_MSG,
    NO_SUBSTITUTE_MSG,
    MedicationSearchContext,
    extract_requested_form,
    ingredient_hint_for_trade,
    resolve_conversation_query,
    row_matches_context_refinement,
    row_matches_requested_form,
    score_row_for_query,
)
from response_grounding import (
    assemble_grounded_response,
    sanitize_medical_text,
    strip_hallucinated_drug_content,
)
from trade_name_utils import (
    classify_query,
    extract_drug_name_from_query,
    is_ambiguous_followup,
    is_show_more_request,
    normalize_text,
    resolve_trade_alias,
    split_multi_drug_names,
)

# ──────────────────────────────────────────────────────────────────────────────
# FASTAPI IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
from pydantic import BaseModel


# ══════════════════════════════════════════════════════════════════════════════
# ① GEMINI CONFIG
# ══════════════════════════════════════════════════════════════════════════════
def _load_gemini_api_keys() -> list:
    keys = []
    for env_name in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"):
        val = os.getenv(env_name, "").strip()
        if val:
            keys.append(val)
    return keys


GEMINI_API_KEYS: list = _load_gemini_api_keys()
GEMINI_MODEL          = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TIMEOUT_SEC    = int(os.getenv("GEMINI_TIMEOUT_SEC", "25"))
CHAT_TIMEOUT_SEC      = int(os.getenv("CHAT_TIMEOUT_SEC", "50"))
# false on Railway (512MB–1GB) — loading PyTorch + the model OOM-kills the container.
# Set ENABLE_SEMANTIC_SEARCH=true only on a host with ≥2GB RAM.
ENABLE_SEMANTIC_SEARCH = os.getenv("ENABLE_SEMANTIC_SEARCH", "false").strip().lower() in ("1", "true", "yes")
RPM_LIMIT             = 10
MIN_INTERVAL          = 60.0 / RPM_LIMIT
_last_call_time: float = 0.0
_gemini_key_index: int = 0
_rag_lock = Lock()


# ══════════════════════════════════════════════════════════════════════════════
# ② SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """أنت صيدلي مصري عندك 15 سنة خبرة — **subagent صيدلة** داخل workflow ElevenLabs.
الوكيل الرئيسي بيدير المحادثة العامة والتشخيص والحجز. أنت للصيدلة فقط.

## دورك
- شرح استخدام، تحذيرات، وملاحظات دوائية.
- إجابات قصيرة بالعامية المصرية.
- البدائل والأسعار من قاعدة البيانات — لا تخترع منتجات.

## استفسارات المنتج
- **لا تسأل** عن سن أو حمل أو حساسية لسؤال سعر/بديل.
- **لا تكتب** أسماء أو أسعار أو مواد فعالة — البطاقات تعرضها.
- اكتب **إرشادات وملاحظات فقط**.

## ممنوع
- لا تشخّص ولا تحجز مواعيد.
- لا تكتب بلوكات 💊 أو أرقام صفوف.
- لو طلب تشخيص أو حجز أو علاج لأعراض أو طوارئ → `RETURN_TO_ORCHESTRATOR: true`.

## بلوك workflow (إلزامي)
───WORKFLOW_RESPONSE───
TASK_STATUS: completed|needs_info|out_of_scope|escalate
RETURN_TO_ORCHESTRATOR: true|false
ESCALATION_REASON: none|booking|diagnosis|emergency|referral
MISSING_INFO: item1 | item2
"""


# ══════════════════════════════════════════════════════════════════════════════
# ③ DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ClinicalPlan:
    visible_text: str = ""
    non_drug_advice: list = field(default_factory=list)
    escalation_level: str = "none"
    workflow: dict = field(default_factory=dict)


@dataclass
class WorkflowResponse:
    """Structured response for ElevenLabs workflow orchestration."""
    response: str
    task_status: str = "completed"
    return_to_orchestrator: bool = False
    escalation_reason: str = "none"
    medications: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    missing_info: list = field(default_factory=list)
    ingredients: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "response": self.response,
            "task_status": self.task_status,
            "return_to_orchestrator": self.return_to_orchestrator,
            "escalation_reason": self.escalation_reason,
            "medications": self.medications,
            "warnings": self.warnings,
            "missing_info": self.missing_info,
            "ingredients": self.ingredients,
        }


@dataclass
class PatientContext:
    age: Optional[int] = None
    sex: str = "unknown"
    pregnant: Optional[bool] = None
    breastfeeding: Optional[bool] = None
    allergies: list = field(default_factory=list)
    allergies_asked: bool = False
    chronic_conditions: list = field(default_factory=list)
    complaint_text: str = ""
    red_flags: list = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# ④ TEXT / NLP UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
AR_NUMS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

REAL_RED_FLAG_PATTERNS = [
    (r"ضيق نفس شديد|مش عارف اتنفس|نهجان شديد|اختناق|صعوبة تنفس شديد", "ضيق نفس شديد"),
    (r"ألم صدر شديد|وجع صدر شديد|ضغط على الصدر|ألم صدر", "ألم صدر"),
    (r"فقدان وعي|اغماء|إغماء|مش واعي", "إغماء/فقدان وعي"),
    (r"تشنجات|convulsion|seizure", "تشنجات"),
    (r"قيء دم|ترجيع دم|دم في القيء", "قيء دم"),
    (r"براز أسود|دم في البراز|نزيف شرجي", "نزيف هضمي محتمل"),
    (r"ضعف مفاجئ|ميل في الوجه|لخبطة كلام|تلعثم مفاجئ", "أعراض عصبية حادة"),
    (r"طفح .* مع ضيق نفس|تورم اللسان|تورم الشفايف", "حساسية شديدة محتملة"),
    (r"حرارة فوق ?40|سخونية فوق ?40|حراره ?4[0-9]", "حرارة شديدة جدًا"),
    (r"جفاف شديد|مش بيشرب|قلة بول شديدة|مفيش بول", "جفاف شديد"),
    (r"ألم بطن شديد جدًا|بطن ناشفة|تيبس البطن", "ألم بطن حاد"),
    (r"حامل.*نزيف|نزيف.*حامل", "نزيف أثناء الحمل"),
    (r"تيبس رقبة|تصلب الرقبة|neck stiffness|مش بقدر احرك رقبتي", "تيبس الرقبة"),
    (r"صداع شديد|وجع راس شديد|اسوا صداع|صداع مفاجئ شديد", "صداع شديد"),
    (r"ترجيع مستمر|قيء مستمر|مش قادر امسك اكل|ترجيع متكرر", "قيء مستمر"),
    (r"ارتباك|مش فاهم|تشويش|confusion|مش عارف فين", "ارتباك/تشويش"),
]

CONDITION_KEYWORDS = {
    "kidney":       ["كلى", "فشل كلوي", "renal", "kidney"],
    "liver":        ["كبد", "التهاب كبد", "cirrhosis", "liver"],
    "ulcer":        ["قرحة", "نزيف معدة", "قرحة معدة"],
    "hypertension": ["ضغط", "ضغط عالي", "hypertension"],
    "diabetes":     ["سكر", "سكري", "diabetes"],
    "asthma":       ["ربو", "asthma"],
    "heart":        ["قلب", "فشل قلبي", "ذبحة", "heart"],
}

INGREDIENT_SAFETY_RULES = {
    "ibuprofen":            {"avoid_in_pregnancy": True, "avoid_conditions": ["kidney", "ulcer", "hypertension", "diabetes"], "caution_conditions": ["heart", "asthma"], "min_age": 12, "purpose": "مسكن ومضاد التهاب"},
    "diclofenac":           {"avoid_in_pregnancy": True, "avoid_conditions": ["kidney", "ulcer", "heart", "hypertension", "diabetes"], "caution_conditions": ["asthma"], "min_age": 14, "purpose": "مسكن ومضاد التهاب"},
    "naproxen":             {"avoid_in_pregnancy": True, "avoid_conditions": ["kidney", "ulcer", "hypertension", "diabetes"], "caution_conditions": ["heart"], "min_age": 12, "purpose": "مسكن ومضاد التهاب"},
    "pseudoephedrine":      {"avoid_in_pregnancy": True, "avoid_conditions": ["hypertension", "heart"], "caution_conditions": ["diabetes"], "min_age": 12, "purpose": "مزيل احتقان الأنف"},
    "phenylephrine":        {"avoid_in_pregnancy": True, "avoid_conditions": ["hypertension", "heart"], "caution_conditions": ["diabetes"], "min_age": 12, "purpose": "مزيل احتقان"},
    "loratadine":           {"min_age": 2, "purpose": "مضاد حساسية"},
    "cetirizine":           {"min_age": 2, "purpose": "مضاد حساسية"},
    "dextromethorphan":     {"min_age": 6, "purpose": "مضاد سعال"},
    "guaifenesin":          {"min_age": 4, "purpose": "طارد بلغم"},
    "loperamide":           {"min_age": 12, "avoid_conditions": ["ulcerative colitis"], "caution_conditions": ["liver"], "purpose": "مضاد إسهال"},
    "paracetamol":          {"caution_conditions": ["liver"], "purpose": "خافض حرارة ومسكن"},
    "acetaminophen":        {"caution_conditions": ["liver"], "purpose": "خافض حرارة ومسكن"},
    "omeprazole":           {"purpose": "مثبط حموضة المعدة"},
    "oral rehydration salts": {"purpose": "تعويض سوائل الجسم"},
}

PRODUCT_INFO_QUERY_TYPES = frozenset({"product_info", "substitute"})
MAX_DRUG_CARDS = 3
MAX_DRUG_CARDS_MORE = 3
DOSAGE_LIKE_RE = re.compile(
    r"^\s*\d+(?:\.\d+)?\s*(?:mg|mcg|g|gm|ml|iu|iu/ml|%)\s*$|^\s*\d+(?:\.\d+)?\s*$",
    re.IGNORECASE,
)
NOT_FOUND_MSG = NO_APPROPRIATE_RESULT_MSG

OUT_OF_SCOPE_PATTERNS = {
    "booking": [
        r"حجز", r"موعد", r"appointment", r"book\s", r"احجز", r"عايز\s*موعد",
        r"ميعاد", r"كشف\s*عند", r"احجزلي",
    ],
    "diagnosis": [
        r"ايه\s*المرض", r"عندي\s*ايه", r"تشخيص", r"diagnos",
        r"ايه\s*سبب", r"هل\s*ده\s*خطير",
    ],
    "referral": [
        r"تحويل\s*لدكتور", r"عايز\s*دكتور", r"طبيب\s*متخصص",
        r"specialist", r"referral",
    ],
    "symptom": [
        r"دواء ل", r"عاوز دواء", r"عايز دواء", r"علاج", r"للصداع", r"للكحة",
        r"للحرارة", r"للزكام", r"للألم", r"للام", r"للمغص", r"للاسهال",
        r"للإسهال", r"medicine for", r"treatment for",
    ],
}

EXCLUDED_FORMS = [
    "vial", "ampoule", "injection", "infusion", "iv", "i.v", "suppository",
    "امبول", "حقن", "وريدي", "امبولة", "لبوس"
]

BABY_KEYWORDS = [
    "teething", "baby", "infant", "toddler", "child",
    "تسنين", "رضع", "أطفال", "طفل", "رضيع"
]



def dedupe_keep_order(items: list) -> list:
    seen, out = set(), []
    for item in items:
        v = item.strip()
        if not v:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def conversation_to_text(history: list) -> str:
    return "\n".join(
        msg.get("content", "") for msg in (history or []) if msg.get("role") == "user"
    )


def extract_age(text: str) -> Optional[int]:
    text = text.translate(AR_NUMS)
    patterns = [
        r"(\d{1,3})\s*سنه", r"عندي\s*(\d{1,3})\s*سنه",
        r"السن\s*(\d{1,3})", r"age\s*[:=]?\s*(\d{1,3})"
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            age = int(m.group(1))
            if 0 < age < 120:
                return age
    for n in re.findall(r"\b(\d{1,3})\b", text):
        age = int(n)
        if age < 1 or age > 110:
            continue
        if re.search(rf"{age}\s*ساعة", text) or re.search(rf"{age}\s*يوم", text):
            continue
        if "سنة" in text or "سنين" in text:
            return age
        if age >= 10:
            return age
    return None


def extract_sex(text: str) -> str:
    norm = normalize_text(text)
    if any(w in norm for w in ["ذكر", "راجل", "male"]):
        return "male"
    if any(w in norm for w in ["انثى", "انثي", "ست", "بنت", "female", "حامل", "مرضع"]):
        return "female"
    return "unknown"


def has_negation_response(text: str) -> bool:
    norm = normalize_text(text)
    return any(neg in norm for neg in ["لا", "مفيش", "مش", "مافيش", "لأ", "لالا", "لا لا"])


def parse_pregnancy_breastfeeding(text: str):
    norm = normalize_text(text)
    pregnant, breastfeeding = None, None
    if re.search(r"(لا|مش|مفيش|لأ|لست)\s*(حامل|حامل؟)", norm) or "غير حامل" in norm or "مش حامل" in norm:
        pregnant = False
    if re.search(r"انا حامل|بنت حامل|حامل في", norm):
        pregnant = True
    if re.search(r"(لا|مش|مفيش|لأ)\s*(مرضع|ترضع|رضاع)", norm) or "مش برضع" in norm or "مش مرضع" in norm:
        breastfeeding = False
    if re.search(r"(مرضع|برضع|بترضع|رضاعة)", norm):
        breastfeeding = True
    return pregnant, breastfeeding


def extract_list_after_keywords(text: str, keywords: list, check_negation: bool = True) -> list:
    if check_negation and has_negation_response(text):
        return []
    found = []
    for kw in keywords:
        m = re.search(rf"{kw}\s*[:：]?\s*([^\n\.،]+)", text, re.IGNORECASE)
        if m:
            raw = m.group(1)
            if has_negation_response(raw):
                return []
            for item in re.split(r"[,،/|+]", raw):
                item = item.strip()
                if item and not has_negation_response(item):
                    found.append(item)
    return dedupe_keep_order(found)


def extract_conditions(text: str) -> list:
    norm = normalize_text(text)
    return dedupe_keep_order([
        canonical for canonical, kws in CONDITION_KEYWORDS.items()
        if any(k in norm for k in kws)
    ])


def detect_out_of_scope(query: str) -> Optional[str]:
    norm = normalize_text(query)
    for reason, patterns in OUT_OF_SCOPE_PATTERNS.items():
        if any(re.search(p, norm) for p in patterns):
            return reason
    return None


def apply_delegation_context(ctx: PatientContext, delegation: Optional[dict]) -> PatientContext:
    """Merge patient context passed from ElevenLabs orchestrator."""
    if not delegation:
        return ctx
    pc = delegation.get("patient_context") or {}

    if pc.get("age") is not None:
        ctx.age = int(pc["age"])
    if pc.get("sex"):
        ctx.sex = str(pc["sex"]).lower()
    if pc.get("pregnant") is not None:
        ctx.pregnant = bool(pc["pregnant"])
    if pc.get("breastfeeding") is not None:
        ctx.breastfeeding = bool(pc["breastfeeding"])
    if pc.get("allergies") is not None:
        ctx.allergies = list(pc["allergies"])
        ctx.allergies_asked = True
    if pc.get("chronic_conditions"):
        ctx.chronic_conditions = dedupe_keep_order(list(pc["chronic_conditions"]))
    if pc.get("red_flags"):
        ctx.red_flags = dedupe_keep_order(list(pc["red_flags"]))
    return ctx


def out_of_scope_response(reason: str) -> WorkflowResponse:
    messages = {
        "booking": "حجز المواعيد مش من اختصاص الصيدلي — هرجعك للمساعد الرئيسي يحجزلك.",
        "diagnosis": "التشخيص الطبي مش من اختصاص الصيدلي — هرجعك للمساعد الرئيسي يقيّم حالتك.",
        "referral": "التحويل لطبيب مختص محتاج المساعد الرئيسي — هرجعك له دلوقتي.",
        "symptom": "اقتراح علاج للأعراض مش من نطاق البحث عن الأدوية — هرجعك للمساعد الرئيسي.",
    }
    return WorkflowResponse(
        response=messages.get(reason, "الطلب ده خارج نطاق الصيدلي — هرجعك للمساعد الرئيسي."),
        task_status="out_of_scope",
        return_to_orchestrator=True,
        escalation_reason=reason,
    )


def extract_context(query: str, history: list) -> PatientContext:
    full_text = (conversation_to_text(history) + "\n" + query).strip()
    norm = normalize_text(full_text)
    child_age = None
    m = re.search(r"عنده\s*(\d+)\s*سنه", norm)
    if m:
        child_age = int(m.group(1))
    sex = extract_sex(full_text)
    pregnant, breastfeeding = parse_pregnancy_breastfeeding(full_text)
    allergies = extract_list_after_keywords(full_text, ["حساسيه", "allergy", "allergies", "allergic to"])
    allergies_asked = bool(allergies) or bool(re.search(r"حساسيه|allergy", norm))
    if has_negation_response(query) and re.search(r"حساسيه|allergy", normalize_text(query)):
        allergies = []
        allergies_asked = True
    chronic_conditions = extract_conditions(full_text)
    red_flags = []
    for pattern, label in REAL_RED_FLAG_PATTERNS:
        if re.search(pattern, norm, re.IGNORECASE):
            red_flags.append(label)
    age = extract_age(full_text)
    if child_age and not age:
        age = child_age
    return PatientContext(
        age=age,
        sex=sex,
        pregnant=pregnant,
        breastfeeding=breastfeeding,
        allergies=allergies,
        allergies_asked=allergies_asked,
        chronic_conditions=chronic_conditions,
        complaint_text=query.strip(),
        red_flags=dedupe_keep_order(red_flags),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ INTAKE GATE
# ══════════════════════════════════════════════════════════════════════════════
def has_real_emergency(ctx: PatientContext) -> bool:
    return bool(ctx.red_flags)


def subagent_emergency_escalation(ctx: PatientContext) -> Optional[WorkflowResponse]:
    """Pharmacy subagent does not handle emergencies — escalate to ElevenLabs."""
    if not has_real_emergency(ctx):
        return None
    return WorkflowResponse(
        response=(
            "🚨 فيه علامات خطر محتملة — ده خارج نطاق الصيدلي.\n"
            "هرجعك للمساعد الرئيسي يتولى التقييم الطارئ فوراً."
        ),
        task_status="escalate",
        return_to_orchestrator=True,
        escalation_reason="emergency",
    )


# ══════════════════════════════════════════════════════════════════════════════
# ⑥ GEMINI API CALL
# ══════════════════════════════════════════════════════════════════════════════
def _trim_history_for_gemini(history: list, max_messages: int = 10) -> list:
    h = history or []
    return h if len(h) <= max_messages else h[-max_messages:]


def _gemini_key_exhausted(error: dict) -> bool:
    code = error.get("code")
    if code in (429, 403):
        return True
    status = str(error.get("status", "")).upper()
    return "RESOURCE_EXHAUSTED" in status or "QUOTA" in status


def call_gemini(messages: list, system_prompt: str = None):
    global _last_call_time, _gemini_key_index
    elapsed = time.time() - _last_call_time
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)

    if not GEMINI_API_KEYS:
        print("❌ No Gemini API keys configured")
        return None, "gemini_error"

    url     = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt or SYSTEM_PROMPT}]},
        "contents": messages,
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
    }
    n_keys = len(GEMINI_API_KEYS)

    for key_attempt in range(n_keys):
        api_key = GEMINI_API_KEYS[_gemini_key_index]
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        try:
            _last_call_time = time.time()
            r    = requests.post(url, headers=headers, json=payload, timeout=GEMINI_TIMEOUT_SEC)
            resp = r.json()
            if "error" in resp:
                err = resp["error"]
                print(f"❌ Gemini API error (key {_gemini_key_index + 1}/{n_keys}): {err}")
                if _gemini_key_exhausted(err) and key_attempt < n_keys - 1:
                    _gemini_key_index = (_gemini_key_index + 1) % n_keys
                    print(f"🔑 Switching to Gemini API key {_gemini_key_index + 1}/{n_keys}")
                    continue
                if _gemini_key_exhausted(err):
                    return None, "rate_limit"
                return None, "gemini_error"
            candidates = resp.get("candidates") or []
            if not candidates:
                print(f"❌ Gemini empty response (key {_gemini_key_index + 1}): {resp}")
                if key_attempt < n_keys - 1:
                    _gemini_key_index = (_gemini_key_index + 1) % n_keys
                    continue
                return None, "rate_limit"
            parts = candidates[0].get("content", {}).get("parts") or []
            if not parts or "text" not in parts[0]:
                print(f"❌ Gemini missing text (key {_gemini_key_index + 1}): {resp}")
                if key_attempt < n_keys - 1:
                    _gemini_key_index = (_gemini_key_index + 1) % n_keys
                    continue
                return None, "rate_limit"
            text = parts[0]["text"].strip()
            text = re.sub(r'\(Internal Reasoning\).*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)
            text = re.sub(r'\(Response.*?\):\s*', '', text)
            return text, GEMINI_MODEL
        except requests.Timeout:
            print(f"❌ Gemini timeout (key {_gemini_key_index + 1}/{n_keys})")
            if key_attempt < n_keys - 1:
                _gemini_key_index = (_gemini_key_index + 1) % n_keys
                continue
            return None, "rate_limit"
        except Exception as e:
            print(f"❌ Gemini call exception (key {_gemini_key_index + 1}): {e}")
            return None, "gemini_error"

    return None, "rate_limit"


# ══════════════════════════════════════════════════════════════════════════════
# CLINICAL PLAN PARSER
# ══════════════════════════════════════════════════════════════════════════════
def _extract_field(pattern: str, txt: str, default: str = "") -> str:
    m = re.search(pattern, txt)
    return m.group(1).strip() if m else default


def parse_workflow_fields(wf_text: str) -> dict:
    raw_missing = _extract_field(r"MISSING_INFO:\s*([^\n]*)", wf_text)
    missing = [m.strip() for m in raw_missing.split("|") if m.strip()] if raw_missing else []
    return {
        "task_status": _extract_field(r"TASK_STATUS:\s*(\w+)", wf_text, "completed"),
        "return_to_orchestrator": _extract_field(
            r"RETURN_TO_ORCHESTRATOR:\s*(true|false)", wf_text, "false"
        ).lower() == "true",
        "escalation_reason": _extract_field(r"ESCALATION_REASON:\s*(\w+)", wf_text, "none"),
        "missing_info": missing,
    }


def parse_clinical_plan(raw_text: str) -> ClinicalPlan:
    plan = ClinicalPlan()
    text = (raw_text or "").strip()

    if "───WORKFLOW_RESPONSE───" in text:
        visible, wf_part = text.split("───WORKFLOW_RESPONSE───", 1)
        plan.workflow = parse_workflow_fields(wf_part)
        text = visible.strip()

    if "───CLINICAL_PLAN───" in text:
        text = text.split("───CLINICAL_PLAN───", 1)[0].strip()
        machine_part = (raw_text or "").split("───CLINICAL_PLAN───", 1)[1]
        raw_advice = _extract_field(r"NON_DRUG_ADVICE:\s*([^\n]*)", machine_part)
        if raw_advice:
            plan.non_drug_advice = [a.strip() for a in raw_advice.split("|") if a.strip()]
        plan.escalation_level = _extract_field(r"ESCALATION_LEVEL:\s*(\w+)", machine_part, "none")

    plan.visible_text = text
    return plan


# ══════════════════════════════════════════════════════════════════════════════
# ⑦ DATASET & INDEX LOADING  (precomputed — zero runtime encoding)
# ══════════════════════════════════════════════════════════════════════════════
# The embeddings and FAISS index are generated OFFLINE via build_index.py.
# At startup we only do cheap file I/O — no encode() call on the dataset.
#
# Required files (commit to repo alongside app.py):
#   egypt_drugs_cleaned_utf8.csv  — drug database
#   faiss.index                   — IndexFlatIP produced by build_index.py

CSV_PATH         = os.getenv("EGYPT_DRUGS_CSV",    "egypt_drugs_cleaned_utf8.csv")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH",   "faiss.index")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME",   "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

index:       Optional[faiss.Index]         = None
embed_model: Optional[Any] = None
_embed_load_failed: bool = False
retrieval_engine: Optional[DrugRetrievalEngine] = None
INGREDIENT_COL = "active_ingredient"
df = pd.DataFrame()

try:
    df_raw = pd.read_csv(CSV_PATH).fillna("").astype(str)
    df, _dataset_meta = clean_dataframe(df_raw)
    INGREDIENT_COL = _dataset_meta.get("ingredient_col", "ingredient_clean")
    if INGREDIENT_COL not in df.columns:
        INGREDIENT_COL = "active_ingredient"
    del df_raw
    import gc; gc.collect()
    print(f"[OK] CSV loaded & cleaned — {len(df)} rows")
except Exception as e:
    print(f"[ERR] CSV load error: {e}")

if os.path.isfile(FAISS_INDEX_PATH):
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"[OK] FAISS index loaded — {index.ntotal} vectors")
    except Exception as e:
        print(f"[ERR] FAISS index load error: {e}")
else:
    print(f"[WARN] {FAISS_INDEX_PATH} not found — semantic search disabled (lexical-only)")

if not df.empty:
    if ENABLE_SEMANTIC_SEARCH and index is not None:
        print("[OK] CSV + index ready — hybrid retrieval (semantic loads on first query)")
    else:
        print("[OK] CSV ready — hybrid retrieval (lexical-only)")


# ══════════════════════════════════════════════════════════════════════════════
# ⑧ DRUG RETRIEVAL (RAG CORE)
# ══════════════════════════════════════════════════════════════════════════════
def ingredient_rule_keys(active_ingredient: str) -> list:
    ai = active_ingredient.lower()
    return [k for k in INGREDIENT_SAFETY_RULES if k in ai]


def screen_ingredient_safety(active_ingredient: str, ctx: PatientContext):
    ai, reasons = active_ingredient.lower(), []
    ai_tokens = {t for t in re.split(r"[+/\s,\-]+", ai) if len(t) >= 3}
    for allergy in ctx.allergies:
        al = allergy.lower().strip()
        if len(al) < 3:
            continue
        if al in ai or ai in al:
            reasons.append("مستبعد بسبب حساسية مذكورة")
            break
        al_tokens = {t for t in re.split(r"[+/\s,\-]+", al) if len(t) >= 3}
        if ai_tokens & al_tokens:
            reasons.append("مستبعد بسبب حساسية مذكورة")
            break
    for key in ingredient_rule_keys(ai):
        rule = INGREDIENT_SAFETY_RULES.get(key, {})
        if "min_age" in rule and ctx.age and ctx.age < rule["min_age"]:
            reasons.append(f"عمر أقل من {rule['min_age']} سنة")
        if rule.get("avoid_in_pregnancy") and ctx.pregnant:
            reasons.append("مستبعد أثناء الحمل")
        if rule.get("avoid_in_breastfeeding") and ctx.breastfeeding:
            reasons.append("مستبعد أثناء الرضاعة")
        for cond in rule.get("avoid_conditions", []):
            if cond in ctx.chronic_conditions:
                reasons.append(f"مستبعد بسبب {cond}")
    nsaids = ["ibuprofen", "diclofenac", "naproxen"]
    if any(n in ai for n in nsaids) and (
        "hypertension" in ctx.chronic_conditions or "diabetes" in ctx.chronic_conditions
    ):
        reasons.append("مضادات الالتهاب ممنوعة تماماً مع الضغط أو السكر - تحتاج استشارة طبيب")
    decongestants = ["phenylephrine", "pseudoephedrine"]
    if any(d in ai for d in decongestants) and "hypertension" in ctx.chronic_conditions:
        reasons.append("مزيلات الاحتقان (فينيل إفرين/سودوإيفيدرين) قد ترفع الضغط — غير مناسبة")
    return len(reasons) == 0, reasons


def caution_notes_for_context(active_ingredient: str, ctx: PatientContext) -> list:
    ai, notes = active_ingredient.lower(), []
    for key in ingredient_rule_keys(ai):
        rule = INGREDIENT_SAFETY_RULES.get(key, {})
        for cond in rule.get("caution_conditions", []):
            if cond in ctx.chronic_conditions:
                notes.append(f"يحتاج حذر مع {cond}")
    nsaids = ["ibuprofen", "diclofenac", "naproxen"]
    if any(n in ai for n in nsaids) and (
        "hypertension" in ctx.chronic_conditions or "diabetes" in ctx.chronic_conditions
    ):
        notes.append("⚠️ خطير: هذا الدواء قد يرفع الضغط ويؤثر على الكلى - لا تستخدمه بدون إشراف طبي")
    decongestants = ["phenylephrine", "pseudoephedrine"]
    if any(d in ai for d in decongestants) and "hypertension" in ctx.chronic_conditions:
        notes.append("⚠️ تحذير: مزيل الاحتقان قد يرفع ضغط الدم — استخدم بحذر أو تجنبه")
    if "liver" in ctx.chronic_conditions and any(p in ai for p in ["paracetamol", "acetaminophen"]):
        notes.append("⚠️ استخدم بحذر مع مرض الكبد — لا تتجاوز الجرعة الموصى بها")
    return dedupe_keep_order(notes)


def get_embed_model() -> Optional[Any]:
    """Lazy-load SentenceTransformer on first call (skipped when ENABLE_SEMANTIC_SEARCH=false)."""
    global embed_model, _embed_load_failed
    if not ENABLE_SEMANTIC_SEARCH or _embed_load_failed:
        return None
    if embed_model is not None:
        return embed_model
    try:
        from sentence_transformers import SentenceTransformer
        print(f"🔵 Loading SentenceTransformer ({EMBED_MODEL_NAME})…")
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        print("🟢 SentenceTransformer ready")
        return embed_model
    except Exception as e:
        print(f"❌ SentenceTransformer load failed — rapidfuzz fallback: {e}")
        _embed_load_failed = True
        return None


def _init_retrieval_engine() -> None:
    global retrieval_engine
    if retrieval_engine is not None or df.empty:
        return
    retrieval_engine = DrugRetrievalEngine(
        df=df,
        index=index,
        ingredient_col=INGREDIENT_COL,
        get_embed_model=get_embed_model,
        enable_semantic=ENABLE_SEMANTIC_SEARCH,
    )


def _drug_row_filter(row: dict, row_index: int, _query: str, ctx: PatientContext) -> Optional[str]:
    ai = row.get(INGREDIENT_COL, "").strip().lower()
    name_ar = row.get("name_ar", "")
    name_en = row.get("name_en", "")
    if is_baby_drug(name_ar, name_en, ctx.age):
        return "baby_form"
    if any(f in name_en.lower() for f in EXCLUDED_FORMS):
        return "excluded_form"
    if any(f in name_ar for f in EXCLUDED_FORMS):
        return "excluded_form"
    allowed, _ = screen_ingredient_safety(ai, ctx)
    if not allowed:
        return "safety"
    return None


def is_baby_drug(name_ar: str, name_en: str, age: Optional[int]) -> bool:
    if age is not None and age > 12:
        name_comb = (name_ar + " " + name_en).lower()
        return any(kw in name_comb for kw in BABY_KEYWORDS)
    return False


def get_matching_drugs_for_ingredient(
    ingredient: str, excluded: set, ctx: PatientContext, max_results: int = 2
) -> List[Dict[str, Any]]:
    _init_retrieval_engine()
    if retrieval_engine is None or retrieval_engine.empty:
        return []

    def row_filter(row: dict, row_index: int, query: str) -> Optional[str]:
        return _drug_row_filter(row, row_index, query, ctx)

    return retrieval_engine.match_by_ingredient(
        ingredient=ingredient,
        excluded=excluded,
        row_filter=row_filter,
        max_results=max_results,
        caution_fn=caution_notes_for_context,
        ctx=ctx,
    )


def ingredient_purpose(ingredient: str) -> str:
    ing = ingredient.lower()
    for key, rule in INGREDIENT_SAFETY_RULES.items():
        if key in ing:
            return rule.get("purpose", "")
    return ""


def _usage_description(row_dict: dict, ai: str) -> str:
    for key in ("mechanism", "composition"):
        val = str(row_dict.get(key, "") or "").strip()
        if val and val.lower() not in ("nan", "unknown", "") and not DOSAGE_LIKE_RE.match(val):
            return val[:200]
    purpose = ingredient_purpose(ai)
    return purpose or ""


def _format_price(row_dict: dict) -> str:
    """Show CSV price as-is; never display estimated/corrected values."""
    raw = row_dict.get("price_egp_raw") or row_dict.get("price_egp", "")
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "0", "0.0"):
        return "السعر غير متاح"
    try:
        return f"{float(s):g} جنيه"
    except (ValueError, TypeError):
        return s if s else "السعر غير متاح"


def row_to_pharmacy_record(row_dict: dict) -> dict:
    """Map a dataset row to a pharmacy record for API + display."""
    ai = (
        row_dict.get("active_ingredient")
        or row_dict.get("ingredient_clean")
        or row_dict.get(INGREDIENT_COL, "")
    )
    ai_str = str(ai).strip()
    form_val = display_form(row_dict)
    dosage = str(row_dict.get("dosage_clean") or row_dict.get("dosage") or "").strip()
    if dosage.lower() in ("nan", "unknown", ""):
        dosage = ""
    rec = {
        "row": row_dict.get("row_id"),
        "row_index": row_dict.get("row_index"),
        "name_ar": row_dict.get("name_ar", ""),
        "name_en": row_dict.get("name_en", ""),
        "active_ingredient": ai_str,
        "price_egp": _format_price(row_dict),
        "dosage": dosage,
        "dose": row_dict.get("dose", ""),
        "usage": _usage_description(row_dict, ai_str),
        "warnings": row_dict.get("safety_cautions") or [],
        "retrieval_score": row_dict.get("retrieval_score"),
    }
    if form_val:
        rec["form"] = form_val
    return rec


def _trade_name_row_filter(ctx: PatientContext, relaxed: bool = False):
    def row_filter(row: dict, row_index: int, query: str) -> Optional[str]:
        if relaxed:
            ai = row.get(INGREDIENT_COL, "").strip().lower()
            name_ar = row.get("name_ar", "")
            name_en = row.get("name_en", "")
            if is_baby_drug(name_ar, name_en, ctx.age):
                return "baby_form"
            if any(f in name_en.lower() for f in EXCLUDED_FORMS):
                return "excluded_form"
            return None
        return _drug_row_filter(row, row_index, query, ctx)
    return row_filter


def search_drugs_by_name(
    query: str,
    ctx: PatientContext,
    max_results: int = MAX_DRUG_CARDS,
    relaxed: bool = False,
    requested_form: str = "",
    require_form: bool = False,
    med_context: Optional[MedicationSearchContext] = None,
) -> List[Dict[str, Any]]:
    """Multi-stage trade-name lookup with query extraction."""
    _init_retrieval_engine()
    if retrieval_engine is None or retrieval_engine.empty:
        return []

    drug_name = extract_drug_name_from_query(query) or query
    row_filter = _trade_name_row_filter(ctx, relaxed=relaxed)
    strengths = extract_strengths(query)
    if med_context and med_context.strengths:
        strengths = med_context.strengths | strengths

    results = retrieval_engine.match_by_trade_name(
        name=drug_name,
        row_filter=row_filter,
        max_results=max_results + 4,
        caution_fn=caution_notes_for_context,
        ctx=ctx,
        normalize_fn=normalize_text,
        relaxed_filter=relaxed,
        requested_form=requested_form,
        query_strengths=strengths,
        require_form=require_form,
    )
    if not results and drug_name != query:
        results = retrieval_engine.match_by_trade_name(
            name=query,
            row_filter=row_filter,
            max_results=max_results + 4,
            caution_fn=caution_notes_for_context,
            ctx=ctx,
            normalize_fn=normalize_text,
            relaxed_filter=relaxed,
            requested_form=requested_form,
            query_strengths=strengths,
            require_form=require_form,
        )

    if not results and requested_form:
        ing_hint = ingredient_hint_for_trade(drug_name)
        if ing_hint:
            ing_rows = get_matching_drugs_for_ingredient(
                ing_hint, set(), ctx, max_results=80
            )
            ing_rows = [r for r in ing_rows if row_matches_requested_form(r, requested_form)]
            if ing_rows:
                ing_rows.sort(key=lambda r: -score_row_for_query(r, drug_name, med_context))
            results = ing_rows[: max_results + 4]
        if not results:
            _init_retrieval_engine()
            if retrieval_engine is not None and not retrieval_engine.empty and ing_hint:
                scanned: List[Dict[str, Any]] = []
                for idx in range(len(retrieval_engine.df)):
                    row = retrieval_engine.df.iloc[idx].to_dict()
                    ai = str(row.get(INGREDIENT_COL, "") or row.get("active_ingredient", "")).lower()
                    if ing_hint not in ai:
                        continue
                    if not row_matches_requested_form(row, requested_form):
                        continue
                    reject = row_filter(row, idx, drug_name)
                    if reject:
                        continue
                    scanned.append({**row, "row_index": idx, "row_id": idx + 1})
                scanned.sort(key=lambda r: -score_row_for_query(r, drug_name, med_context))
                results = scanned[: max_results + 4]

    if med_context and med_context.is_active():
        results = [r for r in results if row_matches_context_refinement(r, med_context)]

    if strengths:
        def _strength_rank(row: dict) -> tuple:
            text = " ".join(str(row.get(k, "") or "") for k in (INGREDIENT_COL, "dosage_clean", "name_en"))
            row_s = extract_strengths(text)
            exact = 0 if strengths & row_s else 1
            wrong_strength = 1 if strengths and row_s and not (strengths & row_s) else 0
            return (wrong_strength, exact, -float(row.get("retrieval_score") or 0))

        results.sort(key=_strength_rank)
        if strengths:
            exact_matches = [
                r for r in results
                if extract_strengths(
                    " ".join(str(r.get(k, "") or "") for k in (INGREDIENT_COL, "dosage_clean", "name_en"))
                ) & strengths
            ]
            if exact_matches:
                results = exact_matches + [r for r in results if r not in exact_matches]

    return results[:max_results]


def pick_primary_product(
    rows: List[Dict[str, Any]],
    query: str,
    med_context: Optional[MedicationSearchContext] = None,
) -> Optional[Dict[str, Any]]:
    """Pick the best source product — prefer query form/strength/volume constraints."""
    if not rows:
        return None
    return max(rows, key=lambda row: score_row_for_query(row, query, med_context))


def llm_extract_query_entities(query: str) -> dict:
    """
    Lightweight pre-search extraction (no conversation history).
    Returns trade_name, active_ingredient, intent, confidence.
    """
    default = {
        "trade_name": "",
        "active_ingredient": "",
        "intent": classify_query(query),
        "confidence": 0.0,
    }
    prompt = (
        'Extract JSON only: {"trade_name":"","active_ingredient":"","intent":"price|substitute|info|symptom","confidence":0.0}\n'
        f"Query: {query.strip()[:200]}"
    )
    text, status = call_gemini(
        [{"role": "user", "parts": [{"text": prompt}]}],
        system_prompt="Return valid JSON only. No markdown.",
    )
    if not text or status in ("rate_limit", "gemini_error"):
        drug = extract_drug_name_from_query(query) or ""
        if drug:
            default["trade_name"] = drug
            default["confidence"] = 0.5
        return default
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(m.group(0) if m else text)
        return {
            "trade_name": str(parsed.get("trade_name", "") or "").strip(),
            "active_ingredient": str(parsed.get("active_ingredient", "") or "").strip().lower(),
            "intent": str(parsed.get("intent", default["intent"]) or default["intent"]).strip(),
            "confidence": float(parsed.get("confidence", 0) or 0),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        drug = extract_drug_name_from_query(query) or ""
        if drug:
            default["trade_name"] = drug
            default["confidence"] = 0.4
        return default


def _search_offset_from_history(history: list) -> int:
    """How many results were already shown for a show-more follow-up."""
    if not history:
        return 0
    last_user = next((m for m in reversed(history) if m.get("role") == "user"), None)
    if last_user and is_show_more_request(last_user.get("content", "")):
        return MAX_DRUG_CARDS
    return 0


def _lookup_single_drug(
    drug_query: str,
    ctx: PatientContext,
    entities: dict,
    max_results: int = MAX_DRUG_CARDS,
    offset: int = 0,
    query_type: str = "product_info",
    requested_form: str = "",
    require_form: bool = False,
    med_context: Optional[MedicationSearchContext] = None,
) -> List[Dict[str, Any]]:
    """Independent lookup for one drug name with LLM-assisted ingredient fallback."""
    _init_retrieval_engine()
    if retrieval_engine is None or retrieval_engine.empty:
        return []

    trade = entities.get("trade_name") or extract_drug_name_from_query(drug_query) or drug_query
    rows = search_drugs_by_name(
        trade,
        ctx,
        max_results=max_results + offset + 2,
        relaxed=False,
        requested_form=requested_form,
        require_form=require_form,
        med_context=med_context,
    )
    if not rows and entities.get("active_ingredient"):
        rows = get_matching_drugs_for_ingredient(
            entities["active_ingredient"], set(), ctx, max_results=max_results + offset + 2
        )
        if requested_form:
            rows = [r for r in rows if row_matches_requested_form(r, requested_form)]
    if not rows and entities.get("confidence", 0) >= 0.35:
        enriched = llm_extract_query_entities(trade)
        if enriched.get("active_ingredient"):
            rows = get_matching_drugs_for_ingredient(
                enriched["active_ingredient"], set(), ctx, max_results=max_results + offset + 2
            )
            if requested_form:
                rows = [r for r in rows if row_matches_requested_form(r, requested_form)]
    if offset:
        rows = rows[offset:]
    return [row_to_pharmacy_record(r) for r in rows[:max_results]]


def _source_strength_variants(drug_name: str, ctx: PatientContext) -> Set[str]:
    rows = search_drugs_by_name(drug_name, ctx, max_results=12, relaxed=True)
    strengths: Set[str] = set()
    for row in rows:
        text = " ".join(str(row.get(k, "") or "") for k in (INGREDIENT_COL, "dosage_clean", "name_en"))
        strengths |= extract_strengths(text)
    return {s for s in strengths if s}


def search_product_queries(
    query: str,
    ctx: PatientContext,
    query_type: str,
    history: list,
) -> tuple:
    """
    Unified product/substitute search.
    Returns (medications, source_records, response_text_or_none, needs_clarification).
    """
    effective_query, med_context, is_refinement, requested_form = resolve_conversation_query(query, history)
    if is_refinement:
        query_type = med_context.query_intent or query_type

    offset = _search_offset_from_history(history) if is_show_more_request(query) else 0
    max_results = MAX_DRUG_CARDS_MORE if offset else MAX_DRUG_CARDS

    multi_names = split_multi_drug_names(effective_query)
    entities = llm_extract_query_entities(effective_query)
    if not entities.get("trade_name"):
        entities["trade_name"] = extract_drug_name_from_query(effective_query) or med_context.drug_name

    if not multi_names:
        trade = entities.get("trade_name") or extract_drug_name_from_query(effective_query)
        if not trade and entities.get("confidence", 0) < 0.35:
            if is_ambiguous_followup(query) and not med_context.is_active():
                return [], [], "ممكن تقول اسم الدواء اللي بتسأل عنه؟", True
            if not trade and not entities.get("active_ingredient") and not is_refinement:
                return [], [], "ممكن تقول اسم الدواء أو المادة الفعالة اللي بتدور عليها؟", True

    if query_type == "substitute":
        drug_name = entities.get("trade_name") or extract_drug_name_from_query(effective_query) or med_context.drug_name or effective_query
        strength_in_query = extract_strengths(effective_query) or med_context.strengths
        strengths = _source_strength_variants(drug_name, ctx)
        if (
            len(strengths) > 1
            and not strength_in_query
            and not offset
            and not is_refinement
            and not med_context.volume_ml
        ):
            opts = ", ".join(sorted(strengths)[:6])
            return [], [], f"الدواء متوفر بتركيزات مختلفة ({opts}) — محتاج أنهي تركيز؟", True
        source_records, sub_records = search_substitutes(
            effective_query,
            ctx,
            max_results=max_results,
            offset=offset,
            requested_form=requested_form or med_context.form_key,
            med_context=med_context,
        )
        if not source_records and entities.get("active_ingredient"):
            ing_rows = get_matching_drugs_for_ingredient(
                entities["active_ingredient"], set(), ctx, max_results=1
            )
            if ing_rows:
                if is_generic_ingredient(ing_rows[0].get(INGREDIENT_COL, "")):
                    return [], [], "مش قادر أحدد المادة الفعالة بدقة — ممكن تقول اسم الدواء أو التركيز؟", True
                source_records, sub_records = search_substitutes_from_row(
                    ing_rows[0], ctx, max_results=max_results, offset=offset
                )
        if not source_records:
            if entities.get("confidence", 0) < 0.35 and not entities.get("active_ingredient") and not med_context.is_active():
                return [], [], "ممكن تقول اسم الدواء اللي عايز بديل ليه؟", True
            return [], [], NOT_FOUND_MSG, False
        if not sub_records:
            src_name = source_records[0].get("name_ar") or source_records[0].get("name_en", "")
            return [], source_records, NO_SUBSTITUTE_MSG, False
        return sub_records, source_records, None, False

    # Multi-drug price/info queries
    if multi_names:
        all_meds: List[Dict[str, Any]] = []
        sections: List[str] = []
        for name in multi_names:
            sub_entities = llm_extract_query_entities(name)
            if not sub_entities.get("trade_name"):
                sub_entities["trade_name"] = name
            found = _lookup_single_drug(
                name,
                ctx,
                sub_entities,
                max_results=max_results,
                offset=offset,
                requested_form=requested_form,
                require_form=bool(requested_form),
                med_context=med_context,
            )
            label = name.strip()
            if found:
                sections.append(f"**{label}**")
                all_meds.extend(found)
            else:
                sections.append(f"**{label}**: {NOT_FOUND_MSG}")
        text = "\n".join(sections) if sections else NOT_FOUND_MSG
        return all_meds, [], text, False

    found = _lookup_single_drug(
        effective_query,
        ctx,
        entities,
        max_results=max_results,
        offset=offset,
        query_type=query_type,
        requested_form=requested_form,
        require_form=bool(requested_form),
        med_context=med_context,
    )
    if found:
        return found, [], None, False
    if requested_form:
        return [], [], FORM_NOT_FOUND_MSG, False
    if entities.get("confidence", 0) < 0.35 and not entities.get("active_ingredient") and not is_refinement:
        return [], [], "ممكن تقول اسم الدواء أو المادة الفعالة اللي بتدور عليها؟", True
    return [], [], NOT_FOUND_MSG, False


def search_substitutes_from_row(
    source_row: dict,
    ctx: PatientContext,
    max_results: int = MAX_DRUG_CARDS,
    offset: int = 0,
) -> tuple:
    _init_retrieval_engine()
    if retrieval_engine is None:
        return [row_to_pharmacy_record(source_row)], []
    subs = retrieval_engine.find_substitutes(
        source_row=source_row,
        source_index=source_row.get("row_index", 0),
        row_filter=_trade_name_row_filter(ctx, relaxed=False),
        max_results=max_results + offset,
        caution_fn=caution_notes_for_context,
        ctx=ctx,
    )
    if offset:
        subs = subs[offset:]
    subs = subs[:max_results]
    return [row_to_pharmacy_record(source_row)], [row_to_pharmacy_record(r) for r in subs]


def search_substitutes(
    query: str,
    ctx: PatientContext,
    max_results: int = MAX_DRUG_CARDS,
    offset: int = 0,
    requested_form: str = "",
    med_context: Optional[MedicationSearchContext] = None,
) -> tuple:
    """Return (source_records, substitute_records) from database only."""
    lookup_query = extract_drug_name_from_query(query) or query
    if med_context and (med_context.trade_name or med_context.drug_name):
        lookup_query = med_context.trade_name or med_context.drug_name
    form_filter = requested_form or (med_context.form_key if med_context else "")
    source_rows = search_drugs_by_name(
        lookup_query,
        ctx,
        max_results=24,
        relaxed=False,
        requested_form=form_filter,
        require_form=bool(form_filter),
        med_context=med_context,
    )
    if not source_rows and (form_filter or (med_context and med_context.volume_ml)):
        source_rows = search_drugs_by_name(
            lookup_query,
            ctx,
            max_results=24,
            relaxed=True,
            requested_form=form_filter,
            require_form=False,
            med_context=med_context,
        )
        if form_filter:
            form_rows = [r for r in source_rows if row_matches_requested_form(r, form_filter)]
            if form_rows:
                source_rows = form_rows
    if med_context and med_context.volume_ml and not source_rows:
        ing = ingredient_hint_for_trade(lookup_query)
        if ing:
            ing_rows = get_matching_drugs_for_ingredient(ing, set(), ctx, max_results=40)
            if form_filter:
                ing_rows = [r for r in ing_rows if row_matches_requested_form(r, form_filter)]
            brand = resolve_trade_alias(lookup_query)
            branded = [
                r for r in ing_rows
                if brand in normalize_text(r.get("name_en", "")) or brand in normalize_text(r.get("name_ar", ""))
            ]
            source_rows = branded or ing_rows
    primary = pick_primary_product(source_rows, query, med_context)
    if not primary:
        return [], []
    return search_substitutes_from_row(primary, ctx, max_results=max_results, offset=offset)


def build_patient_summary(ctx: PatientContext) -> str:
    def val(v, fallback="غير مذكور"):
        if v is None:
            return fallback
        if isinstance(v, list):
            return ", ".join(v) if v else fallback
        if isinstance(v, bool):
            return "نعم" if v else "لا"
        return str(v).strip() or fallback

    return f"""ملخص صيدلي (من الوكيل الرئيسي أو المحادثة):
- العمر: {val(ctx.age)}
- الجنس: {ctx.sex}
- حمل/رضاعة: {val(ctx.pregnant)}/{val(ctx.breastfeeding)}
- الحساسية: {val(ctx.allergies) if ctx.allergies_asked or ctx.allergies else "غير مؤكدة"}
- الأمراض المزمنة: {val(ctx.chronic_conditions)}
- سؤال المريض: {ctx.complaint_text[:120]}
"""


# ══════════════════════════════════════════════════════════════════════════════
# ⑨ MAIN RAG FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def sanitize_visible_text(text: str) -> str:
    return strip_hallucinated_drug_content(sanitize_medical_text(text))


def _workflow_from_plan(plan: ClinicalPlan, **overrides) -> dict:
    wf = dict(plan.workflow or {})
    wf.setdefault("task_status", "completed")
    wf.setdefault("return_to_orchestrator", False)
    wf.setdefault("escalation_reason", "none")
    wf.setdefault("missing_info", [])
    wf.update(overrides)
    return wf


def _build_workflow_response(
    text: str,
    plan: Optional[ClinicalPlan] = None,
    medications: Optional[list] = None,
    warnings: Optional[list] = None,
    ingredients: Optional[list] = None,
    **wf_overrides,
) -> WorkflowResponse:
    wf = _workflow_from_plan(plan or ClinicalPlan(), **wf_overrides)
    return WorkflowResponse(
        response=text,
        task_status=wf.get("task_status", "completed"),
        return_to_orchestrator=wf.get("return_to_orchestrator", False),
        escalation_reason=wf.get("escalation_reason", "none"),
        medications=medications or [],
        warnings=warnings or [],
        missing_info=wf.get("missing_info", []),
        ingredients=ingredients or [],
    )


def _guidance_suffix(plan: ClinicalPlan) -> str:
    if not plan.non_drug_advice:
        return ""
    return "\n\n💡 **ملاحظات:**\n" + "\n".join(f"• {a}" for a in plan.non_drug_advice)


def _response_text_early(text: str, records: list) -> str:
    return assemble_grounded_response(sanitize_visible_text(text), "", records, cards_only=True)


def _fallback_product_response(query: str, query_type: str, medications: list, source_name: str = "") -> str:
    if not medications:
        drug_q = extract_drug_name_from_query(query) or ""
        return NOT_FOUND_MSG if not drug_q else f"{NOT_FOUND_MSG} ({drug_q})"
    if query_type == "substitute":
        return f"دي بدائل متاحة من قاعدة البيانات لـ {source_name or 'الدواء'} — نفس المادة الفعالة والشكل."
    return "دي البيانات المتاحة من قاعدة الأدوية المصرية."


def pharmacy_consult(
    query: str,
    history: list = None,
    delegation: Optional[dict] = None,
) -> WorkflowResponse:
    """
    Pharmacy subagent entry point for ElevenLabs workflow delegation.

    delegation keys:
      - task_type: medication_recommendation | medication_info | drug_safety_check
      - patient_context: dict with age, sex, allergies, chronic_conditions, etc.
      - delegated_by: orchestrator identifier (optional)
    """
    history = history or []
    delegation = delegation or {}
    task_type = delegation.get("task_type", "pharmacy_consult")
    query_type = classify_query(query)

    scope_reason = detect_out_of_scope(query)
    if scope_reason:
        return out_of_scope_response(scope_reason)
    if query_type == "symptom_treatment":
        return out_of_scope_response("symptom")

    ctx = extract_context(query, history)
    ctx = apply_delegation_context(ctx, delegation)

    emergency = subagent_emergency_escalation(ctx)
    if emergency:
        return emergency

    is_product_query = query_type in PRODUCT_INFO_QUERY_TYPES
    medications: list = []
    direct_records: list = []
    source_records: list = []
    prebuilt_text: Optional[str] = None

    if is_product_query:
        medications, source_records, prebuilt_text, needs_clarify = search_product_queries(
            query, ctx, query_type, history
        )
        if needs_clarify:
            return _build_workflow_response(
                prebuilt_text or "ممكن توضّح اسم الدواء؟",
                task_status="needs_info",
                missing_info=[prebuilt_text] if prebuilt_text else [],
            )
        direct_records = medications + source_records
        if prebuilt_text and not medications:
            return _build_workflow_response(
                _response_text_early(prebuilt_text, medications),
                medications=medications,
                task_status="completed",
            )
    else:
        direct_rows = search_drugs_by_name(query, ctx, max_results=MAX_DRUG_CARDS, relaxed=False)
        direct_records = [row_to_pharmacy_record(r) for r in direct_rows]
        medications = direct_records
        if medications:
            text = _fallback_product_response(query, "product_info", medications, "")
            return _build_workflow_response(
                _response_text_early(text, medications),
                medications=medications,
                task_status="completed",
            )
        return _build_workflow_response(
            "ممكن تقول اسم الدواء أو المادة الفعالة اللي بتدور عليها؟",
            task_status="needs_info",
        )

    dataset_context = ""
    if direct_records:
        dataset_context = (
            "\n\nبيانات من قاعدة الأدوية (للمرجعية — لا تكررها في النص، البطاقات تعرضها):\n"
            + "\n".join(
                f"- {r['name_ar'] or r['name_en']} (صف {r['row']}): {r['active_ingredient']}, {r['price_egp']}"
                for r in direct_records[:8]
            )
        )

    gemini_messages = [
        {"role": "user" if m["role"] == "user" else "model",
         "parts": [{"text": m["content"]}]}
        for m in _trim_history_for_gemini(history)
    ]
    task_note = f"\nنوع المهمة: {task_type} | نوع السؤال: {query_type}"
    if is_product_query:
        task_note += "\n⚠️ استفسار منتج — لا تسأل عن سن/حمل/حساسية. لا تكتب تفاصيل أدوية في النص."
    delegated_by = delegation.get("delegated_by")
    if delegated_by:
        task_note += f" | مُفوَّض من: {delegated_by}"
    augmented = (
        build_patient_summary(ctx) +
        task_note +
        "\nسؤال الصيدلية:\n" + query.strip() +
        dataset_context +
        "\n\nأجب كصيدلي — إرشادات وملاحظات فقط (البطاقات تعرض الأدوية). ارجع WORKFLOW_RESPONSE."
    )
    gemini_messages.append({"role": "user", "parts": [{"text": augmented}]})

    llm_response, status = call_gemini(gemini_messages, system_prompt=SYSTEM_PROMPT)

    if status == "rate_limit":
        if medications:
            src = source_records[0]["name_ar"] if source_records else ""
            text = prebuilt_text or _fallback_product_response(query, query_type, medications, src)
            return _build_workflow_response(text, medications=medications, task_status="completed")
        if prebuilt_text:
            return _build_workflow_response(prebuilt_text, task_status="completed" if medications else "needs_info", medications=medications)
        return WorkflowResponse(
            response="معلش، النظام مشغول دلوقتي — استنى شوية.",
            task_status="needs_info",
        )

    if not llm_response:
        if medications:
            src = (source_records[0].get("name_ar") or source_records[0].get("name_en", "")) if source_records else ""
            text = prebuilt_text or _fallback_product_response(query, query_type, medications, src)
            return _build_workflow_response(text, medications=medications, task_status="completed")
        if prebuilt_text:
            return _build_workflow_response(prebuilt_text, medications=medications, task_status="completed")
        return WorkflowResponse(
            response="مش قادر أجيب إجابة دلوقتي — جرّب تاني بعد شوية.",
            task_status="needs_info",
        )

    plan = parse_clinical_plan(llm_response)
    plan.visible_text = sanitize_visible_text(plan.visible_text)

    if plan.workflow.get("return_to_orchestrator"):
        return _build_workflow_response(
            plan.visible_text or "هرجعك للمساعد الرئيسي.",
            plan=plan,
            medications=medications,
        )

    if plan.escalation_level == "urgent":
        return _build_workflow_response(
            plan.visible_text + "\n\n🚨 الحالة تستدعي تقييم طارئ من المساعد الرئيسي.",
            plan=plan,
            task_status="escalate",
            return_to_orchestrator=True,
            escalation_reason="emergency",
            medications=medications,
        )

    def _response_text(text: str, records: list, extra: str = "") -> str:
        return assemble_grounded_response(
            sanitize_visible_text(text + extra),
            "",
            records,
            cards_only=True,
        )

    # Product / substitute queries — database-driven, no safety triage
    if is_product_query or query_type == "substitute":
        guidance = plan.visible_text or prebuilt_text or _fallback_product_response(
            query,
            query_type,
            medications,
            (source_records[0].get("name_ar") or "") if source_records else "",
        )
        if query_type == "substitute" and not medications and source_records and not prebuilt_text:
            guidance += "\n\n⚠️ مفيش بدائل تانية بنفس المادة والشكل في الداتاسيت."
        if not medications and not source_records and prebuilt_text:
            guidance = prebuilt_text
        return _build_workflow_response(
            _response_text(guidance, medications, _guidance_suffix(plan)),
            plan=plan,
            medications=medications,
            warnings=dedupe_keep_order([w for r in medications for w in (r.get("warnings") or [])]),
            task_status="completed",
        )


# ── Pydantic request/response schemas (used by app.py) ───────────────────────
class PatientContextPayload(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None
    pregnant: Optional[bool] = None
    breastfeeding: Optional[bool] = None
    allergies: list = []
    chronic_conditions: list = []
    red_flags: list = []


class DelegationPayload(BaseModel):
    task_type: str = "pharmacy_consult"
    patient_context: Optional[PatientContextPayload] = None
    delegated_by: Optional[str] = "elevenlabs_orchestrator"


class ChatRequest(BaseModel):
    message: str
    history: list = []
    delegation: Optional[DelegationPayload] = None


class ChatResponse(BaseModel):
    response: str
    task_status: str = "completed"
    return_to_orchestrator: bool = False
    escalation_reason: str = "none"
    medications: list = []
    warnings: list = []
    missing_info: list = []
    ingredients: list = []


def delegation_to_dict(delegation: Optional[DelegationPayload]) -> Optional[dict]:
    if not delegation:
        return None
    d = delegation.model_dump()
    if d.get("patient_context"):
        d["patient_context"] = {k: v for k, v in d["patient_context"].items() if v is not None}
    return d


