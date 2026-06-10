"""
rag_logic.py — Egyptian Pharmacy Subagent for ElevenLabs Workflow Orchestration
==============================================================================
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
  2. Load faiss.index with faiss.read_index()       (~instant, file I/O)
  3. Load SentenceTransformer weights               (~5–15 s, once)

It does NOT call embed_model.encode() on the dataset — ever.
Per-request cost: encode one short query string (~1–5 tokens).

Run with:
    uvicorn app:app --host 0.0.0.0 --port $PORT

Required files next to app.py (generate with build_index.py):
    egypt_drugs_cleaned_utf8.csv
    faiss.index
"""

# ──────────────────────────────────────────────────────────────────────────────
# STANDARD IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import re
import time
from threading import Lock
import requests
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# ──────────────────────────────────────────────────────────────────────────────
# AI / RAG IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import faiss

from retrieval import DrugRetrievalEngine
from response_grounding import (
    assemble_grounded_response,
    sanitize_medical_text,
    strip_hallucinated_drug_content,
)
from trade_name_utils import (
    classify_query,
    extract_drug_name_from_query,
    resolve_trade_alias,
    strip_form_noise,
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
- شرح أدوية، مواد فعالة، تحذيرات، تفاعلات.
- اقتراح أدوية OTC آمنة من **قاعدة البيانات المصرية** فقط.
- إجابات قصيرة وواضحة بالعامية المصرية.
- لو المريض ذكر معلومة بنفسه (سن، أمراض مزمنة، أدوية، حساسية) متسألش عنها تاني.

## استفسارات المنتج (سعر / بديل / مادة فعالة / تفاصيل / توفر)
- **لا تسأل** عن سن أو حمل أو حساسية أو أدوية حالية.
- **لا تكتب** تفاصيل الأدوية (اسم، سعر، صف، مادة فعالة) في النص — النظام يعرضها في بطاقات منفصلة.
- اكتب **إرشادات وملاحظات فقط** (تحذيرات، نصائح استخدام، توصيات).
- البدائل يحددها النظام من الداتاسيت — لا تخترع منتجات.

## أعراض بسيطة OTC (صداع، كحة، حرارة...)
- اسأل **سؤال واحد أو اتنين بس** لو ناقص معلومة ضرورية.
- لا تسأل عن الحمل لرجل أو لمراهق ذكر.
- لا تكرر أسئلة أجاب عليها المريض في نفس المحادثة.

## ممنوع تماماً
- **لا تشخّص** — لا تذكر اسم مرض كتشخيص.
- لا تحجز مواعيد ولا تعمل triage طبي.
- لا تكتب بلوكات 💊 أو أرقام صفوف — البطاقات تعرض الدواء.
- لو المريض عايز تشخيص أو حجز أو طوارئ → `RETURN_TO_ORCHESTRATOR: true`.

## سلامة دوائية (عند اقتراح علاج لأعراض فقط)
- لا NSAIDs مع ضغط/سكر/كلى.
- لا فينيل إفرين/سودوإيفيدرين مع ضغط.
- استخدم المواد الفعالة **بالإنجليزي بدقة** في INGREDIENTS (مثل paracetamol مش "مسكن").

## بلوك الأدوية (لاقتراح علاج أعراض فقط — ليس لاستفسارات المنتج)
───CLINICAL_PLAN───
INGREDIENTS: paracetamol
EXCLUDED_INGREDIENTS: ibuprofen
NON_DRUG_ADVICE: نصيحة1 | نصيحة2

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
    ingredients: list = field(default_factory=list)
    excluded_ingredients: set = field(default_factory=set)
    escalation_level: str = "none"
    diagnosis_confidence: str = "medium"
    differential: list = field(default_factory=list)
    non_drug_advice: list = field(default_factory=list)
    is_conversational: bool = True
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
    duration_text: str = ""
    fever_text: str = ""
    symptoms: list = field(default_factory=list)
    allergies: list = field(default_factory=list)
    allergies_asked: bool = False
    chronic_conditions: list = field(default_factory=list)
    current_meds: list = field(default_factory=list)
    meds_confirmed: bool = False
    meds_denied: bool = False
    liver_disease_details: str = ""
    liver_assessed: bool = False
    complaint_text: str = ""
    red_flags: list = field(default_factory=list)
    cough_type: str = ""
    sore_throat: bool = False
    nasal_congestion: bool = False
    breathing_difficulty: bool = False
    symptom_severity: str = ""
    diarrhea_blood: bool = False
    diarrhea_fever: bool = False
    dental_swelling: bool = False
    dental_pus: bool = False
    is_caregiver: bool = False
    child_age: Optional[int] = None


# ══════════════════════════════════════════════════════════════════════════════
# ④ TEXT / NLP UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
AR_NUMS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

COMMON_SYMPTOMS = [
    "حرارة", "سخونية", "كحة", "كحه", "رشح", "احتقان", "التهاب حلق", "زكام",
    "إسهال", "اسهال", "ترجيع", "قيء", "غثيان", "مغص", "صداع", "دوخة",
    "ضيق نفس", "وجع صدر", "ألم صدر", "حرقان بول", "حرقان", "طفح", "هرش",
    "ألم بطن", "وجع بطن", "ألم معدة", "حموضة", "إمساك", "امساك", "ألم ظهر"
]

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

OPHTHALMIC_KEYWORDS = [
    "eye drop", "eye drops", "ophthalmic", "optical", "قطرة عين", "قطرة للعين",
    "قطرة عيون", "optofrine", "eye solution",
]

NASAL_KEYWORDS = [
    "nasal", "nose spray", "أنف", "انف", "بخاخ أنف", "بخاخ انف", "nasal spray",
]

EYE_SYMPTOMS = ["عين", "عيون", "احمرار العين", "حكة العين", "دموع", "رمد", "conjunctivitis"]

RESPIRATORY_SYMPTOMS = [
    "حرارة", "سخونية", "كحة", "كحه", "رشح", "احتقان", "التهاب حلق", "زكام", "صداع",
]

MEDS_DENIAL_PATTERNS = [
    r"مش باخد\s*(ادويه|ادوية|علاج)",
    r"مفيش\s*(ادويه|ادوية|علاج)",
    r"لا\s*باخدش",
    r"مش بتناول",
    r"مش باخد حاجه",
    r"مش باخد حاجة",
    r"مش على علاج",
    r"مش باخد ادويه",
]

MEDS_CONFIRM_PATTERNS = [
    r"باخد\s+", r"بتناول\s+", r"على علاج\s+", r"ادويتي\s+", r"ادوية\s+",
    r"meds?\s*:", r"medications?\s*:",
]

PRODUCT_INFO_QUERY_TYPES = frozenset({"product_info", "substitute"})

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
}

VALID_TASK_TYPES = {
    "medication_recommendation",
    "medication_info",
    "drug_safety_check",
    "pharmacy_consult",
}

LIVER_DETAIL_KEYWORDS = [
    "كبد دهني", "التهاب كبد", "تليف", "cirrhosis", "hepatitis", "فيروس",
    "تحاليل كبد", "alt", "ast", "وظائف كبد",
]

EXCLUDED_FORMS = [
    "vial", "ampoule", "injection", "infusion", "iv", "i.v", "suppository",
    "امبول", "حقن", "وريدي", "امبولة", "لبوس"
]

BABY_KEYWORDS = [
    "teething", "baby", "infant", "toddler", "child",
    "تسنين", "رضع", "أطفال", "طفل", "رضيع"
]


def normalize_text(text: str) -> str:
    text = (text or "").translate(AR_NUMS)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ى", "ي")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


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


def arabic_words_to_number(words: str) -> Optional[int]:
    word_map = {
        "واحد": 1, "واحده": 1, "واحدة": 1, "اثنان": 2, "اثنين": 2, "اتنين": 2,
        "ثلاثة": 3, "تلاته": 3, "تلاتة": 3, "اربع": 4, "اربعة": 4, "اربعه": 4,
        "أربع": 4, "أربعة": 4, "خمسة": 5, "خمسه": 5, "ستة": 6, "سته": 6,
        "سبعة": 7, "سبعه": 7, "ثمانية": 8, "ثمانيه": 8, "تسعة": 9, "تسعه": 9,
        "عشرة": 10, "عشره": 10,
    }
    for w, num in word_map.items():
        if w in words:
            return num
    return None


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


def extract_duration(text: str) -> str:
    text = normalize_text(text)
    m = re.search(r"من\s+([^\s]+)\s+(يوم|ساعة|ساعه|ساعات|ايام|أيام|دقيقة|دقائق)", text)
    if m:
        num_word, unit = m.group(1), m.group(2)
        num = int(num_word) if num_word.isdigit() else arabic_words_to_number(num_word)
        if num:
            if "دقيق" in unit: return f"{num} دقيقة"
            if "ساع" in unit:  return f"{num} ساعة"
            return f"{num} يوم"
    m = re.search(r"(\d+|واحد|اثنين|اتنين|ثلاثة|تلاته|اربع|اربعة)\s+(يوم|ساعة|دقيقة|دقائق)", text)
    if m:
        num_word, unit = m.group(1), m.group(2)
        num = int(num_word) if num_word.isdigit() else arabic_words_to_number(num_word)
        if num:
            if "دقيق" in unit: return f"{num} دقيقة"
            if "ساع" in unit:  return f"{num} ساعة"
            return f"{num} يوم"
    if re.search(r"يومين", text):                       return "يومين"
    if re.search(r"دلوقتي|النهارده|لسه من شويه", text): return "أقل من يوم"
    if re.search(r"امبارح|أمس", text):                  return "يوم"
    return ""


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


def extract_symptoms(text: str) -> list:
    norm = normalize_text(text)
    return [s for s in COMMON_SYMPTOMS if normalize_text(s) in norm]


def extract_cough_type(text: str) -> str:
    if re.search(r"ببلغم|معاها بلغم|كحة بلغم|كحه بلغم", text, re.IGNORECASE):
        return "wet"
    if re.search(r"جافه|جافة|ناشفة|كحة جافه|كحه ناشفه", text, re.IGNORECASE):
        return "dry"
    return "unknown"


def extract_diarrhea_flags(text: str):
    blood = bool(re.search(r"دم في البراز|براز فيه دم|دم مع البراز", text, re.IGNORECASE))
    fever = bool(re.search(r"حراره|سخونيه", text, re.IGNORECASE))
    return blood, fever


def extract_dental_flags(text: str):
    swelling = bool(re.search(r"تورم|ورم في اللثه|وجه وارم", text, re.IGNORECASE))
    pus      = bool(re.search(r"صديد|ريحة كريهه|إفرازات", text, re.IGNORECASE))
    return swelling, pus


def extract_meds_status(text: str) -> tuple:
    norm = normalize_text(text)
    if any(re.search(p, norm) for p in MEDS_DENIAL_PATTERNS):
        return [], True, True
    meds = extract_list_after_keywords(
        text, ["ادويه", "ادوية", "meds", "medications", "باخد", "باخد علاج", "بتناول"],
        check_negation=False,
    )
    confirmed = bool(meds) or any(re.search(p, norm) for p in MEDS_CONFIRM_PATTERNS)
    if confirmed and not meds and has_negation_response(text):
        return [], True, True
    return meds, confirmed, False


def extract_liver_details(text: str) -> tuple:
    norm = normalize_text(text)
    if "liver" not in norm and "كبد" not in norm:
        return "", False
    details = []
    for kw in LIVER_DETAIL_KEYWORDS:
        if normalize_text(kw) in norm:
            details.append(kw)
    assessed = bool(details) or bool(re.search(r"تحاليل|alt|ast|وظائف", norm))
    return "، ".join(dedupe_keep_order(details)), assessed


def extract_respiratory_assessment(text: str) -> dict:
    norm = normalize_text(text)
    return {
        "sore_throat": bool(re.search(r"التهاب حلق|وجع حلق|حلق يوجع|حلق ملتهب", norm)),
        "nasal_congestion": bool(re.search(r"احتقان|انسداد انف|رشح|زكام|انسداد الأنف", norm)),
        "breathing_difficulty": bool(re.search(r"ضيق نفس|صعوبة تنفس|مش قادر اتنفس", norm)),
        "symptom_severity": (
            "شديدة" if re.search(r"شديد|جدا|مش قادر|تعبان اوي", norm)
            else "متوسطة" if re.search(r"متوسط|عادي", norm)
            else "خفيفة" if re.search(r"خفيف|بسيط", norm)
            else ""
        ),
    }


def has_eye_symptoms(ctx: PatientContext) -> bool:
    combined = normalize_text(" ".join(ctx.symptoms) + " " + ctx.complaint_text)
    return any(normalize_text(s) in combined for s in EYE_SYMPTOMS)


def has_respiratory_symptoms(ctx: PatientContext) -> bool:
    combined = normalize_text(" ".join(ctx.symptoms))
    return any(normalize_text(s) in combined for s in RESPIRATORY_SYMPTOMS)


def is_ophthalmic_product(name_ar: str, name_en: str) -> bool:
    combined = (name_ar + " " + name_en).lower()
    return any(kw in combined for kw in OPHTHALMIC_KEYWORDS)


def is_nasal_product(name_ar: str, name_en: str) -> bool:
    combined = (name_ar + " " + name_en).lower()
    return any(kw in combined for kw in NASAL_KEYWORDS)


def is_form_relevant_for_context(name_ar: str, name_en: str, ctx: PatientContext) -> bool:
    if is_ophthalmic_product(name_ar, name_en):
        return has_eye_symptoms(ctx)
    if is_nasal_product(name_ar, name_en):
        if has_respiratory_symptoms(ctx):
            return True
        return ctx.nasal_congestion or any(s in ctx.symptoms for s in ["رشح", "احتقان", "زكام"])
    return True


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
    if pc.get("symptoms"):
        ctx.symptoms = dedupe_keep_order(list(pc["symptoms"]))
    if pc.get("allergies") is not None:
        ctx.allergies = list(pc["allergies"])
        ctx.allergies_asked = True
    if pc.get("chronic_conditions"):
        ctx.chronic_conditions = dedupe_keep_order(list(pc["chronic_conditions"]))
    if pc.get("current_medications") is not None:
        meds = pc["current_medications"]
        if isinstance(meds, list):
            ctx.current_meds = meds
            ctx.meds_confirmed = True
            ctx.meds_denied = len(meds) == 0
        elif isinstance(meds, str) and meds.strip().lower() in ("none", "لا", "مفيش"):
            ctx.meds_denied = True
            ctx.meds_confirmed = True
    if pc.get("duration"):
        ctx.duration_text = str(pc["duration"])
    if pc.get("fever"):
        ctx.fever_text = str(pc["fever"])
    if pc.get("liver_disease_details"):
        ctx.liver_disease_details = str(pc["liver_disease_details"])
        ctx.liver_assessed = True
    if pc.get("red_flags"):
        ctx.red_flags = dedupe_keep_order(list(pc["red_flags"]))
    return ctx


def out_of_scope_response(reason: str) -> WorkflowResponse:
    messages = {
        "booking": "حجز المواعيد مش من اختصاص الصيدلي — هرجعك للمساعد الرئيسي يحجزلك.",
        "diagnosis": "التشخيص الطبي مش من اختصاص الصيدلي — هرجعك للمساعد الرئيسي يقيّم حالتك.",
        "referral": "التحويل لطبيب مختص محتاج المساعد الرئيسي — هرجعك له دلوقتي.",
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
    is_caregiver = bool(re.search(r"ابني|بنتي|طفلي|العيل|البنت|الولد", norm))
    child_age = None
    if is_caregiver:
        m = re.search(r"عنده\s*(\d+)\s*سنه", norm)
        if m:
            child_age = int(m.group(1))
    sex = extract_sex(full_text)
    pregnant, breastfeeding = parse_pregnancy_breastfeeding(full_text)
    duration_text = extract_duration(full_text)
    fever_match   = re.search(r"(حراره\s*\d+(?:\.\d+)?|سخونيه\s*\d+(?:\.\d+)?|حراره|سخونيه|سخونية)", norm, re.IGNORECASE)
    fever_text    = fever_match.group(1).strip() if fever_match else ""
    allergies = extract_list_after_keywords(full_text, ["حساسيه", "allergy", "allergies", "allergic to"])
    allergies_asked = bool(allergies) or bool(re.search(r"حساسيه|allergy", norm))
    if has_negation_response(query) and re.search(r"حساسيه|allergy", normalize_text(query)):
        allergies = []
        allergies_asked = True
    current_meds, meds_confirmed, meds_denied = extract_meds_status(full_text)
    liver_details, liver_assessed = extract_liver_details(full_text)
    resp = extract_respiratory_assessment(full_text)
    chronic_conditions = extract_conditions(full_text)
    symptoms           = extract_symptoms(full_text)
    red_flags          = []
    for pattern, label in REAL_RED_FLAG_PATTERNS:
        if re.search(pattern, norm, re.IGNORECASE):
            red_flags.append(label)
    cough_type                    = extract_cough_type(full_text)
    diarrhea_blood, diarrhea_fever = extract_diarrhea_flags(full_text)
    dental_swelling, dental_pus   = extract_dental_flags(full_text)
    age = extract_age(full_text)
    if child_age and not age:
        age = child_age
    return PatientContext(
        age=age, sex=sex, pregnant=pregnant, breastfeeding=breastfeeding,
        duration_text=duration_text, fever_text=fever_text, symptoms=symptoms,
        allergies=allergies, allergies_asked=allergies_asked,
        chronic_conditions=chronic_conditions, current_meds=current_meds,
        meds_confirmed=meds_confirmed, meds_denied=meds_denied,
        liver_disease_details=liver_details, liver_assessed=liver_assessed,
        complaint_text=query.strip(), red_flags=dedupe_keep_order(red_flags),
        cough_type=cough_type, sore_throat=resp["sore_throat"],
        nasal_congestion=resp["nasal_congestion"],
        breathing_difficulty=resp["breathing_difficulty"],
        symptom_severity=resp["symptom_severity"],
        diarrhea_blood=diarrhea_blood, diarrhea_fever=diarrhea_fever,
        dental_swelling=dental_swelling, dental_pus=dental_pus,
        is_caregiver=is_caregiver, child_age=child_age,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ INTAKE GATE
# ══════════════════════════════════════════════════════════════════════════════
REAL_URGENT_FLAGS = [
    "ضيق نفس شديد", "ألم صدر", "إغماء", "تشنجات",
    "قيء دم", "نزيف", "شلل مفاجئ", "حرارة فوق 40",
    "تيبس الرقبة", "صداع شديد", "قيء مستمر", "ارتباك",
]


def has_real_emergency(ctx: PatientContext) -> bool:
    text = (ctx.complaint_text + " " + " ".join(ctx.symptoms)).lower()
    return any(flag in text for flag in REAL_URGENT_FLAGS) or bool(ctx.red_flags)


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


def pre_prescription_gate(
    ctx: PatientContext,
    plan: ClinicalPlan,
    query_type: str = "general",
) -> Optional[list]:
    """Safety checks before suggesting drugs — skipped for product/substitute queries."""
    if query_type in PRODUCT_INFO_QUERY_TYPES:
        return None
    if plan.is_conversational or not plan.ingredients:
        return None

    missing = []
    needs_age = any(
        ing in (i.lower() for i in plan.ingredients)
        for ing in ("ibuprofen", "diclofenac", "naproxen", "dextromethorphan", "loperamide")
    ) or ctx.is_caregiver

    if needs_age and ctx.age is None:
        missing.append("السن (للجرعة المناسبة)")

    has_nsaids = any(
        n in ing.lower()
        for ing in plan.ingredients
        for n in ("ibuprofen", "diclofenac", "naproxen")
    )
    if has_nsaids and not ctx.allergies_asked and not ctx.allergies:
        missing.append("عندك حساسية من أي دواء؟ (لو لأ قول 'مفيش حساسية')")

    if ctx.sex == "female" and ctx.age and 15 <= ctx.age <= 50 and ctx.pregnant is None:
        missing.append("حامل أو مرضع؟ (لو ينطبق)")

    if len(missing) > 2:
        missing = missing[:2]

    return missing or None


def filter_llm_missing_info(missing: list, query_type: str, ctx: PatientContext) -> list:
    """Strip triage questions from LLM output for product queries and males."""
    if query_type in PRODUCT_INFO_QUERY_TYPES:
        return []
    if not missing:
        return []
    blocked = ("حامل", "مرضع", "حساسية", "أدوية", "ادوية", "سن", "عمر", "age", "pregnan", "allerg", "meds")
    if ctx.sex == "male":
        blocked = blocked + ("حامل", "مرضع", "pregnan", "breast")
    filtered = [m for m in missing if not any(b in m.lower() for b in blocked)]
    return filtered[:2]


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

    if "───CLINICAL_PLAN───" not in text:
        plan.visible_text = text
        plan.is_conversational = True
        return plan

    plan.is_conversational = False
    visible_part, machine_part = text.split("───CLINICAL_PLAN───", 1)
    plan.visible_text = visible_part.strip()

    raw_ing = _extract_field(r"INGREDIENTS:\s*([^\n]*)", machine_part)
    non_valid = {"useful for cough", "useful for pain", "علاج", "دواء", "unknown", " ", ""}
    plan.ingredients = [
        i.strip().lower()
        for i in raw_ing.split(",")
        if i.strip().lower() not in non_valid and len(i.strip()) > 3
    ]

    raw_excl = _extract_field(r"EXCLUDED_INGREDIENTS:\s*([^\n]*)", machine_part)
    plan.excluded_ingredients = {e.strip().lower() for e in raw_excl.split(",") if e.strip()}

    plan.escalation_level = _extract_field(r"ESCALATION_LEVEL:\s*(\w+)", machine_part, "none")
    plan.diagnosis_confidence = _extract_field(r"DIAGNOSIS_CONFIDENCE:\s*(\w+)", machine_part, "medium")

    raw_advice = _extract_field(r"NON_DRUG_ADVICE:\s*([^\n]*)", machine_part)
    if raw_advice:
        plan.non_drug_advice = [a.strip() for a in raw_advice.split("|") if a.strip()]

    raw_diff = _extract_field(r"DIFFERENTIAL:\s*([^\n]*)", machine_part)
    if raw_diff:
        plan.differential = [d.strip() for d in raw_diff.split("|") if d.strip()]

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
    # ① Load CSV — pure pandas, instant
    df_raw = pd.read_csv(CSV_PATH).fillna("").astype(str)
    INGREDIENT_COL = "ingredient_clean" if "ingredient_clean" in df_raw.columns else "active_ingredient"
    if "combined" not in df_raw.columns:
        df_raw["combined"] = (
            df_raw.get("name_ar",      pd.Series([""] * len(df_raw))) + " " +
            df_raw.get("name_en",      pd.Series([""] * len(df_raw))) + " " +
            df_raw.get(INGREDIENT_COL, pd.Series([""] * len(df_raw)))
        )
    df = df_raw.reset_index(drop=True)
    del df_raw  # release duplicate DataFrame memory before loading index
    import gc; gc.collect()
    print(f"✅ CSV loaded — {len(df)} rows")

    # ② Load precomputed FAISS index — faiss.read_index(), no rebuild
    index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"✅ FAISS index loaded — {index.ntotal} vectors")

    if ENABLE_SEMANTIC_SEARCH:
        print("✅ CSV + index ready — hybrid retrieval (semantic loads on first query)")
    else:
        print("✅ CSV + index ready — hybrid retrieval (lexical-only)")

except FileNotFoundError as e:
    print(f"❌ Missing precomputed file: {e}")
    print("   Run build_index.py offline to generate faiss.index")
except Exception as e:
    print(f"❌ Startup error: {e}")


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
    if "loperamide" in ai and (ctx.diarrhea_blood or ctx.diarrhea_fever):
        reasons.append("يمنع لوبيراميد مع دم أو حرارة في الإسهال")
    if "dextromethorphan" in ai and ctx.cough_type == "wet":
        reasons.append("دا مضاد سعال - مش مناسب للكحة ببلغم")
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
    if ctx.diarrhea_blood or ctx.diarrhea_fever:
        notes.append("الإسهال مع دم/حرارة يستدعي طبيباً - لا تستخدم أدوية إسهال بدون استشارة")
    if ctx.cough_type == "wet" and "dextromethorphan" in ai:
        notes.append("للكحة ببلغم الأفضل طارد بلغم (guaifenesin) مش مضاد سعال")
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
    if not is_form_relevant_for_context(name_ar, name_en, ctx):
        return "form_context"
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
    for key in ("mechanism", "composition", "dosage_clean"):
        val = str(row_dict.get(key, "") or "").strip()
        if val and val.lower() not in ("nan", "unknown", ""):
            return val[:200]
    purpose = ingredient_purpose(ai)
    return purpose or ""


def row_to_pharmacy_record(row_dict: dict) -> dict:
    """Map a dataset row to a pharmacy record for API + display."""
    ai = (
        row_dict.get("active_ingredient")
        or row_dict.get("ingredient_clean")
        or row_dict.get(INGREDIENT_COL, "")
    )
    ai_str = str(ai).strip()
    price_val = row_dict.get("price_egp", "")
    price_note = ""
    if row_dict.get("price_corrected") in (True, "True", "true", 1):
        raw = row_dict.get("price_egp_raw")
        if raw not in (None, "", "nan") and str(raw) != str(price_val):
            price_note = f"(مُقدَّر — السعر الأصلي في المصدر: {raw})"
    if price_val and str(price_val).strip() not in ("", "nan"):
        try:
            price = f"{float(price_val):g} جنيه"
        except (ValueError, TypeError):
            price = str(price_val)
    else:
        price = "غير متوفر"
    if price_note:
        price = f"{price} {price_note}"
    form = row_dict.get("form") or row_dict.get("form_clean", "")
    dosage = row_dict.get("dosage") or row_dict.get("dosage_clean", "")
    return {
        "row": row_dict.get("row_id"),
        "row_index": row_dict.get("row_index"),
        "name_ar": row_dict.get("name_ar", ""),
        "name_en": row_dict.get("name_en", ""),
        "active_ingredient": ai_str,
        "price_egp": price,
        "price_estimated": bool(price_note),
        "form": form,
        "dosage": dosage,
        "dose": row_dict.get("dose", ""),
        "usage": _usage_description(row_dict, ai_str),
        "warnings": row_dict.get("safety_cautions") or [],
        "retrieval_score": row_dict.get("retrieval_score"),
    }


def format_drug_block(rec: dict) -> str:
    row_label = rec.get("row")
    lines = [f"💊 **{rec['name_ar'] or rec['name_en']}** `(صف {row_label})`"]
    if rec.get("name_en") and rec.get("name_ar"):
        lines.append(f"   • الاسم الإنجليزي: {rec['name_en']}")
    lines.append(f"   • المادة الفعالة: {rec['active_ingredient'] or '—'}")
    lines.append(f"   • السعر: {rec['price_egp']}")
    if rec.get("form"):
        lines.append(f"   • الشكل: {rec['form']}")
    if rec.get("dosage"):
        lines.append(f"   • التركيز/الجرعة: {rec['dosage']}")
    if rec.get("dose"):
        lines.append(f"   • الجرعة: {rec['dose']}")
    if rec.get("warnings"):
        lines.append(f"   • ⚠️ { ' | '.join(rec['warnings'])}")
    return "\n".join(lines)


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
    max_results: int = 5,
    relaxed: bool = False,
) -> List[Dict[str, Any]]:
    """Multi-stage trade-name lookup with query extraction."""
    _init_retrieval_engine()
    if retrieval_engine is None or retrieval_engine.empty:
        return []

    drug_name = extract_drug_name_from_query(query) or query
    row_filter = _trade_name_row_filter(ctx, relaxed=relaxed)

    results = retrieval_engine.match_by_trade_name(
        name=drug_name,
        row_filter=row_filter,
        max_results=max_results,
        caution_fn=caution_notes_for_context,
        ctx=ctx,
        normalize_fn=normalize_text,
        relaxed_filter=relaxed,
    )
    if not results and drug_name != query:
        results = retrieval_engine.match_by_trade_name(
            name=query,
            row_filter=row_filter,
            max_results=max_results,
            caution_fn=caution_notes_for_context,
            ctx=ctx,
            normalize_fn=normalize_text,
            relaxed_filter=relaxed,
        )
    return results


_VARIANT_PENALTY = ("extra", "cold", "flu", "sinus", "joint", "migraine", "baby", "infant", "advance", "actifast")


def pick_primary_product(rows: List[Dict[str, Any]], query: str) -> Optional[Dict[str, Any]]:
    """Pick the best source product — prefer plain formulations when unspecified."""
    if not rows:
        return None
    drug_name = extract_drug_name_from_query(query) or query
    norm_query = resolve_trade_alias(drug_name)
    stripped = strip_form_noise(norm_query)

    def score_row(row: dict) -> float:
        s = float(row.get("retrieval_score") or 0)
        name_en = normalize_text(row.get("name_en", ""))
        profile = len((row.get(INGREDIENT_COL) or "").split("+"))
        s += 0.12 / max(profile, 1)
        for mod in _VARIANT_PENALTY:
            if mod in name_en and mod not in norm_query:
                s -= 0.07
        if stripped and (name_en.startswith(stripped) or f" {stripped}" in f" {name_en}"):
            s += 0.15
        if "panadol 500" in name_en or "panadol advance" in name_en:
            if "بانادول" in normalize_text(query) or "panadol" in norm_query:
                if "extra" not in norm_query and "cold" not in norm_query:
                    s += 0.12
        return s

    return max(rows, key=score_row)


def search_substitutes(
    query: str,
    ctx: PatientContext,
    max_results: int = 5,
) -> tuple:
    """Return (source_records, substitute_records) from database only."""
    source_rows = search_drugs_by_name(query, ctx, max_results=8, relaxed=True)
    primary = pick_primary_product(source_rows, query)
    if not primary:
        return [], []
    source_rows = [primary]

    _init_retrieval_engine()
    if retrieval_engine is None:
        return [row_to_pharmacy_record(source_rows[0])], []

    source = source_rows[0]
    subs = retrieval_engine.find_substitutes(
        source_row=source,
        source_index=source.get("row_index", 0),
        row_filter=_trade_name_row_filter(ctx, relaxed=False),
        max_results=max_results,
        caution_fn=caution_notes_for_context,
        ctx=ctx,
    )
    return (
        [row_to_pharmacy_record(source)],
        [row_to_pharmacy_record(r) for r in subs],
    )


def retrieve_drugs_structured(plan: ClinicalPlan, ctx: PatientContext) -> tuple:
    """Return (formatted_text, medications_list, warnings_list, ingredients_list)."""
    if not plan.ingredients:
        return "", [], [], []

    excluded, seen_ingredients, seen_rows = set(plan.excluded_ingredients), set(), set()
    drug_blocks, medications, warnings, ingredients_out = [], [], [], []
    found_any = False

    for ing in plan.ingredients[:3]:
        if ing in seen_ingredients:
            continue
        rows = get_matching_drugs_for_ingredient(ing, excluded, ctx, max_results=3)
        if rows:
            found_any = True
            seen_ingredients.add(ing)
            ingredients_out.append(ing)
            purpose = ingredient_purpose(ing)
            if purpose:
                drug_blocks.append(f"🔹 **{ing.title()}** — {purpose}")
            else:
                drug_blocks.append(f"🔹 **{ing.title()}**")
            for r in rows:
                rid = r.get("row_id")
                if rid in seen_rows:
                    continue
                seen_rows.add(rid)
                rec = row_to_pharmacy_record(r)
                medications.append(rec)
                warnings.extend(rec.get("warnings") or [])
                drug_blocks.append(format_drug_block(rec))
            drug_blocks.append("")

    if not found_any:
        return (
            "\n\n⚠️ مفيش أدوية في الداتاسيت مطابقة بعد فلاتر الأمان.",
            [], dedupe_keep_order(warnings), ingredients_out,
        )

    result = "\n\n---\n📋 **من قاعدة الأدوية المصرية:**\n\n" + "\n".join(drug_blocks)
    if plan.non_drug_advice:
        result += "\n\n💡 **ملاحظات:**\n" + "\n".join(f"• {a}" for a in plan.non_drug_advice)
    return result, medications, dedupe_keep_order(warnings), ingredients_out


def build_patient_summary(ctx: PatientContext) -> str:
    def val(v, fallback="غير مذكور"):
        if v is None:
            return fallback
        if isinstance(v, list):
            return ", ".join(v) if v else fallback
        if isinstance(v, bool):
            return "نعم" if v else "لا"
        return str(v).strip() or fallback

    meds_status = "لا يتناول أدوية" if ctx.meds_denied else (
        val(ctx.current_meds) if ctx.meds_confirmed else "غير مؤكد"
    )
    return f"""ملخص صيدلي (من الوكيل الرئيسي أو المحادثة):
- العمر: {val(ctx.age)}
- الجنس: {ctx.sex}
- حمل/رضاعة: {val(ctx.pregnant)}/{val(ctx.breastfeeding)}
- الحساسية: {val(ctx.allergies) if ctx.allergies_asked or ctx.allergies else "غير مؤكدة"}
- الأمراض المزمنة: {val(ctx.chronic_conditions)}
- الأدوية الحالية: {meds_status}
- طلب المريض/الأعراض المذكورة: {val(ctx.symptoms) or ctx.complaint_text[:120]}
"""


# ══════════════════════════════════════════════════════════════════════════════
# ⑨ MAIN RAG FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def sanitize_visible_text(text: str) -> str:
    return strip_hallucinated_drug_content(sanitize_medical_text(text))


def enforce_plan_safety_exclusions(plan: ClinicalPlan, ctx: PatientContext) -> None:
    if "hypertension" in ctx.chronic_conditions:
        blocked = {"phenylephrine", "pseudoephedrine"}
        plan.ingredients = [i for i in plan.ingredients if i not in blocked and not any(b in i for b in blocked)]
        plan.excluded_ingredients.update(blocked)


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


def _fallback_product_response(query: str, query_type: str, medications: list, source_name: str = "") -> str:
    if not medications:
        drug_q = extract_drug_name_from_query(query) or ""
        return f"مش لاقي '{drug_q or 'الدواء'}' في قاعدة البيانات — تأكد من الاسم التجاري أو اكتبه بالإنجليزي."
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

    ctx = extract_context(query, history)
    ctx = apply_delegation_context(ctx, delegation)

    emergency = subagent_emergency_escalation(ctx)
    if emergency:
        return emergency

    is_product_query = query_type in PRODUCT_INFO_QUERY_TYPES
    medications: list = []
    direct_records: list = []
    source_records: list = []

    if query_type == "substitute":
        source_records, sub_records = search_substitutes(query, ctx, max_results=5)
        medications = sub_records
        direct_records = source_records + sub_records
    else:
        relaxed = is_product_query
        direct_rows = search_drugs_by_name(query, ctx, max_results=5 if is_product_query else 3, relaxed=relaxed)
        direct_records = [row_to_pharmacy_record(r) for r in direct_rows]
        medications = direct_records

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
            text = _fallback_product_response(query, query_type, medications, src)
            return _build_workflow_response(text, medications=medications, task_status="completed")
        return WorkflowResponse(
            response="معلش، النظام مشغول دلوقتي — استنى شوية.",
            task_status="needs_info",
        )

    if not llm_response:
        if medications:
            src = (source_records[0].get("name_ar") or source_records[0].get("name_en", "")) if source_records else ""
            text = _fallback_product_response(query, query_type, medications, src)
            return _build_workflow_response(text, medications=medications, task_status="completed")
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
    if is_product_query:
        guidance = plan.visible_text or _fallback_product_response(
            query,
            query_type,
            medications,
            (source_records[0].get("name_ar") or "") if source_records else "",
        )
        if query_type == "substitute" and not medications and source_records:
            guidance += "\n\n⚠️ مفيش بدائل تانية بنفس المادة والشكل في الداتاسيت."
        return _build_workflow_response(
            _response_text(guidance, medications, _guidance_suffix(plan)),
            plan=plan,
            medications=medications,
            warnings=dedupe_keep_order([w for r in medications for w in (r.get("warnings") or [])]),
            task_status="completed",
        )

    if task_type in ("medication_info", "drug_safety_check") and plan.is_conversational:
        return _build_workflow_response(
            _response_text(plan.visible_text, medications, _guidance_suffix(plan)),
            plan=plan,
            medications=medications,
            warnings=dedupe_keep_order([w for r in medications for w in (r.get("warnings") or [])]),
        )

    if plan.is_conversational or not plan.ingredients:
        missing = filter_llm_missing_info(plan.workflow.get("missing_info") or [], query_type, ctx)
        status_out = "needs_info" if missing else "completed"
        return _build_workflow_response(
            _response_text(plan.visible_text, medications, _guidance_suffix(plan)),
            plan=plan,
            task_status=status_out,
            missing_info=missing,
            medications=medications,
        )

    missing_safety = pre_prescription_gate(ctx, plan, query_type)
    if missing_safety:
        msg = "محتاج معلومة واحدة لسلامة الاقتراح:\n" + "\n".join(f"• {m}" for m in missing_safety)
        return _build_workflow_response(
            msg,
            plan=plan,
            task_status="needs_info",
            missing_info=missing_safety,
        )

    enforce_plan_safety_exclusions(plan, ctx)
    if not plan.ingredients:
        return _build_workflow_response(
            plan.visible_text + "\n\n⚠️ بناءً على الحالة المزمنة، مفيش أدوية OTC آمنة بدون استشارة طبيب.",
            plan=plan,
            task_status="escalate",
            return_to_orchestrator=True,
            escalation_reason="specialist",
        )

    drug_text, medications, warnings, ingredients = retrieve_drugs_structured(plan, ctx)
    suffix = []
    if "diabetes" in ctx.chronic_conditions or "hypertension" in ctx.chronic_conditions:
        suffix.append("⚠️ مرض مزمن (سكر/ضغط) — استشر طبيبك قبل الاستمرار.")
        warnings.append(suffix[-1])
    if "liver" in ctx.chronic_conditions:
        suffix.append("⚠️ مرض كبد — استخدم بحذر ولا تتجاوز الجرعة الموصى بها.")
        warnings.append(suffix[-1])

    final = assemble_grounded_response(
        plan.visible_text,
        drug_text,
        medications,
        safety_suffix=suffix or None,
        cards_only=True,
    )
    if not medications and drug_text:
        final = sanitize_visible_text(plan.visible_text) + drug_text
        if suffix:
            final += "\n\n" + "\n".join(suffix)

    return _build_workflow_response(
        final,
        plan=plan,
        medications=medications,
        warnings=dedupe_keep_order(warnings),
        ingredients=ingredients,
        task_status="completed",
    )


def rag(query: str, history: list = None, delegation: Optional[dict] = None) -> str:
    """Backward-compatible text-only wrapper for pharmacy_consult."""
    return pharmacy_consult(query, history, delegation).response


# ── Pydantic request/response schemas (used by app.py) ───────────────────────
class PatientContextPayload(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None
    pregnant: Optional[bool] = None
    breastfeeding: Optional[bool] = None
    symptoms: list = []
    allergies: list = []
    chronic_conditions: list = []
    current_medications: Optional[list] = None
    duration: Optional[str] = None
    fever: Optional[str] = None
    liver_disease_details: Optional[str] = None
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


