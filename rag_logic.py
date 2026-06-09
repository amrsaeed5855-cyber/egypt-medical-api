"""
rag_logic.py — Egyptian Pharmacy Subagent for ElevenLabs Workflow Orchestration
==============================================================================
This service is a **pharmacy subagent**, not the primary medical assistant.

Architecture
------------
* ElevenLabs orchestrates the overall conversation (triage, booking, referrals).
* This subagent receives **delegated tasks** for medication guidance only.
* It returns **structured workflow responses** so ElevenLabs can resume control.

Scope: medication info, OTC recommendations, interactions, contraindications.
Out of scope: appointments, diagnosis, emergency triage, specialist referrals
→ signal `RETURN_TO_ORCHESTRATOR: true` back to ElevenLabs.

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
import asyncio
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
from rapidfuzz import process

# ──────────────────────────────────────────────────────────────────────────────
# FASTAPI IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


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
SYSTEM_PROMPT = """أنت صيدلي مصري خبير — **subagent صيدلة** داخل workflow ElevenLabs.
الوكيل الرئيسي (ElevenLabs) بيدير المحادثة العامة والتقييم الطبي والحجز.
أنت بتستقبل مهام مُفوَّضة للصيدلة فقط وترجع ردوداً منظمة للـ workflow.

## نطاق عملك (فقط)
- شرح الأدوية والمواد الفعالة والجرعات والتحذيرات والتفاعلات والبدائل.
- توصيات OTC آمنة بناءً على المعلومات المُمرَّرة من الوكيل الرئيسي.
- فحص موانع الاستعمال والتفاعلات الدوائية وملاءمة شكل الدواء.

## خارج نطاقك — أرجع التحكم للوكيل الرئيسي
- حجز مواعيد أو إدارة العيادات → `RETURN_TO_ORCHESTRATOR: true` + `ESCALATION_REASON: booking`
- تشخيص طبي نهائي أو تقييم أعراض من الصفر → `ESCALATION_REASON: diagnosis`
- حالات طوارئ أو علامات خطر → `ESCALATION_REASON: emergency`
- تحويل لطبيب مختص → `ESCALATION_REASON: referral`
لا تحاول تنفيذ هذه المهام — أخبر الوكيل الرئيسي يتولى الأمر.

## قواعد السلامة الدوائية
- **لا تفترض** أن المريض يتناول أدوية لمجرد وجود مرض مزمن — استخدم الملخص المُمرَّر أو اسأل سؤالاً واحداً لو ناقص.
- لا تكتب `───CLINICAL_PLAN───` إلا لو المهمة تتطلب اقتراح أدوية والمعلومات كافية لسلامة الدواء.
- متوصفش NSAIDs (إيبوبروفين، ديكلوفيناك، نابروكسين) مع ضغط أو سكر أو كلى.
- متوصفش فينيل إفرين (phenylephrine) أو سودوإيفيدرين (pseudoephedrine) مع ضغط عالي.
- متوصفش قطرة عين لأعراض عامة — للعين فقط.
- متوصفش حقن أو أدوية وريدية.
- متوصفش دواءين نفس الشغل.
- لو عند المريض مرض كبد: تأكد من نوعه قبل باراسيتامول أو أدوية كبدية.
- استخدم المواد الفعالة بالإنجليزي في البلوك.

## أسلوب التواصل
- إجابات قصيرة، واضحة، بالعامية المصرية.
- **ممنوع** "متقلقش خالص" أو "بسيطة" أو "إن شاء الله حاجة بسيطة".
- استخدم: "بناءً على المعلومات المتاحة، فيه أكتر من احتمال ومحتاجين تفاصيل أكتر."
- لا تشخّص بثقة مفرطة — التشخيص مسؤولية الوكيل الرئيسي.
- لا تسأل عن السن/الجنس/الأمراض المزمنة إلا لو ضروري لسلامة الدواء وغير موجود في الملخص.

## بلوك العلاج (عند اقتراح أدوية فقط)
───CLINICAL_PLAN───
INGREDIENTS: ingredient1, ingredient2
EXCLUDED_INGREDIENTS: ingredient_a
ESCALATION_LEVEL: none|caution|urgent
DIAGNOSIS_CONFIDENCE: low|medium|high
NON_DRUG_ADVICE: نصيحة1 | نصيحة2

## بلوك الرد للـ workflow (إلزامي في كل رد)
───WORKFLOW_RESPONSE───
TASK_STATUS: completed|needs_info|out_of_scope|escalate
RETURN_TO_ORCHESTRATOR: true|false
ESCALATION_REASON: none|booking|diagnosis|emergency|referral|specialist
MISSING_INFO: item1 | item2
"""

PHARMACIST_PROMPT = SYSTEM_PROMPT


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

PHARMACIST_QUERY_PATTERNS = [
    r"بيعمل\s*ايه", r"ايه\s*استخدام", r"ايه\s*فايده", r"ايه\s*فائده",
    r"جرعة", r"جرعات", r"تفاعل", r"بديل", r"side\s*effect",
    r"مضاعفات", r"تحذير", r"ينفع\s*اخده", r"ينفع\s*آخذه",
    r"what\s*does.*do", r"dosage", r"interaction",
]

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
    if re.search(r"(لا|مش|مفيش|لأ)\s*(مرضع|ترضع|رضاعة)", norm) or "بضح" not in norm:
        breastfeeding = False
    if re.search(r"(مرضع|برضع|باخد رضاعة)", norm):
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


def is_greeting(text: str) -> bool:
    greeting_words = ["هاي", "هلا", "سلام", "عامل ايه", "عامل اي", "ازيك", "اخبارك", "صباح", "مساء", "يعمعلم"]
    text_norm = normalize_text(text)
    if len(text_norm) < 15 or any(g in text_norm for g in greeting_words):
        if not any(normalize_text(s) in text_norm for s in COMMON_SYMPTOMS[:10]):
            return True
    return False


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


def pre_prescription_gate(ctx: PatientContext, plan: ClinicalPlan) -> Optional[list]:
    """Return missing medication-safety fields only (not full medical triage)."""
    if plan.is_conversational or not plan.ingredients:
        return None
    if plan.escalation_level == "urgent":
        return None

    missing = []
    if ctx.age is None:
        missing.append("السن (لسلامة الجرعة)")
    if not ctx.meds_confirmed and not ctx.meds_denied:
        missing.append("الأدوية الحالية (أو تأكيد عدم تناول أدوية)")
    if not ctx.allergies_asked and not ctx.allergies:
        missing.append("الحساسية الدوائية (أو تأكيد عدم وجود حساسية)")
    if ctx.sex == "female" and ctx.age and ctx.age >= 18 and ctx.pregnant is None:
        missing.append("الحمل/الرضاعة (لو ينطبق)")

    if "liver" in ctx.chronic_conditions:
        liver_meds = ["paracetamol", "acetaminophen", "loperamide"]
        if any(ing in liver_meds for ing in plan.ingredients) and not ctx.liver_assessed:
            missing.append("نوع مرض الكبد وآخر تحاليل وظائف كبد")

    return missing or None


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
        print("✅ Startup complete — SentenceTransformer loads on first drug search")
    else:
        print("✅ Startup complete — drug lookup via rapidfuzz (semantic search off)")

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
    if any(allergy.lower() in ai or ai in allergy.lower() for allergy in ctx.allergies):
        reasons.append("مستبعد بسبب حساسية مذكورة")
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


def semantic_candidate_indices(query_text: str, top_k: int = 40) -> list:
    """
    Encode query_text with SentenceTransformer (one short string, ~ms),
    then search the preloaded FAISS index for the top_k nearest neighbours.

    This is full semantic vector search — identical quality to the original
    notebook — but with zero startup cost because the index was built offline
    by build_index.py and loaded from disk at startup.

    Falls back to rapidfuzz text-matching if the FAISS index or model is
    unavailable (e.g. missing precomputed files).
    """
    if df.empty:
        return []

    # ── Semantic path (preferred) ────────────────────────────────────────────
    if index is not None and get_embed_model() is not None:
        try:
            # Encode only the short query string — NOT the dataset
            q = get_embed_model().encode([query_text]).astype("float32")
            faiss.normalize_L2(q)
            _scores, ids = index.search(q, min(top_k, index.ntotal))
            return [int(i) for i in ids[0] if i >= 0]
        except Exception as exc:
            print(f"⚠️ FAISS search failed ({exc}), falling back to rapidfuzz")

    # ── Fallback: rapidfuzz text matching ────────────────────────────────────
    try:
        hits = process.extract(query_text, df[INGREDIENT_COL].tolist(), limit=top_k)
        return [hit[2] for hit in hits if hit[1] > 0]
    except Exception:
        return list(range(min(len(df), top_k)))


def is_baby_drug(name_ar: str, name_en: str, age: Optional[int]) -> bool:
    if age is not None and age > 12:
        name_comb = (name_ar + " " + name_en).lower()
        return any(kw in name_comb for kw in BABY_KEYWORDS)
    return False


def get_matching_drugs_for_ingredient(
    ingredient: str, excluded: set, ctx: PatientContext, max_results: int = 2
) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    skip_terms = {"useful for cough", "useful for pain", "علاج", "دواء", "unknown"}
    if ingredient in skip_terms or len(ingredient) < 4:
        return []

    cand_ids   = semantic_candidate_indices(ingredient, top_k=60)
    cand_texts = [(idx, df.iloc[idx].get(INGREDIENT_COL, "")) for idx in cand_ids]
    fuzzy      = process.extract(ingredient, [t[1] for t in cand_texts], limit=30)

    base_map: Dict[str, list] = {}
    for hit in fuzzy:
        pos, score = hit[2], hit[1]
        if score < 85:
            continue
        idx  = cand_texts[pos][0]
        row  = df.iloc[idx]
        ai   = row.get(INGREDIENT_COL, "").strip().lower()
        name_en = row.get("name_en", "")
        name_ar = row.get("name_ar", "")
        if is_baby_drug(name_ar, name_en, ctx.age):                              continue
        if any(f in name_en.lower() for f in EXCLUDED_FORMS):                    continue
        if any(f in name_ar         for f in EXCLUDED_FORMS):                    continue
        if any(excl in ai           for excl in excluded):                        continue
        if not is_form_relevant_for_context(name_ar, name_en, ctx):               continue
        allowed, _ = screen_ingredient_safety(ai, ctx)
        if not allowed:                                                            continue
        base     = re.split(r"[+/\s\-]", ai)[0].strip()
        row_dict = row.to_dict()
        row_dict["row_id"]        = idx
        row_dict["safety_cautions"] = caution_notes_for_context(ai, ctx)
        base_map.setdefault(base, []).append((score, row_dict))

    if not base_map:
        return []
    base  = next(iter(base_map.keys()))
    items = sorted(base_map[base], key=lambda x: x[0], reverse=True)
    return [row_dict for _, row_dict in items[:max_results]]


def ingredient_purpose(ingredient: str) -> str:
    ing = ingredient.lower()
    for key, rule in INGREDIENT_SAFETY_RULES.items():
        if key in ing:
            return rule.get("purpose", "")
    return ""


def retrieve_drugs_structured(plan: ClinicalPlan, ctx: PatientContext) -> tuple:
    """Return (formatted_text, medications_list, warnings_list, ingredients_list)."""
    if not plan.ingredients:
        return "", [], [], []

    excluded, seen_ingredients = set(plan.excluded_ingredients), set()
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
            brands = [r.get("name_ar", "—") for r in rows if r.get("name_ar")]
            cautions = dedupe_keep_order([
                c for r in rows for c in (r.get("safety_cautions") or [])
            ])
            warnings.extend(cautions)
            medications.append({
                "active_ingredient": ing,
                "purpose": purpose,
                "brands_ar": brands,
                "warnings": cautions,
            })
            drug_blocks.append(f"🔹 **{ing.title()}**")
            if purpose:
                drug_blocks.append(f"   📌 **الاستخدام:** {purpose}")
            if brands:
                drug_blocks.append(f"   💊 **العلامات المصرية:** {' | '.join(brands)}")
            if cautions:
                drug_blocks.append(f"   ⚠️ **تحذيرات:** {' | '.join(cautions)}")
            drug_blocks.append("")

    if not found_any:
        return (
            "\n\n⚠️ مفيش أدوية متوفرة في القاعدة مطابقة للمواد الفعالة دي بعد فلاتر الأمان.",
            [], dedupe_keep_order(warnings), ingredients_out,
        )

    result = "\n\n---\n✅ **أدوية مناسبة:**\n\n" + "\n".join(drug_blocks)
    if plan.non_drug_advice:
        result += "\n\n📋 **نصائح إضافية:**\n" + "\n".join(f"• {a}" for a in plan.non_drug_advice)
    return result, medications, dedupe_keep_order(warnings), ingredients_out


def retrieve_drugs(plan: ClinicalPlan, ctx: PatientContext) -> str:
    text, _, _, _ = retrieve_drugs_structured(plan, ctx)
    return text


def format_differential(plan: ClinicalPlan) -> str:
    if not plan.differential:
        return ""
    lines = ["\n\n🔍 **احتمالات تشخيصية (بناءً على المعلومات المتاحة):**"]
    for item in plan.differential:
        if ":" in item and "%" in item:
            name, pct = item.rsplit(":", 1)
            lines.append(f"   • {name.strip()}: {pct.strip()}")
        else:
            lines.append(f"   • {item}")
    return "\n".join(lines)


def build_patient_summary(ctx: PatientContext) -> str:
    def val(v, fallback="غير مذكور"):
        if v is None:               return fallback
        if isinstance(v, list):     return ", ".join(v) if v else fallback
        if isinstance(v, bool):     return "نعم" if v else "لا"
        return str(v).strip() or fallback

    meds_status = "لا يتناول أدوية (مؤكد)" if ctx.meds_denied else (
        val(ctx.current_meds) if ctx.meds_confirmed else "غير مؤكد — اسأل صراحة"
    )
    return f"""ملخص الحالة المهيكل:
- العمر: {val(ctx.age)}
- الجنس: {ctx.sex}
- حمل/رضاعة: {val(ctx.pregnant or ctx.breastfeeding)}
- مدة الأعراض: {val(ctx.duration_text)}
- الحرارة: {val(ctx.fever_text)}
- الأعراض المستخرجة: {val(ctx.symptoms)}
- شدة الأعراض: {val(ctx.symptom_severity)}
- التهاب حلق: {val(ctx.sore_throat)}
- احتقان أنف: {val(ctx.nasal_congestion)}
- صعوبة تنفس: {val(ctx.breathing_difficulty)}
- الحساسية: {val(ctx.allergies) if ctx.allergies_asked or ctx.allergies else "غير مؤكدة — اسأل"}
- الأمراض المزمنة: {val(ctx.chronic_conditions)}
- الأدوية الحالية: {meds_status}
- تفاصيل الكبد: {val(ctx.liver_disease_details) if "liver" in ctx.chronic_conditions else "غير مطلوب"}
- علامات خطر: {val(ctx.red_flags)}
- نوع الكحة: {ctx.cough_type}
- دم/حرارة بالإسهال: {val(ctx.diarrhea_blood or ctx.diarrhea_fever)}
"""


# ══════════════════════════════════════════════════════════════════════════════
# ⑨ MAIN RAG FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
OVERCONFIDENT_PHRASES = [
    "متقلقش خالص", "متقلقش", "بسيطة", "حاجة بسيطة", "إن شاء الله حاجة بسيطة",
    "مفيش حاجة تقلق", "عادي خالص",
]


def sanitize_visible_text(text: str) -> str:
    out = text or ""
    for phrase in OVERCONFIDENT_PHRASES:
        out = out.replace(phrase, "محتاجين نجمع تفاصيل أكتر قبل الحكم")
    return out


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

    # Out-of-scope requests → return control to ElevenLabs orchestrator
    scope_reason = detect_out_of_scope(query)
    if scope_reason:
        return out_of_scope_response(scope_reason)

    ctx = extract_context(query, history)
    ctx = apply_delegation_context(ctx, delegation)

    emergency = subagent_emergency_escalation(ctx)
    if emergency:
        return emergency

    gemini_messages = [
        {"role": "user" if m["role"] == "user" else "model",
         "parts": [{"text": m["content"]}]}
        for m in _trim_history_for_gemini(history)
    ]
    task_note = f"\nنوع المهمة المُفوَّضة من ElevenLabs: {task_type}"
    delegated_by = delegation.get("delegated_by")
    if delegated_by:
        task_note += f"\nمُفوَّض من: {delegated_by}"
    augmented = (
        build_patient_summary(ctx) +
        task_note +
        "\nسؤال المهمة:\n" + query.strip() +
        "\n\nأجب كصيدلي subagent — ارجع بلوك WORKFLOW_RESPONSE في كل رد."
    )
    gemini_messages.append({"role": "user", "parts": [{"text": augmented}]})

    llm_response, status = call_gemini(gemini_messages, system_prompt=SYSTEM_PROMPT)
    if status == "rate_limit":
        return WorkflowResponse(
            response="معلش، النظام مشغول دلوقتي — استنى شوية.",
            task_status="needs_info",
            return_to_orchestrator=False,
        )
    if not llm_response:
        return WorkflowResponse(
            response="عذراً، حدث خطأ مؤقت — حاول تاني بعد شوية.",
            task_status="needs_info",
            return_to_orchestrator=False,
        )

    plan = parse_clinical_plan(llm_response)
    plan.visible_text = sanitize_visible_text(plan.visible_text)

    # LLM signalled out-of-scope or escalation via workflow block
    if plan.workflow.get("return_to_orchestrator"):
        return _build_workflow_response(
            plan.visible_text or "هرجعك للمساعد الرئيسي.",
            plan=plan,
        )

    if plan.escalation_level == "urgent":
        return _build_workflow_response(
            plan.visible_text + "\n\n🚨 الحالة تستدعي تقييم طارئ من المساعد الرئيسي.",
            plan=plan,
            task_status="escalate",
            return_to_orchestrator=True,
            escalation_reason="emergency",
        )

    # Medication info / safety check — no drug lookup needed
    if task_type in ("medication_info", "drug_safety_check") and plan.is_conversational:
        suffix = ""
        if plan.non_drug_advice:
            suffix = "\n\n📋 **ملاحظات:**\n" + "\n".join(f"• {a}" for a in plan.non_drug_advice)
        return _build_workflow_response(plan.visible_text + suffix, plan=plan)

    # Conversational pharmacy turn (needs more info, no drugs yet)
    if plan.is_conversational or not plan.ingredients:
        missing = plan.workflow.get("missing_info") or []
        suffix = ""
        if plan.non_drug_advice:
            suffix = "\n\n📋 **ملاحظات:**\n" + "\n".join(f"• {a}" for a in plan.non_drug_advice)
        status_out = "needs_info" if missing else "completed"
        return _build_workflow_response(
            plan.visible_text + suffix,
            plan=plan,
            task_status=status_out,
            missing_info=missing,
        )

    # Medication-safety gate before drug recommendations
    missing_safety = pre_prescription_gate(ctx, plan)
    if missing_safety:
        msg = (
            "بناءً على المعلومات المتاحة، محتاج تفاصيل لسلامة الدواء:\n"
            + "\n".join(f"• {m}" for m in missing_safety)
        )
        return _build_workflow_response(
            msg,
            plan=plan,
            task_status="needs_info",
            return_to_orchestrator=False,
            missing_info=missing_safety,
        )

    enforce_plan_safety_exclusions(plan, ctx)
    if not plan.ingredients:
        return _build_workflow_response(
            plan.visible_text +
            "\n\n⚠️ بناءً على الحالة المزمنة، مفيش أدوية OTC آمنة بدون استشارة طبيب.",
            plan=plan,
            task_status="escalate",
            return_to_orchestrator=True,
            escalation_reason="specialist",
        )

    drug_text, medications, warnings, ingredients = retrieve_drugs_structured(plan, ctx)
    suffix = []
    if "diabetes" in ctx.chronic_conditions or "hypertension" in ctx.chronic_conditions:
        suffix.append("⚠️ عندك مرض مزمن (سكر/ضغط) — دي أدوية مؤقتة فقط، تابع مع طبيب مختص.")
        warnings.append(suffix[-1])
    if "liver" in ctx.chronic_conditions:
        suffix.append("⚠️ عندك مرض كبد — استخدم الأدوية بحذر ولا تتجاوز الجرعات الموصى بها.")
        warnings.append(suffix[-1])
    if plan.diagnosis_confidence == "low":
        suffix.append("⚠️ الثقة في التوصية منخفضة — لو الأعراض زادت، ارجع للمساعد الرئيسي.")
    elif plan.escalation_level == "caution":
        suffix.append("⚠️ خد بالك وحافظ على متابعة الأعراض.")

    final = plan.visible_text + drug_text
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


# ══════════════════════════════════════════════════════════════════════════════
# ⑩ FASTAPI APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Egyptian Pharmacy Subagent API",
    description=(
        "Pharmacy subagent for ElevenLabs workflow orchestration. "
        "Handles medication guidance and returns structured workflow responses."
    ),
    version="2.0.0",
)

# Allow all origins — restrict this in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic request/response schemas ────────────────────────────────────────
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


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse("index.html")


def delegation_to_dict(delegation: Optional[DelegationPayload]) -> Optional[dict]:
    if not delegation:
        return None
    d = delegation.model_dump()
    if d.get("patient_context"):
        d["patient_context"] = {k: v for k, v in d["patient_context"].items() if v is not None}
    return d


@app.post("/chat", response_model=ChatResponse, summary="Pharmacy subagent consult (ElevenLabs delegation)")
@app.post("/pharmacy/consult", response_model=ChatResponse, include_in_schema=True)
async def chat_endpoint(body: ChatRequest):
    """
    ElevenLabs workflow subagent endpoint.
    Accepts delegated pharmacy tasks with optional pre-filled patient_context.
    Returns structured response for workflow orchestration.
    """
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    delegation = delegation_to_dict(body.delegation)

    def _run():
        with _rag_lock:
            return pharmacy_consult(body.message, body.history, delegation)

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_run),
            timeout=CHAT_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        print("❌ /chat timed out")
        return ChatResponse(
            response="معلش، الطلب أخد وقت طويل — استنى شوية وحاول تاني.",
            task_status="needs_info",
        )
    except Exception as e:
        print(f"❌ /chat unhandled error: {e}")
        return ChatResponse(
            response="عذراً، حدث خطأ مؤقت — حاول تاني بعد شوية.",
            task_status="needs_info",
        )

    return ChatResponse(**result.to_dict())


@app.get("/health", summary="Health check")
def health_check():
    """Returns service status and whether the drug index is loaded."""
    return {
        "status": "ok",
        "index_loaded": index is not None,
        "drug_count": len(df) if not df.empty else 0,
        "gemini_keys_loaded": len(GEMINI_API_KEYS),
        "semantic_search": ENABLE_SEMANTIC_SEARCH and not _embed_load_failed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
