"""
app.py — Production FastAPI backend for the Egyptian Medical AI Chatbot (RAG)
==============================================================================
Extracted from the original notebook. All AI/RAG/chatbot logic is preserved
exactly as written. Gradio and ngrok code has been removed. FastAPI endpoints
wrap the core `rag()` function.

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
SYSTEM_PROMPT = """أنت دكتور مصري عندك 15 سنة خبرة، ودود ومتفهم. بتتكلم مع مريض، وهدفك توصله لأسرع تشخيص وعلاج آمن.

قواعدك الصارمة:
- رحب بالمريض واسأل بطريقة لطيفة عن الأعراض.
- اجمع المعلومات الأساسية (السن، الجنس، الأمراض المزمنة، الحساسية، الحمل/الرضاعة للإناث فوق 18 سنة). لا تُلح على مدة الأعراض.
- متوصفش أبداً أدوية مضادة للالتهاب (NSAIDs) مثل إيبوبروفين أو ديكلوفيناك أو نابروكسين لمريض عنده ضغط أو سكر أو كلى.
- متوصفش حقن أو أدوية وريدية.
- متوصفش دواءين نفس الشغل.
- استخدم المواد الفعالة بالإنجليزي في البلوك النهائي.
- **مهم جداً: لما تكتب البلوك النهائي `───CLINICAL_PLAN───`، لازم تختار مواد فعالة تغطي كل الأعراض الرئيسية اللي ذكرها المريض.**
- لا تترك عرضاً بدون مادة فعالة. لا تكتب مواد وهمية.
- لو المريض بيحكي عن طفل، ركز على عمر الطفل ووزنه التقريبي.
- في حالة الاشتباه بمرض خطير، اكتب `ESCALATION_LEVEL: urgent` مع كتابة الأدوية المناسبة (لأن المريض قد يطلب علاجاً مؤقتاً). لا تفرغ قائمة INGREDIENTS.

ردودك بالعامية المصرية الدافية والمشجعة.

لما تبقى جاهز للعلاج، اختم بالبلوك ده:

───CLINICAL_PLAN───
INGREDIENTS: ingredient1, ingredient2, ingredient3
EXCLUDED_INGREDIENTS: ingredient_a, ingredient_b
ESCALATION_LEVEL: none|caution|urgent
DIAGNOSIS_CONFIDENCE: low|medium|high
NON_DRUG_ADVICE: نصيحة1 | نصيحة2 | نصيحة3
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
    non_drug_advice: list = field(default_factory=list)
    is_conversational: bool = True


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
    chronic_conditions: list = field(default_factory=list)
    current_meds: list = field(default_factory=list)
    complaint_text: str = ""
    red_flags: list = field(default_factory=list)
    cough_type: str = ""
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
    (r"ضيق نفس شديد|مش عارف اتنفس|نهجان شديد|اختناق", "ضيق نفس شديد"),
    (r"ألم صدر شديد|وجع صدر شديد|ضغط على الصدر", "ألم صدر شديد"),
    (r"فقدان وعي|اغماء|إغماء|مش واعي", "إغماء/فقدان وعي"),
    (r"تشنجات|convulsion|seizure", "تشنجات"),
    (r"قيء دم|ترجيع دم|دم في القيء", "قيء دم"),
    (r"براز أسود|دم في البراز|نزيف شرجي", "نزيف هضمي محتمل"),
    (r"ضعف مفاجئ|ميل في الوجه|لخبطة كلام|تلعثم مفاجئ", "أعراض عصبية حادة"),
    (r"طفح .* مع ضيق نفس|تورم اللسان|تورم الشفايف", "حساسية شديدة محتملة"),
    (r"حرارة فوق ?40|سخونية فوق ?40", "حرارة شديدة جدًا"),
    (r"جفاف شديد|مش بيشرب|قلة بول شديدة|مفيش بول", "جفاف شديد"),
    (r"ألم بطن شديد جدًا|بطن ناشفة|تيبس البطن", "ألم بطن حاد"),
    (r"حامل.*نزيف|نزيف.*حامل", "نزيف أثناء الحمل"),
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
    "ibuprofen":            {"avoid_in_pregnancy": True, "avoid_conditions": ["kidney", "ulcer", "hypertension", "diabetes"], "caution_conditions": ["heart", "asthma"], "min_age": 12},
    "diclofenac":           {"avoid_in_pregnancy": True, "avoid_conditions": ["kidney", "ulcer", "heart", "hypertension", "diabetes"], "caution_conditions": ["asthma"], "min_age": 14},
    "naproxen":             {"avoid_in_pregnancy": True, "avoid_conditions": ["kidney", "ulcer", "hypertension", "diabetes"], "caution_conditions": ["heart"], "min_age": 12},
    "pseudoephedrine":      {"avoid_in_pregnancy": True, "avoid_conditions": ["hypertension", "heart"], "caution_conditions": ["diabetes"], "min_age": 12},
    "loratadine":           {"min_age": 2},
    "cetirizine":           {"min_age": 2},
    "dextromethorphan":     {"min_age": 6},
    "loperamide":           {"min_age": 12, "avoid_conditions": ["ulcerative colitis"], "caution_conditions": ["liver"]},
    "paracetamol":          {"caution_conditions": ["liver"]},
    "acetaminophen":        {"caution_conditions": ["liver"]},
    "omeprazole":           {},
    "oral rehydration salts": {},
}

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
    allergies     = extract_list_after_keywords(full_text, ["حساسيه", "allergy", "allergies", "allergic to"])
    if has_negation_response(full_text) and not any(k in norm for k in ["حساسيه", "allergy"]):
        allergies = []
    current_meds       = extract_list_after_keywords(full_text, ["ادويه", "ادوية", "meds", "medications", "باخد", "باخد علاج"])
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
        allergies=allergies, chronic_conditions=chronic_conditions, current_meds=current_meds,
        complaint_text=query.strip(), red_flags=dedupe_keep_order(red_flags),
        cough_type=cough_type, diarrhea_blood=diarrhea_blood, diarrhea_fever=diarrhea_fever,
        dental_swelling=dental_swelling, dental_pus=dental_pus,
        is_caregiver=is_caregiver, child_age=child_age,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ INTAKE GATE
# ══════════════════════════════════════════════════════════════════════════════
REAL_URGENT_FLAGS = [
    "ضيق نفس شديد", "ألم صدر شديد", "إغماء", "تشنجات",
    "قيء دم", "نزيف", "شلل مفاجئ", "حرارة فوق 40"
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


def intake_gate(ctx: PatientContext, history: list, query: str) -> Optional[str]:
    if has_real_emergency(ctx):
        return "🚨 فيه علامات خطر حقيقية - لازم تروح الطوارئ فوراً."
    if is_greeting(query) and ctx.age is None and not ctx.symptoms:
        return "أهلاً بك! أنا دكتور مساعد. عرفني على أعراضك عشان أقدر أساعدك. قولي السن والجنس."
    if ctx.chronic_conditions:
        return None
    missing = []
    if ctx.age is None:        missing.append("السن")
    if ctx.sex == "unknown":   missing.append("الجنس (ذكر/أنثى)")
    if ctx.sex == "female" and ctx.age is not None and ctx.age >= 18:
        if ctx.pregnant is None and ctx.breastfeeding is None:
            preg_latest, breast_latest = parse_pregnancy_breastfeeding(query)
            if preg_latest  is not None: ctx.pregnant      = preg_latest
            if breast_latest is not None: ctx.breastfeeding = breast_latest
            if ctx.pregnant is None or ctx.breastfeeding is None:
                missing.append("هل أنت حامل أو مرضع؟")
    elif ctx.sex == "female" and ctx.age is not None and ctx.age < 18:
        if ctx.pregnant      is None: ctx.pregnant      = False
        if ctx.breastfeeding is None: ctx.breastfeeding = False
    if missing:
        return "عشان أساعدك، قولي " + " و ".join(missing) + "."
    return None


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


def call_gemini(messages: list):
    global _last_call_time, _gemini_key_index
    elapsed = time.time() - _last_call_time
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)

    if not GEMINI_API_KEYS:
        print("❌ No Gemini API keys configured")
        return None, "gemini_error"

    url     = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
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
def parse_clinical_plan(raw_text: str) -> ClinicalPlan:
    plan = ClinicalPlan()
    if "───CLINICAL_PLAN───" not in raw_text:
        plan.visible_text     = raw_text.strip()
        plan.is_conversational = True
        return plan

    plan.is_conversational = False
    visible_part, machine_part = raw_text.split("───CLINICAL_PLAN───", 1)
    plan.visible_text = visible_part.strip()

    def _extract(pattern: str, txt: str, default: str = "") -> str:
        m = re.search(pattern, txt)
        return m.group(1).strip() if m else default

    raw_ing  = _extract(r"INGREDIENTS:\s*([^\n]*)", machine_part)
    non_valid = {"useful for cough", "useful for pain", "علاج", "دواء", "unknown", " ", ""}
    plan.ingredients = [
        i.strip().lower()
        for i in raw_ing.split(",")
        if i.strip().lower() not in non_valid and len(i.strip()) > 3
    ]

    raw_excl = _extract(r"EXCLUDED_INGREDIENTS:\s*([^\n]*)", machine_part)
    plan.excluded_ingredients = {e.strip().lower() for e in raw_excl.split(",") if e.strip()}

    plan.escalation_level    = _extract(r"ESCALATION_LEVEL:\s*(\w+)",      machine_part, "none")
    plan.diagnosis_confidence = _extract(r"DIAGNOSIS_CONFIDENCE:\s*(\w+)", machine_part, "medium")

    raw_advice = _extract(r"NON_DRUG_ADVICE:\s*([^\n]*)", machine_part)
    if raw_advice:
        plan.non_drug_advice = [a.strip() for a in raw_advice.split("|") if a.strip()]

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


def retrieve_drugs(plan: ClinicalPlan, ctx: PatientContext) -> str:
    if not plan.ingredients:
        return ""
    excluded, seen_ingredients = set(plan.excluded_ingredients), set()
    drug_blocks, caution_blocks = [], []
    found_any = False

    for ing in plan.ingredients[:3]:
        if ing in seen_ingredients:
            continue
        rows = get_matching_drugs_for_ingredient(ing, excluded, ctx, max_results=2)
        if rows:
            found_any = True
            seen_ingredients.add(ing)
            drug_blocks.append(f"🔹 **المادة الفعالة: {ing.title()}**")
            for r in rows:
                drug_blocks.append(
                    f"   💊 **{r.get('name_ar', '—')}** | {r.get('name_en', '—')}\n"
                    f"      • (row {r.get('row_id', '?')})"
                )
                if r.get("safety_cautions"):
                    caution_blocks.append(f"   ⚠️ {r.get('name_ar', '')}: " + " | ".join(r["safety_cautions"]))
            drug_blocks.append("")

    if not found_any:
        return "\n\n⚠️ مفيش أدوية متوفرة في القاعدة مطابقة للمواد الفعالة دي بعد فلاتر الأمان."

    result = "\n\n---\n✅ **أدوية مناسبة:**\n\n" + "\n\n".join(drug_blocks)
    if caution_blocks:
        result += "\n" + "\n".join(caution_blocks)
    if plan.non_drug_advice:
        result += "\n\n📋 **نصائح إضافية:**\n" + "\n".join(f"• {a}" for a in plan.non_drug_advice)
    return result


def build_patient_summary(ctx: PatientContext) -> str:
    def val(v, fallback="غير مذكور"):
        if v is None:               return fallback
        if isinstance(v, list):     return ", ".join(v) if v else fallback
        if isinstance(v, bool):     return "نعم" if v else "لا"
        return str(v).strip() or fallback

    return f"""ملخص الحالة المهيكل:
- العمر: {val(ctx.age)}
- الجنس: {ctx.sex}
- حمل/رضاعة: {val(ctx.pregnant or ctx.breastfeeding)}
- مدة الأعراض (إذا ذكرت): {val(ctx.duration_text)}
- الحرارة: {val(ctx.fever_text)}
- الأعراض المستخرجة: {val(ctx.symptoms)}
- الحساسية: {val(ctx.allergies)}
- الأمراض المزمنة: {val(ctx.chronic_conditions)}
- الأدوية الحالية: {val(ctx.current_meds)}
- علامات خطر: {val(ctx.red_flags)}
- نوع الكحة: {ctx.cough_type}
- دم/حرارة بالإسهال: {val(ctx.diarrhea_blood or ctx.diarrhea_fever)}
"""


# ══════════════════════════════════════════════════════════════════════════════
# ⑨ MAIN RAG FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def rag(query: str, history: list) -> str:
    """
    Core RAG inference function.
    - Accepts the user query and full conversation history.
    - Returns a plain-text Arabic response.
    """
    # ── Check if the last bot message was an emergency and user now asks for
    #    temporary treatment anyway ──────────────────────────────────────────
    last_bot_message = ""
    for msg in reversed(history or []):
        if msg.get("role") == "assistant":
            last_bot_message = msg.get("content", "")
            break

    urgent_temp_keywords = ["مؤقت", "علاج مؤقت", "دواء مؤقت", "اقترح",
                             "مش قادر اروح", "حاجة تخفف", "بس عايز حاجة"]
    asks_for_temporary = any(kw in query.lower() for kw in urgent_temp_keywords)

    if "روح الطوارئ فوراً" in last_bot_message and asks_for_temporary:
        ctx = extract_context(query, history)
        gemini_messages = [
            {"role": "user" if m["role"] == "user" else "model",
             "parts": [{"text": m["content"]}]}
            for m in _trim_history_for_gemini(history)
        ]
        augmented = (
            build_patient_summary(ctx) +
            "\nسؤال المريض الحالي:\n" + query.strip() +
            "\n\nملاحظة: المريض يطلب علاجاً مؤقتاً على الرغم من أن الحالة قد تكون طارئة، "
            "ولكن لا يمكنه الذهاب للطوارئ الآن. اقترح أدوية مؤقتة آمنة مع تحذير قوي "
            "بأنها ليست بديلاً عن الطوارئ."
        )
        gemini_messages.append({"role": "user", "parts": [{"text": augmented}]})
        llm_response, _ = call_gemini(gemini_messages)
        if not llm_response:
            return "عذراً، حدث خطأ في الاتصال."
        plan = parse_clinical_plan(llm_response)
        if plan.is_conversational or not plan.ingredients:
            return "عذراً، لم أستطع اقتراح علاج مؤقت مناسب. يرجى التوجه للطوارئ فوراً."
        emergency_warning = (
            "🚨 **تنبيه خطير:** حالتك تستدعي الطوارئ فوراً. "
            "الأدوية التالية هي حل مؤقت فقط حتى تتمكن من الوصول للمستشفى.\n\n"
        )
        return emergency_warning + plan.visible_text + retrieve_drugs(plan, ctx)

    # ── Standard flow ────────────────────────────────────────────────────────
    ctx  = extract_context(query, history)
    gate = intake_gate(ctx, history, query)
    if gate:
        return gate

    # Build conversation for Gemini
    gemini_messages = [
        {"role": "user" if m["role"] == "user" else "model",
         "parts": [{"text": m["content"]}]}
        for m in _trim_history_for_gemini(history)
    ]
    augmented = build_patient_summary(ctx) + "\nسؤال المريض الحالي:\n" + query.strip()
    gemini_messages.append({"role": "user", "parts": [{"text": augmented}]})

    llm_response, status = call_gemini(gemini_messages)
    if status == "rate_limit":
        return "معلش، النظام مشغول دلوقتي — استنى شوية."
    if not llm_response:
        return "عذراً، حدث خطأ في الاتصال."

    plan = parse_clinical_plan(llm_response)

    # Urgent escalation — emergency redirect only
    if plan.escalation_level == "urgent":
        return f"{plan.visible_text}\n\n🚨 **روح الطوارئ فوراً.**"

    # Conversational turn (no drug prescription yet)
    if plan.is_conversational or not plan.ingredients:
        if plan.non_drug_advice:
            return plan.visible_text + "\n\n📋 **نصائح:**\n" + "\n".join(f"• {a}" for a in plan.non_drug_advice)
        return plan.visible_text

    # Full drug recommendation
    suffix = []
    if "diabetes" in ctx.chronic_conditions or "hypertension" in ctx.chronic_conditions:
        suffix.append("⚠️ عندك مرض مزمن (سكر/ضغط) - دي مجرد أدوية مؤقتة، لازم تروح طبيب مختص عشان تتابع حالتك الأساسية.")
    if plan.diagnosis_confidence == "low":
        suffix.append("⚠️ الثقة التشخيصية منخفضة — لو الأعراض زادت، اكشف.")
    elif plan.escalation_level == "caution":
        suffix.append("⚠️ خد بالك وحافظ على متابعة الأعراض.")

    final = plan.visible_text + retrieve_drugs(plan, ctx)
    if suffix:
        final += "\n\n" + "\n".join(suffix)
    return final


# ══════════════════════════════════════════════════════════════════════════════
# ⑩ FASTAPI APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Egyptian Medical AI API",
    description="RAG-powered Arabic medical chatbot backed by Gemini and a local drug database.",
    version="1.0.0",
)

# Allow all origins — restrict this in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic request/response schemas ────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list = []   # list of {"role": "user"|"assistant", "content": "..."}


class ChatResponse(BaseModel):
    response: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse("index.html")


@app.post("/chat", response_model=ChatResponse, summary="Send a message and receive an AI response")
async def chat_endpoint(body: ChatRequest):
    """
    Accepts a user message and optional conversation history.
    Returns an AI-generated medical response in Egyptian Arabic.
    """
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    def _run_rag():
        with _rag_lock:
            return rag(body.message, body.history)

    try:
        response_text = await asyncio.wait_for(
            asyncio.to_thread(_run_rag),
            timeout=CHAT_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        print("❌ /chat timed out")
        return ChatResponse(
            response="معلش، الطلب أخد وقت طويل — استنى شوية وحاول تاني."
        )
    except Exception as e:
        print(f"❌ /chat unhandled error: {e}")
        return ChatResponse(
            response="عذراً، حدث خطأ مؤقت — حاول تاني بعد شوية."
        )

    return ChatResponse(response=response_text)


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
