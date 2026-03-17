
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR   = Path(os.getenv("BASE_DIR",   str(Path(__file__).parent.parent)))
DATA_DIR   = BASE_DIR / "data"
DOCS_DIR   = Path(os.getenv("DOCS_DIR",   str(DATA_DIR / "legal_docs")))
CHROMA_DIR = Path(os.getenv("CHROMA_DB_DIR", str(DATA_DIR / "chroma_db")))
TEMP_DIR   = Path(os.getenv("TEMP_DIR",   str(DATA_DIR / "temp")))

for _d in [DOCS_DIR, CHROMA_DIR, TEMP_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL      = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash-lite")
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "1024"))
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))

EMBEDDING_MODEL  = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")


CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rural_legal_docs")


RETRIEVAL_K       = int(os.getenv("RETRIEVAL_K",    "4"))
RETRIEVAL_FETCH_K = int(os.getenv("RETRIEVAL_FETCH_K", "16"))  # fetch more, rerank to K
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE",     "500"))   # reduced for legal docs
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP",  "100"))   # more overlap


# FlashRank cross-encoder reranker (local, no API needed, ~50MB)
ENABLE_RERANKER   = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
RERANKER_MODEL    = os.getenv("RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")


WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE     = os.getenv("WHISPER_DEVICE",     "cuda")


TTS_MODEL  = os.getenv("TTS_MODEL",  "facebook/mms-tts-tel")
TTS_DEVICE = os.getenv("TTS_DEVICE", "gpu")

OCR_LANG = os.getenv("OCR_LANG", "tel+eng")


TEMP_MAX_AGE_SECONDS = int(os.getenv("TEMP_MAX_AGE_SECONDS", "3600"))


ENABLE_WEB_SEARCH      = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))

# Keywords that always trigger web search
WEB_SEARCH_TRIGGERS = [
    "latest", "current", "today", "recent", "2024", "2025", "2026",
    "new scheme", "new law", "amendment", "notification",
    "ఇప్పుడు", "తాజా", "కొత్త",
]


ENABLE_CALCULATOR = os.getenv("ENABLE_CALCULATOR", "true").lower() == "true"


LLM_MODEL_PATH    = os.getenv("LLM_MODEL_PATH", str(BASE_DIR / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"))
LLM_N_GPU_LAYERS  = int(os.getenv("LLM_N_GPU_LAYERS", "28"))
LLM_CONTEXT_LEN   = int(os.getenv("LLM_CONTEXT_LEN",  "4096"))
LLM_MAX_TOKENS    = int(os.getenv("LLM_MAX_TOKENS",   "512"))
LLM_TEMPERATURE   = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TOP_P         = float(os.getenv("LLM_TOP_P",       "0.9"))

# ── Translation (kept for fallback if Gemini is unavailable) ─────────────────
# NOTE: Primary translation is now done by Gemini (native multilingual).
# Helsinki-NLP models are used as offline fallback only.
TE_TO_EN_MODEL     = os.getenv("TE_TO_EN_MODEL", "Helsinki-NLP/opus-mt-mul-en")
EN_TO_TE_MODEL     = os.getenv("EN_TO_TE_MODEL", "Helsinki-NLP/opus-mt-en-mul")
TRANSLATION_DEVICE = os.getenv("TRANSLATION_DEVICE", "cpu")
TRANSLATION_MAX_LEN = int(os.getenv("TRANSLATION_MAX_LEN", "512"))
USE_GEMINI_TRANSLATION = os.getenv("USE_GEMINI_TRANSLATION", "true").lower() == "true"


SYSTEM_INSTRUCTION = """You are a legal assistant helping rural Indian villagers understand
their legal rights and government schemes. Be helpful, accurate, and use simple language.

Use ONLY the context provided from official legal documents to answer questions.
If the context does not contain the answer, say clearly:
"I don't have information about this in the documents. Please consult a local legal aid centre."

Do NOT make up laws, scheme amounts, or eligibility criteria.

Language rule: Detect the language of the user's question and respond in THAT SAME language.
If the question is in Telugu (తెలుగు), respond entirely in Telugu.
If the question is in English, respond in English.
Use simple, clear language that a rural villager can understand.

Answer based on context from legal documents:
{context}

Question: {question}

Answer:"""

LEGAL_SYSTEM_PROMPT = SYSTEM_INSTRUCTION
