
import logging
import os
import time
from pathlib import Path
from typing import Optional, List, Tuple

import chromadb
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from modules.config import (
    CHROMA_DIR, CHROMA_COLLECTION,
    EMBEDDING_MODEL, EMBEDDING_DEVICE,
    RETRIEVAL_K, RETRIEVAL_FETCH_K, CHUNK_SIZE, CHUNK_OVERLAP,
    ENABLE_WEB_SEARCH, WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_TRIGGERS,
    ENABLE_RERANKER, RERANKER_MODEL,
    ENABLE_CALCULATOR,
    GEMINI_API_KEY, GEMINI_MODEL, GEMINI_MAX_TOKENS, GEMINI_TEMPERATURE,
    LLM_MODEL_PATH, LLM_N_GPU_LAYERS, LLM_CONTEXT_LEN,
    LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_TOP_P,
    TE_TO_EN_MODEL, EN_TO_TE_MODEL, TRANSLATION_DEVICE, TRANSLATION_MAX_LEN,
    LEGAL_SYSTEM_PROMPT,
)
from modules.agents import run_calculator, run_web_search, should_web_search

logger = logging.getLogger(__name__)
# "gemini"  → Gemini API is working, use it for everything
# "mistral" → Gemini unavailable, use Mistral + Helsinki-NLP
_BACKEND: Optional[str] = None   # set once during first call to _init_backend()

#  Gemini singletons
_gemini_client  = None
_active_model   = GEMINI_MODEL

#  Mistral singleton 
_mistral_llm    = None

#  Helsinki-NLP singletons 
_te_to_en_pipe  = None   # Telugu → English
_en_to_te_pipe  = None   # English → Telugu

# ── Shared singletons
_embeddings     = None
_vectordb       = None
_reranker       = None

def _init_backend() -> str:
    """
    Probe Gemini with a lightweight test call.
    Returns "gemini" if it works, "mistral" otherwise.
    Called once and result cached in _BACKEND.
    """
    global _BACKEND, _gemini_client, _active_model

    #  Try Gemini 
    if GEMINI_API_KEY:
        try:
            try:
                from google import genai
                from google.genai import types
                _NEW_SDK = True
            except ImportError:
                import google.generativeai as genai
                _NEW_SDK = False

            if _NEW_SDK:
                client = genai.Client(api_key=GEMINI_API_KEY)
                # Auto-discover best available model
                _MODEL_CANDIDATES = [
                    "models/gemini-2.0-flash-lite",
                    "models/gemini-2.0-flash",
                    "models/gemini-1.5-flash",
                ]
                try:
                    available = {m.name for m in client.models.list()}
                    for candidate in _MODEL_CANDIDATES:
                        if candidate in available:
                            _active_model = candidate
                            break
                except Exception:
                    _active_model = "models/gemini-2.0-flash-lite"

                # Lightweight probe call
                from google.genai import types as _types
                client.models.generate_content(
                    model=_active_model,
                    contents="Reply with the single word: OK",
                    config=_types.GenerateContentConfig(max_output_tokens=5),
                )
                _gemini_client = ("new_sdk", client)
            else:
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel(model_name=_active_model)
                model.generate_content("Reply with the single word: OK")
                _gemini_client = ("old_sdk", model)

            logger.info(f"Backend: GEMINI  (model={_active_model})")
            _BACKEND = "gemini"
            return "gemini"

        except Exception as e:
            logger.warning(
                f"Gemini unavailable ({e}). "
                f"Falling back to Mistral-7B + Helsinki-NLP."
            )
    else:
        logger.warning(
            "GEMINI_API_KEY not set. "
            "Falling back to Mistral-7B + Helsinki-NLP."
        )

    # Fall back to Mistral
    _init_mistral()
    _BACKEND = "mistral"
    return "mistral"


def _get_backend() -> str:
    """Return cached backend, initialising on first call."""
    global _BACKEND
    if _BACKEND is None:
        _init_backend()
    return _BACKEND

# Gemini helpers

_GEMINI_SYSTEM = (
    "You are a legal assistant helping rural Indian villagers understand "
    "their legal rights and government schemes. Be helpful, accurate, and "
    "use simple language. Use ONLY the context provided from official legal "
    "documents to answer. If the context does not contain the answer, say: "
    "'I don't have information about this in the documents. Please consult a "
    "local legal aid centre.' Do NOT make up laws, scheme amounts, or "
    "eligibility criteria. Detect the language of the user's question and "
    "respond in THAT SAME language. If the question is in Telugu, respond "
    "entirely in Telugu. Use simple language a rural villager can understand."
)


def _call_gemini(prompt: str, system: str = _GEMINI_SYSTEM,
                 max_tokens: int = GEMINI_MAX_TOKENS) -> str:
    """Call Gemini with retry on 429 rate-limit errors."""
    sdk_type, client = _gemini_client

    for attempt in range(4):
        try:
            if sdk_type == "new_sdk":
                from google.genai import types as _types
                response = client.models.generate_content(
                    model=_active_model,
                    contents=prompt,
                    config=_types.GenerateContentConfig(
                        system_instruction=system,
                        max_output_tokens=max_tokens,
                        temperature=GEMINI_TEMPERATURE,
                    ),
                )
                return response.text.strip()
            else:
                response = client.generate_content(prompt)
                return response.text.strip()

        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                wait = [0, 15, 30, 60][attempt]
                logger.warning(
                    f"Gemini rate-limited (attempt {attempt+1}/4). "
                    f"Waiting {wait}s ..."
                )
                if wait:
                    time.sleep(wait)
            else:
                raise

    raise RuntimeError("Gemini failed after 4 attempts.")


def _gemini_translate_to_english(text: str) -> str:
    """Translate Telugu → English using Gemini. Skip if already English."""
    telugu_chars = sum(1 for c in text if "\u0C00" <= c <= "\u0C7F")
    if len(text) == 0 or (telugu_chars / len(text)) < 0.10:
        return text   # already mostly English

    prompt = (
        f"Translate the following Telugu text to English. "
        f"Preserve legal terminology accurately. "
        f"Output ONLY the English translation, nothing else.\n\n"
        f"Telugu: {text}\n\nEnglish:"
    )
    try:
        return _call_gemini(
            prompt, max_tokens=512,
            system="You are an expert Telugu-English legal translator. "
                   "Output only the translation."
        )
    except Exception as e:
        logger.warning(f"Gemini translation failed: {e}. Using original text.")
        return text


def _gemini_translate_to_telugu(text: str) -> str:
    """Translate English → Telugu using Gemini."""
    prompt = (
        f"Translate the following English legal answer to Telugu. "
        f"Output ONLY the Telugu translation:\n\n{text}"
    )
    try:
        return _call_gemini(
            prompt, max_tokens=800,
            system="You are an expert translator. Output only the Telugu translation."
        )
    except Exception as e:
        logger.warning(f"Gemini En→Te translation failed: {e}.")
        return text

# Helsinki-NLP helpers (used only when backend == "mistral")
def _init_helsinki():
    """Load Helsinki-NLP translation pipelines (lazy, cached)."""
    global _te_to_en_pipe, _en_to_te_pipe

    if _te_to_en_pipe is not None:
        return  # already loaded

    from transformers import pipeline as hf_pipeline

    logger.info(f"Loading Helsinki-NLP  Te→En  ({TE_TO_EN_MODEL}) ...")
    _te_to_en_pipe = hf_pipeline(
        "translation",
        model=TE_TO_EN_MODEL,
        device=0 if TRANSLATION_DEVICE == "cuda" else -1,
        max_length=TRANSLATION_MAX_LEN,
    )
    logger.info("Helsinki Te→En loaded ✓")

    logger.info(f"Loading Helsinki-NLP  En→Te  ({EN_TO_TE_MODEL}) ...")
    _en_to_te_pipe = hf_pipeline(
        "translation",
        model=EN_TO_TE_MODEL,
        device=0 if TRANSLATION_DEVICE == "cuda" else -1,
        max_length=TRANSLATION_MAX_LEN,
    )
    logger.info("Helsinki En→Te loaded ✓")


def _helsinki_translate_to_english(text: str) -> str:
    """Translate Telugu → English using Helsinki-NLP."""
    _init_helsinki()
    telugu_chars = sum(1 for c in text if "\u0C00" <= c <= "\u0C7F")
    if len(text) == 0 or (telugu_chars / len(text)) < 0.10:
        return text   # already mostly English
    try:
        result = _te_to_en_pipe(text, max_length=TRANSLATION_MAX_LEN)
        return result[0]["translation_text"].strip()
    except Exception as e:
        logger.warning(f"Helsinki Te→En failed: {e}. Using original.")
        return text


def _helsinki_translate_to_telugu(text: str) -> str:
    """Translate English → Telugu using Helsinki-NLP."""
    _init_helsinki()
    try:
        result = _en_to_te_pipe(text, max_length=TRANSLATION_MAX_LEN)
        return result[0]["translation_text"].strip()
    except Exception as e:
        logger.warning(f"Helsinki En→Te failed: {e}. Returning English.")
        return text

# Mistral (LlamaCpp) helper (used only when backend == "mistral")

def _init_mistral():
    """Load Mistral-7B via LlamaCpp (lazy, cached)."""
    global _mistral_llm

    if _mistral_llm is not None:
        return

    from langchain_community.llms import LlamaCpp

    model_path = Path(LLM_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Mistral GGUF model not found at: {model_path}\n"
            "Download with:\n"
            "  huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF "
            "mistral-7b-instruct-v0.2.Q4_K_M.gguf "
            "--local-dir . --local-dir-use-symlinks False\n"
            "Then set LLM_MODEL_PATH in .env"
        )

    logger.info(
        f"Loading Mistral-7B  (gpu_layers={LLM_N_GPU_LAYERS}, "
        f"ctx={LLM_CONTEXT_LEN}) ..."
    )
    _mistral_llm = LlamaCpp(
        model_path=str(model_path),
        n_gpu_layers=LLM_N_GPU_LAYERS,
        n_ctx=LLM_CONTEXT_LEN,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        verbose=False,
        seed=42,
    )
    logger.info("Mistral-7B loaded ✓")
    logger.info("Backend: MISTRAL + HELSINKI-NLP")


def _call_mistral(prompt: str) -> str:
    """Invoke Mistral via LlamaCpp."""
    if _mistral_llm is None:
        _init_mistral()
    result = _mistral_llm.invoke(prompt)
    return result.strip() if isinstance(result, str) else str(result).strip()

# Shared: embeddings, reranker, vector store

def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"Embeddings loaded ✓  ({EMBEDDING_MODEL})")
    return _embeddings


def _get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker
    if not ENABLE_RERANKER:
        return None
    try:
        from flashrank import Ranker
        _reranker = Ranker(
            model_name=RERANKER_MODEL,
            cache_dir="/tmp/flashrank_cache",
        )
        logger.info(f"Reranker loaded ✓  ({RERANKER_MODEL})")
        return _reranker
    except ImportError:
        logger.warning("flashrank not installed — reranking disabled.")
        return None
    except Exception as e:
        logger.warning(f"Reranker load failed: {e} — skipping.")
        return None


def _rerank_docs(query: str, docs: list, top_k: int) -> list:
    """Run FlashRank cross-encoder reranking. Returns top_k docs."""
    ranker = _get_reranker()
    if ranker is None or len(docs) <= top_k:
        return docs[:top_k]
    try:
        from flashrank import RerankRequest
        passages = [
            {"id": i, "text": doc.page_content}
            for i, doc in enumerate(docs)
        ]
        reranked = ranker.rerank(RerankRequest(query=query, passages=passages))
        return [docs[r["id"]] for r in reranked[:top_k]]
    except Exception as e:
        logger.warning(f"Reranking failed: {e}. Using original order.")
        return docs[:top_k]


def _get_vectordb():
    global _vectordb
    if _vectordb is not None:
        return _vectordb
    embeddings = _get_embeddings()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _vectordb = Chroma(
        client=client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
    )
    count = _vectordb._collection.count()
    if count == 0:
        logger.warning("ChromaDB is EMPTY. Run: python scripts/index_documents.py")
    else:
        logger.info(f"ChromaDB loaded ✓  ({count} chunks in '{CHROMA_COLLECTION}')")
    return _vectordb


def _make_splitter() -> RecursiveCharacterTextSplitter:
    """Legal-aware splitter with Telugu danda support."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\nSection", "\n\nArticle", "\n\nRule ", "\n\nClause",
            "\n\n", "\u0964\n", "\u0964", ".\n", ".", " ", "",
        ],
        keep_separator=True,
    )

# Web search helper (shared by both backends)
def _run_web_search(query: str) -> str:
    try:
        from duckduckgo_search import DDGS
        search_q = f"{query} India government scheme law site:.gov.in OR site:.nic.in"
        with DDGS() as ddgs:
            results = list(ddgs.text(search_q, max_results=WEB_SEARCH_MAX_RESULTS,
                                     region="in-en"))
        if not results:
            return ""
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[{i}] {r.get('title','')}\n{r.get('body','')[:300]}\n"
                f"Source: {r.get('href','')}"
            )
        logger.info(f"Web search returned {len(results)} result(s)")
        return "\n\n".join(parts)
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return ""
# Public interface

def get_qa_chain():
    """
    Startup probe — returns a truthy object if the pipeline is ready.
    Triggers backend detection on first call.
    """
    backend = _get_backend()
    if backend == "gemini":
        return _gemini_client          # truthy if Gemini is loaded
    else:
        try:
            _init_mistral()
            return _mistral_llm        # truthy if Mistral is loaded
        except Exception as e:
            logger.error(f"Mistral init failed: {e}")
            return None


def query_legal_bot(
    user_query: str,
) -> Tuple[str, str, List[str], bool, dict]:
    """
    Main query entry point.

    Returns:
        answer_primary  — answer in user's language (Telugu or English)
        answer_en       — English version of the answer (for UI toggle)
        sources         — list of source document filenames
        used_web        — whether the web search agent fired
        agents_used     — dict with "calculator" and "web_search" details
    """
    agents_used = {"calculator": None, "web_search": None}
    backend     = _get_backend()

    # ── Step 1: Calculator agent (runs regardless of backend) 
    calc_result = None
    if ENABLE_CALCULATOR:
        calc_result = run_calculator(user_query)
        if calc_result:
            agents_used["calculator"] = calc_result
            logger.info(f"[CalculatorAgent] {calc_result}")

    # ── Step 2: Translate query to English for retrieval
    if backend == "gemini":
        english_query = _gemini_translate_to_english(user_query)
    else:
        english_query = _helsinki_translate_to_english(user_query)

    # ── Step 3: ChromaDB retrieval + reranking 
    sources = []
    context = ""
    try:
        db = _get_vectordb()
        if db._collection.count() > 0:
            retriever = db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k":           RETRIEVAL_FETCH_K,
                    "fetch_k":     RETRIEVAL_FETCH_K * 2,
                    "lambda_mult": 0.7,
                },
            )
            docs    = retriever.invoke(english_query)
            sources = list({doc.metadata.get("source", "unknown") for doc in docs})
            docs    = _rerank_docs(english_query, docs, top_k=RETRIEVAL_K)
            context = "\n\n---\n\n".join(
                f"[Source: {doc.metadata.get('source','unknown')}]\n{doc.page_content}"
                for doc in docs
            )
            logger.info(f"Retrieved {len(docs)} chunks. Sources: {sources}")
        else:
            context = "No documents available in the knowledge base."
    except Exception as e:
        logger.error(f"ChromaDB retrieval failed: {e}")
        context = "Knowledge base temporarily unavailable."

    # ── Step 4: Build prompt 
    calc_section = f"\n\nCalculation result:\n{calc_result}\n" if calc_result else ""

    prompt = (
        f"Context from legal documents:\n\n{context}"
        f"{calc_section}"
        f"\n\n---\n\n"
        f"Question: {user_query}\n\n"
        f"Answer:"
    )

    # ── Step 5: Generate answer 
    try:
        if backend == "gemini":
            answer = _call_gemini(prompt)
        else:
            # Mistral needs the question in English for best quality
            mistral_prompt = (
                f"{LEGAL_SYSTEM_PROMPT.format(context=context, question=english_query)}"
            )
            answer = _call_mistral(mistral_prompt)

        logger.info(f"Answer generated ({backend}): {len(answer)} chars")

    except Exception as e:
        logger.error(f"Answer generation failed ({backend}): {e}")
        fallback_msg = (
            "నేను ఇప్పుడు సమాధానం ఇవ్వలేకపోతున్నాను. దయచేసి తర్వాత ప్రయత్నించండి.\n"
            "(I cannot answer right now. Please try again later.)"
        )
        return fallback_msg, "I cannot answer right now. Please try again.", sources, False, agents_used

    # ── Step 6: Web search agent (post-generation, both backends) ─────────────
    used_web = False
    if should_web_search(user_query, answer, enabled=ENABLE_WEB_SEARCH):
        logger.info("[WebSearchAgent] Triggered")
        web_results = _run_web_search(english_query)
        if web_results:
            agents_used["web_search"] = True
            web_prompt = (
                f"Recent web search results:\n\n{web_results}\n\n"
                f"---\n\n"
                f"Original answer:\n{answer}\n\n"
                f"---\n\n"
                f"Question: {user_query}\n\n"
                f"Provide an updated answer combining both sources. "
                f"Respond in the same language as the question."
            )
            try:
                if backend == "gemini":
                    enriched = _call_gemini(web_prompt, max_tokens=800)
                else:
                    enriched = _call_mistral(web_prompt)

                if enriched:
                    answer   = enriched
                    used_web = True
                    sources.append("web_search")
            except Exception as e:
                logger.warning(f"[WebSearchAgent] Enrichment failed: {e}")

    # ── Step 7: Produce English version for UI toggle 
    telugu_chars   = sum(1 for c in user_query if "\u0C00" <= c <= "\u0C7F")
    is_telugu      = len(user_query) > 0 and (telugu_chars / len(user_query)) > 0.20

    if is_telugu:
        # answer is already in Telugu — translate it back to English for UI
        try:
            if backend == "gemini":
                answer_en = _gemini_translate_to_english(answer)
            else:
                answer_en = _helsinki_translate_to_english(answer)
        except Exception:
            answer_en = "(English translation unavailable)"
    else:
        # Query was English — answer is in English
        # For Mistral backend: also produce a Telugu version as answer_primary
        if backend == "mistral":
            try:
                answer_primary = _helsinki_translate_to_telugu(answer)
            except Exception:
                answer_primary = answer
        else:
            answer_primary = answer
        answer_en = answer

    # For Telugu queries, answer is already in Telugu
    answer_primary = answer if is_telugu else (
        answer_primary if backend == "mistral" else answer
    )

    return answer_primary, answer_en, sources, used_web, agents_used

# Document indexing (shared, backend-independent)

def index_documents(docs_dir: Optional[str] = None) -> int:
    """
    Batch-index all documents in docs_dir into ChromaDB.
    Deletes and rebuilds the collection for a clean index.
    """
    from modules.doc_extraction import process_uploaded_file, get_supported_extensions

    source_dir    = Path(docs_dir) if docs_dir else (CHROMA_DIR.parent / "legal_docs")
    supported_ext = set(get_supported_extensions())
    all_files     = [f for f in source_dir.rglob("*") if f.suffix.lower() in supported_ext]

    if not all_files:
        logger.warning(f"No supported documents found in {source_dir}.")
        return 0

    raw_docs: List[Document] = []
    for file_path in all_files:
        try:
            text = process_uploaded_file(str(file_path))
            if text.strip():
                raw_docs.append(Document(
                    page_content=text,
                    metadata={"source": file_path.name, "path": str(file_path)},
                ))
                logger.info(f"  Loaded: {file_path.name} ({len(text):,} chars)")
            else:
                logger.warning(f"  Empty — skipped: {file_path.name}")
        except Exception as e:
            logger.error(f"  Failed: {file_path.name} — {e}")

    if not raw_docs:
        logger.error("No content extracted. Aborting.")
        return 0

    splitter = _make_splitter()
    chunks   = splitter.split_documents(raw_docs)
    logger.info(f"Chunked: {len(raw_docs)} docs → {len(chunks)} chunks")

    embeddings = _get_embeddings()
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(CHROMA_COLLECTION)
        logger.info(f"Old collection '{CHROMA_COLLECTION}' deleted for fresh index")
    except Exception:
        pass

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=CHROMA_COLLECTION,
    )

    count = vectordb._collection.count()
    logger.info(f"Indexed {count} chunks into ChromaDB at '{CHROMA_DIR}'")

    # Reset cached vectordb so next query picks up the new index
    global _vectordb
    _vectordb = None

    return count


def add_document_to_index(text: str, source_name: str) -> int:
    """Dynamically add a single document to the existing index (no rebuild)."""
    if not text.strip():
        logger.warning(f"Empty text for '{source_name}' — skipped.")
        return 0

    splitter = _make_splitter()
    doc      = Document(
        page_content=text,
        metadata={"source": source_name, "dynamic": True},
    )
    chunks = splitter.split_documents([doc])

    embeddings = _get_embeddings()
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    vectordb   = Chroma(
        client=client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
    )
    vectordb.add_documents(chunks)

    # Reset cache so next query sees the new chunks
    global _vectordb
    _vectordb = None

    logger.info(f"Indexed {len(chunks)} chunks from '{source_name}'")
    return len(chunks)