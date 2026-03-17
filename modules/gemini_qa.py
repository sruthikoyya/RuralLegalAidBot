
import logging
import os
import time
from pathlib import Path
from typing import Optional, List, Tuple

import chromadb
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

try:
    from google import genai
    from google.genai import types
    _NEW_SDK = True
except ImportError:
    import google.generativeai as genai
    _NEW_SDK = False

from modules.config import (
    CHROMA_DIR, CHROMA_COLLECTION,
    EMBEDDING_MODEL, EMBEDDING_DEVICE,
    RETRIEVAL_K, RETRIEVAL_FETCH_K, CHUNK_SIZE, CHUNK_OVERLAP,
    ENABLE_WEB_SEARCH, WEB_SEARCH_MAX_RESULTS,
    ENABLE_RERANKER, RERANKER_MODEL,
    ENABLE_CALCULATOR,
    GEMINI_API_KEY, GEMINI_MODEL, GEMINI_MAX_TOKENS, GEMINI_TEMPERATURE,
)
from modules.agents import (
    run_calculator, run_web_search,
    should_web_search, route_agents,
)

logger = logging.getLogger(__name__)

_vectordb     = None
_gemini_client = None
_reranker     = None
_embeddings   = None
_active_model = GEMINI_MODEL   

_MODEL_CANDIDATES = [
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.0-flash",
    "models/gemini-2.5-flash",
    "models/gemini-1.5-flash",
    "models/gemini-flash-lite-latest",
]

SYSTEM_INSTRUCTION = (
    "You are a legal assistant helping rural Indian villagers understand "
    "their legal rights and government schemes. Be helpful, accurate, and "
    "use simple language. "
    "Use ONLY the context provided from official legal documents to answer. "
    "If the context does not contain the answer, say: "
    "'I don't have information about this in the documents. "
    "Please consult a local legal aid centre.' "
    "Do NOT make up laws, scheme amounts, or eligibility criteria. "
    "Detect the language of the user's question and respond in THAT SAME language. "
    "If the question is in Telugu (తెలుగు), respond entirely in Telugu. "
    "If the question is in English, respond in Telugu only. "
    "Use simple language a rural villager can understand."
)

def _find_working_model(client) -> str:
    """Auto-discover the best available Gemini model for this API key / region."""
    try:
        available = {m.name for m in client.models.list()}
        for candidate in _MODEL_CANDIDATES:
            if candidate in available:
                logger.info(f"Auto-selected Gemini model: {candidate}")
                return candidate
        for m in sorted(available):
            if "gemini" in m and "flash" in m:
                logger.info(f"Fallback Gemini model: {m}")
                return m
    except Exception as e:
        logger.warning(f"Could not list Gemini models: {e}")
    return "models/gemini-2.0-flash-lite"


def _get_gemini():
    """Load and cache the Gemini client (supports both old and new google-genai SDK)."""
    global _gemini_client, _active_model
    if _gemini_client is not None:
        return _gemini_client

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not set.\n"
            "  1. Get free key: https://aistudio.google.com/app/apikey\n"
            "  2. Add to .env:  GEMINI_API_KEY=your_key_here\n"
            "  3. Restart server."
        )

    if _NEW_SDK:
        client = genai.Client(api_key=GEMINI_API_KEY)
        if not os.getenv("GEMINI_MODEL"):
            _active_model = _find_working_model(client)
        _gemini_client = client
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_client = genai.GenerativeModel(
            model_name=_active_model,
            generation_config=genai.GenerationConfig(
                max_output_tokens=GEMINI_MAX_TOKENS,
                temperature=GEMINI_TEMPERATURE,
            ),
            system_instruction=SYSTEM_INSTRUCTION,
        )
    logger.info(f"Gemini loaded ✓  (model={_active_model})")
    return _gemini_client


def _call_gemini(prompt: str, max_tokens: int = GEMINI_MAX_TOKENS, system: str = SYSTEM_INSTRUCTION) -> str:
    """
    Calls Gemini with retry logic for rate limits (429).
    Returns the response text.
    """
    client = _get_gemini()
    last_error = None

    for attempt in range(4):
        try:
            if _NEW_SDK:
                response = client.models.generate_content(
                    model=_active_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
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
            last_error = e
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                wait = [0, 15, 30, 60][attempt]
                if wait > 0:
                    logger.warning(f"Gemini rate limited (attempt {attempt+1}/4). Waiting {wait}s ...")
                    time.sleep(wait)
                else:
                    logger.warning(f"Gemini rate limited (attempt {attempt+1}/4). Retrying ...")
            else:
                logger.error(f"Gemini API error: {e}")
                raise

    raise RuntimeError(f"Gemini failed after 4 attempts. Last error: {last_error}")

def translate_to_english(text: str) -> str:
    
    if not text.strip():
        return text

    # Quick check: if already mostly ASCII/English, skip translation
    telugu_chars = sum(1 for c in text if "\u0C00" <= c <= "\u0C7F")
    if len(text) > 0 and (telugu_chars / len(text)) < 0.10:
        logger.debug("Input appears English — skipping translation")
        return text

    prompt = (
        f"Translate the following Telugu text to English. "
        f"Preserve legal terminology accurately. "
        f"Output ONLY the English translation, nothing else.\n\n"
        f"Telugu: {text}\n\nEnglish:"
    )
    try:
        translation = _call_gemini(
            prompt,
            max_tokens=512,
            system="You are an expert Telugu-English legal translator. Output only the translation."
        )
        logger.info(f"Translated to English: '{translation[:80]}...'")
        return translation
    except Exception as e:
        logger.warning(f"Gemini translation failed: {e}. Using original text.")
        return text  # graceful fallback

def _get_embeddings() -> HuggingFaceEmbeddings:
    """Load and cache multilingual MiniLM embedding model."""
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
    """
    Load FlashRank cross-encoder reranker (lazy, cached).
    Falls back gracefully if flashrank is not installed.
    """
    global _reranker
    if _reranker is not None:
        return _reranker

    if not ENABLE_RERANKER:
        return None

    try:
        from flashrank import Ranker
        _reranker = Ranker(model_name=RERANKER_MODEL, cache_dir="/tmp/flashrank_cache")
        logger.info(f"Reranker loaded ✓  ({RERANKER_MODEL})")
        return _reranker
    except ImportError:
        logger.warning(
            "flashrank not installed — reranking disabled.\n"
            "  Install with: pip install flashrank"
        )
        return None
    except Exception as e:
        logger.warning(f"Reranker load failed: {e}. Continuing without reranking.")
        return None


def _rerank_docs(query: str, docs: list, top_k: int = RETRIEVAL_K) -> list:
    
    if not docs:
        return docs

    reranker = _get_reranker()
    if reranker is None:
        return docs[:top_k]

    try:
        from flashrank import RerankRequest
        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(docs)
        ]
        request  = RerankRequest(query=query, passages=passages)
        reranked = reranker.rerank(request)

        # Map back to original documents
        top_docs = [docs[r["id"]] for r in reranked[:top_k]]
        logger.info(f"Reranked {len(docs)} → top {len(top_docs)} docs")
        return top_docs

    except Exception as e:
        logger.warning(f"Reranking failed: {e}. Using original order.")
        return docs[:top_k]

def _get_vectordb():
    """Load and cache ChromaDB vector store."""
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


def get_qa_chain():
    """
    Compatibility stub for main.py startup check.
    Returns the Gemini client if API key is set, else None.
    """
    try:
        return _get_gemini()
    except Exception as e:
        logger.warning(f"Gemini not ready: {e}")
        return None

def query_legal_bot(
    user_query: str,
    already_english: bool = False,
) -> Tuple[str, str, List[str], bool, dict]:

    agents_used = {"calculator": None, "web_search": None}

        
    calc_result = None
    if ENABLE_CALCULATOR:
        calc_result = run_calculator(user_query)
        if calc_result:
            agents_used["calculator"] = calc_result
            logger.info(f"[CalculatorAgent] Result: {calc_result}")

        
    if already_english:
        english_query = user_query
    else:
        english_query = translate_to_english(user_query)
        if english_query == user_query and not already_english:
            logger.debug("Translation was a no-op (input likely already English)")
    sources = []
    docs    = []
    context = ""

    try:
        db = _get_vectordb()
        if db._collection.count() > 0:
            retriever = db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k":       RETRIEVAL_FETCH_K,   # fetch 16, rerank to 4
                    "fetch_k": RETRIEVAL_FETCH_K * 2,
                    "lambda_mult": 0.7,             # diversity vs relevance trade-off
                },
            )
            docs    = retriever.invoke(english_query)
            sources = list({doc.metadata.get("source", "unknown") for doc in docs})
            logger.info(f"Retrieved {len(docs)} chunks from ChromaDB. Sources: {sources}")

            
            docs = _rerank_docs(english_query, docs, top_k=RETRIEVAL_K)

            context = "\n\n---\n\n".join(
                f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                for doc in docs
            )
        else:
            logger.warning("ChromaDB is empty — answering without document context")
            context = "No documents available in the knowledge base."

    except Exception as e:
        logger.error(f"ChromaDB retrieval failed: {e}")
        context = "Knowledge base temporarily unavailable."
    calc_section = ""
    if calc_result:
        calc_section = f"\n\nCalculation result for this query:\n{calc_result}\n"

    prompt = (
        f"Context from legal documents:\n\n{context}"
        f"{calc_section}"
        f"\n\n---\n\n"
        f"Question: {user_query}\n\n"
        f"Answer:"
    )
     
    try:
        answer = _call_gemini(prompt)
        logger.info(f"Gemini answer: {len(answer)} chars")
    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        return (
            "నేను ఇప్పుడు సమాధానం ఇవ్వలేకపోతున్నాను. దయచేసి తర్వాత మళ్ళీ ప్రయత్నించండి.\n"
            "(I cannot answer right now. Please try again later.)",
            "I encountered an error. Please try again.",
            sources, False, agents_used,
        )

  
    used_web = False
    if should_web_search(user_query, answer, enabled=ENABLE_WEB_SEARCH):
        logger.info("[WebSearchAgent] Triggered")
        web_results = run_web_search(english_query, max_results=WEB_SEARCH_MAX_RESULTS)
        if web_results:
            agents_used["web_search"] = True
            web_prompt = (
                f"Recent web search results about this legal topic:\n\n{web_results}\n\n"
                f"---\n\n"
                f"Original RAG answer:\n{answer}\n\n"
                f"---\n\n"
                f"Question: {user_query}\n\n"
                f"Based on both the document context AND the web results, "
                f"provide an updated, comprehensive answer. "
                f"If web results add new info, include it and note it's from recent web sources. "
                f"Respond in the same language as the question."
            )
            try:
                web_answer = _call_gemini(web_prompt, max_tokens=800)
                if web_answer:
                    answer = web_answer
                    sources.append("web_search")
                    used_web = True
                    logger.info(f"[WebSearchAgent] Answer enriched ({len(web_answer)} chars)")
            except Exception as e:
                logger.warning(f"[WebSearchAgent] Gemini enrichment failed: {e}")


    telugu_chars = sum(1 for c in user_query if "\u0C00" <= c <= "\u0C7F")
    is_telugu_query = len(user_query) > 0 and (telugu_chars / len(user_query)) > 0.20

    if is_telugu_query:
        # Get English version for the "Show English" toggle in UI
        try:
            en_prompt = (
                f"Translate this Telugu legal answer to English. "
                f"Output ONLY the English translation:\n\n{answer}"
            )
            answer_en = _call_gemini(
                en_prompt, max_tokens=800,
                system="You are an expert translator. Output only the English translation."
            )
        except Exception:
            answer_en = "(English translation unavailable)"
    else:
        answer_en = answer  # already English

    return answer, answer_en, sources, used_web, agents_used

def _make_splitter() -> RecursiveCharacterTextSplitter:
    """
    Legal-aware text splitter.
    Splits on section headers, paragraphs, Telugu punctuation before characters.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n## ",       # Markdown headers (legal section titles)
            "\n\nSection",   # Section headers
            "\n\nArticle",   # Article headers
            "\n\nRule ",     # Rule headers
            "\n\nClause",    # Clause headers
            "\n\n",          # Paragraphs
            "\u0964\n",      # Telugu danda + newline
            "\u0964",        # Telugu danda (sentence boundary)
            ".\n",
            ".",
            " ",
            "",
        ],
        keep_separator=True,
    )
def add_document_to_index(text: str, source_name: str) -> int:

    if not text.strip():
        logger.warning(f"Empty text for '{source_name}' — skipping")
        return 0

    splitter = _make_splitter()
    doc      = Document(page_content=text, metadata={"source": source_name, "dynamic": True})
    chunks   = splitter.split_documents([doc])

    embeddings = _get_embeddings()
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    vectordb   = Chroma(
        client=client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
    )
    vectordb.add_documents(chunks)

    global _vectordb
    _vectordb = None

    logger.info(f"Indexed {len(chunks)} chunks from '{source_name}'")
    return len(chunks)


def index_documents(docs_dir: Optional[str] = None) -> int:
    """Delegates to qa_system.index_documents() — embeddings & chunking unchanged."""
    from modules.qa_system import index_documents as _index
    return _index(docs_dir)
