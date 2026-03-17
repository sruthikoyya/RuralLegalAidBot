import logging

from pathlib import Path
from typing import Optional, List, Tuple

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from modules.config import (
    CHROMA_DIR, CHROMA_COLLECTION,
    EMBEDDING_MODEL, EMBEDDING_DEVICE,
    LLM_MODEL_PATH, LLM_N_GPU_LAYERS, LLM_CONTEXT_LEN,
    LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_TOP_P,
    RETRIEVAL_K, CHUNK_SIZE, CHUNK_OVERLAP,
    LEGAL_SYSTEM_PROMPT,
    ENABLE_WEB_SEARCH, WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_TRIGGERS,
)

logger = logging.getLogger(__name__)

# Module-level cache — chain is built once then reused for all queries
_qa_chain: Optional[RetrievalQA] = None

def _get_embeddings() -> HuggingFaceEmbeddings:

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

def _needs_web_search(question: str, rag_answer: str) -> bool:
    
    if not ENABLE_WEB_SEARCH:
        return False

    uncertainty_phrases = [
        "i don't have information",
        "not in the documents",
        "i cannot find",
        "not available in",
        "no information",
        "cannot answer",
        "please consult",
    ]
    answer_lower   = rag_answer.lower()
    question_lower = question.lower()

    rag_uncertain = any(phrase in answer_lower for phrase in uncertainty_phrases)
    time_sensitive = any(kw in question_lower for kw in WEB_SEARCH_TRIGGERS)

    return rag_uncertain or time_sensitive


def _web_search(query: str) -> str:
    
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning(
            "duckduckgo-search not installed. "
            "Run: pip install duckduckgo-search"
        )
        return ""

    search_query = f"{query} India government scheme law site:.gov.in OR site:.nic.in"
    logger.info(f"Web search: '{search_query[:80]}'")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=WEB_SEARCH_MAX_RESULTS))

        if not results:
            return ""

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body  = r.get("body", "")
            href  = r.get("href", "")
            formatted.append(f"[{i}] {title}\n{body}\nSource: {href}")

        result_text = "\n\n".join(formatted)
        logger.info(f"Web search returned {len(results)} result(s)")
        return result_text

    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return ""


def _answer_with_web(question: str, web_results: str) -> str:

    chain = get_qa_chain()
    if chain is None or not web_results:
        return ""

    # Re-use the same LLM but with web results as context
    llm = chain.combine_documents_chain.llm_chain.llm

    web_prompt = (
        f"Based on the following recent web search results, answer this question "
        f"about Indian law or government schemes.\n\n"
        f"Web Search Results:\n{web_results}\n\n"
        f"Question: {question}\n\n"
        f"Answer (be clear and accurate, mention this is from recent web sources):"
    )

    try:
        answer = llm.invoke(web_prompt)
        if isinstance(answer, str):
            return answer.strip()
        return str(answer).strip()
    except Exception as e:
        logger.warning(f"Web-based answering failed: {e}")
        return ""

def index_documents(docs_dir: Optional[str] = None) -> int:

    from modules.doc_extraction import process_uploaded_file, get_supported_extensions

    source_dir = Path(docs_dir) if docs_dir else (CHROMA_DIR.parent / "legal_docs")
    logger.info(f"Indexing documents from: {source_dir}")

    supported_ext = set(get_supported_extensions())
    all_files = [f for f in source_dir.rglob("*") if f.suffix.lower() in supported_ext]

    if not all_files:
        logger.warning(
            f"No supported documents found in {source_dir}.\n"
            f"Supported types: {sorted(supported_ext)}\n"
            f"Add PDF/DOCX/TXT files and re-run."
        )
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
                logger.info(f"  ✓ Loaded: {file_path.name} ({len(text):,} chars)")
            else:
                logger.warning(f"  ⚠ Empty content skipped: {file_path.name}")
        except Exception as e:
            logger.error(f"  ✗ Failed to load {file_path.name}: {e}")

    if not raw_docs:
        logger.error("No content extracted. Aborting indexing.")
        return 0

    # Chunk with Telugu-aware separators
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "\u0964", ".", " ", ""],
        #           ^^^ \u0964 = Telugu/Hindi danda (।)
    )
    chunks = splitter.split_documents(raw_docs)
    logger.info(f"Chunked: {len(raw_docs)} docs → {len(chunks)} chunks")

    # Embed and store
    embeddings = _get_embeddings()
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete old collection to allow clean re-indexing
    try:
        client.delete_collection(CHROMA_COLLECTION)
        logger.info(f"Deleted old '{CHROMA_COLLECTION}' collection for fresh index")
    except Exception:
        pass

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=CHROMA_COLLECTION,
    )

    count = vectordb._collection.count()
    logger.info(f" Indexed {count} chunks into ChromaDB at '{CHROMA_DIR}'")
    return count

def get_qa_chain() -> Optional[RetrievalQA]:

    global _qa_chain
    if _qa_chain is not None:
        return _qa_chain

    # ── Load ChromaDB 
    embeddings = _get_embeddings()
    try:
        client   = chromadb.PersistentClient(path=str(CHROMA_DIR))
        vectordb = Chroma(
            client=client,
            collection_name=CHROMA_COLLECTION,
            embedding_function=embeddings,
        )
        count = vectordb._collection.count()
        if count == 0:
            logger.warning(
                "ChromaDB collection is EMPTY. "
                "Run:  python scripts/index_documents.py"
            )
        else:
            logger.info(f"ChromaDB loaded ✓  ({count} chunks in '{CHROMA_COLLECTION}')")
    except Exception as e:
        logger.error(f"ChromaDB load failed: {e}")
        return None

    # ── Load Mistral-7B via LlamaCpp 
    model_path = Path(LLM_MODEL_PATH)
    if not model_path.exists():
        logger.error(
            f"LLM model NOT FOUND at: {model_path}\n"
            "\n"
            "Download it with:\n"
            "  huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \\\n"
            "      mistral-7b-instruct-v0.2.Q4_K_M.gguf \\\n"
            "      --local-dir . --local-dir-use-symlinks False\n"
            "\n"
            "Then set LLM_MODEL_PATH in your .env to the absolute path."
        )
        return None

    try:
        llm = LlamaCpp(
            model_path=str(model_path),
            n_gpu_layers=LLM_N_GPU_LAYERS,
            n_ctx=LLM_CONTEXT_LEN,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            verbose=False,
            seed=42,            # reproducibility
        )
        logger.info(
            f"Mistral-7B loaded ✓  "
            f"(gpu_layers={LLM_N_GPU_LAYERS}, ctx={LLM_CONTEXT_LEN})"
        )
    except Exception as e:
        logger.error(f"LlamaCpp load failed: {e}")
        return None

    # ── Build Prompt + Chain 
    prompt = PromptTemplate(
        template=LEGAL_SYSTEM_PROMPT,
        input_variables=["context", "question"],
    )

    _qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",               # concatenate all chunks into one prompt
        retriever=vectordb.as_retriever(
            search_type="mmr",            # Maximal Marginal Relevance = diverse chunks
            search_kwargs={
                "k": RETRIEVAL_K,
                "fetch_k": RETRIEVAL_K * 3,
            },
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,     # include source file names in response
    )

    web_status = "ENABLED" if ENABLE_WEB_SEARCH else "disabled"
    logger.info(f"RAG chain ready ✓  (web search fallback: {web_status})")
    return _qa_chain

def query_legal_bot(english_question: str) -> Tuple[str, List[str], bool]:

    chain = get_qa_chain()
    if chain is None:
        return (
            "I'm sorry — my knowledge base is currently offline. "
            "Please contact your local legal aid centre.",
            [],
            False,
        )

    logger.info(f"RAG query: '{english_question[:80]}'")

    # ── Step 1: RAG answer ────────────────────────────────────────────────────
    try:
        result      = chain.invoke({"query": english_question})
        rag_answer  = result.get("result", "").strip()
        source_docs = result.get("source_documents", [])
        sources     = list({
            doc.metadata.get("source", "unknown") for doc in source_docs
        })
        logger.info(f"RAG answer: {len(rag_answer)} chars. Sources: {sources}")
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        rag_answer = (
            "I encountered an error processing your question. Please try again."
        )
        sources = []

    # ── Step 2: Web search fallback ───────────────────────────────────────────
    used_web = False
    if _needs_web_search(english_question, rag_answer):
        logger.info("Triggering web search fallback ...")
        web_results = _web_search(english_question)
        if web_results:
            web_answer = _answer_with_web(english_question, web_results)
            if web_answer:
                # Combine: RAG context + web context
                if "i don't have information" in rag_answer.lower():
                    final_answer = (
                        f"{web_answer}\n\n"
                        f"(This information is from recent web sources — "
                        f"please verify with official government portals.)"
                    )
                else:
                    final_answer = (
                        f"{rag_answer}\n\n"
                        f"Additional recent information: {web_answer}"
                    )
                sources.append("web_search")
                used_web = True
                return final_answer, sources, used_web

    return rag_answer, sources, used_web

def add_document_to_index(text: str, source_name: str) -> int:

    if not text.strip():
        logger.warning(f"Empty text for '{source_name}' — not added to index.")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "\u0964", ".", " ", ""],
    )
    doc    = Document(
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

    logger.info(f"Dynamically indexed {len(chunks)} chunks from '{source_name}'")
    return len(chunks)
