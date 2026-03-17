# RuralLegalAidBot

**Multilingual Agentic RAG System for Rural Legal Aid — Telugu and English**

---

## Overview

RuralLegalAidBot is a voice-enabled legal information system for rural communities in India, primarily targeting the 80 million Telugu-speaking population of Andhra Pradesh and Telangana. It answers legal queries from indexed government documents using a multi-stage RAG pipeline with an agentic tool-calling layer, generating responses natively in Telugu or English via voice or text.

---

## System Architecture

```
Audio Input
    |
[ Whisper STT — CUDA ]
    |
User Query (Telugu / English)
    |
    +---> [ Calculator Agent — AST evaluator ]
    |
[ Gemini 2.0 Flash — Translate query to English ]
    |
[ ChromaDB — MMR Retrieval — fetch_k=16 ]
    |
[ FlashRank Reranker — top_k=4 ]
    |
[ Gemini 2.0 Flash — Answer in user's language ]
    |
[ Web Search Agent — DuckDuckGo — post-generation ]
    |
[ Facebook MMS-TTS — Telugu speech ]
    |
Audio + Text Response
```

---

## Component Reference

### 1. Document Ingestion and Text Extraction

PyMuPDF handles PDFs with a digital-text-first approach — if a page yields fewer than 50 characters, it is treated as scanned and passed to Tesseract OCR at 200 DPI with the `tel+eng` language pack. DOCX files are parsed for both paragraphs and tables (critical for government scheme documents with eligibility grids). Images use Tesseract with grayscale conversion and LSTM engine mode for accuracy on printed legal notices.

| Format | Extractor | OCR Fallback |
|---|---|---|
| .pdf | PyMuPDF (fitz) | Tesseract tel+eng |
| .docx / .doc | python-docx | No |
| .txt | pathlib | No |
| .png / .jpg / .jpeg | Pillow + pytesseract | Yes |

---

### 2. Text Chunking

`RecursiveCharacterTextSplitter` with a legal-aware separator hierarchy is used because legal documents have structured boundaries (Section, Article, Rule, Clause) that must not be split mid-provision. The Telugu danda (U+0964) is included as a sentence boundary for Telugu-script documents. Chunk size is 500 characters with 100-character overlap to keep legal provisions intact while fitting within the reranker's context window.

**Separator priority:** `\n\nSection` → `\n\nArticle` → `\n\nRule` → `\n\nClause` → `\n\n` → `\u0964` → `.` → ` ` → ``

---

### 3. Embedding Model

**Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (CPU)

Trained on 50+ language parallel corpora including Telugu, it produces semantically aligned cross-lingual vectors — a Telugu query and its English document equivalent land close in vector space, enabling cross-lingual retrieval without translating every document. It runs on CPU to preserve all GPU VRAM for Whisper. Embeddings are L2-normalised at inference time.

---

### 4. Vector Store and Retrieval

**Database:** ChromaDB (persistent, no separate server required)

MMR retrieval is used over standard cosine similarity to avoid returning near-duplicate chunks when the same legal provision appears across multiple document sections. The pipeline fetches 32 candidates, MMR-filters to 16 for diversity, then the reranker compresses to the final 4 chunks passed to the LLM.

---

### 5. Reranking

**Model:** FlashRank `ms-marco-MiniLM-L-12-v2` (CPU, ONNX-optimised)

Cross-encoders score (query, chunk) pairs jointly with full attention, giving significantly more accurate relevance scores than the bi-encoder used for retrieval. FlashRank adds ~150-200ms latency with a measured 30% improvement in answer precision. It degrades gracefully — if not installed, the system falls back to MMR-ordered results without error.

---

### 6. Language Model and Generation

**Primary:** Gemini 2.0 Flash (closed source, cloud API)
**Fallback:** Mistral-7B-Instruct-v0.2 GGUF via LlamaCpp (open source, local CUDA)

Gemini serves three roles: (1) translating the user query to English for retrieval; (2) generating the final answer in the user's detected language from retrieved context; (3) translating Telugu answers to English for the UI toggle. All Gemini calls include exponential backoff retry for the 15 req/min free-tier rate limit. The system auto-discovers the best available Gemini model for the configured API key.

---

### 7. Agent Layer

**Calculator Agent (pre-retrieval):** Regex detection of math intent (EMI, percentages, land area, arithmetic) followed by a whitelist-only AST evaluator. Only `Add`, `Sub`, `Mult`, `Div`, `Pow`, `Mod` nodes are permitted — all builtins are rejected to prevent code injection. Results are injected into the Gemini prompt as verified numerical context.

**Web Search Agent (post-generation):** Activates when the RAG answer contains uncertainty phrases or the query contains temporal keywords ("latest", "2025", "తాజా"). Calls DuckDuckGo with `region=in-en` biased toward `.gov.in` and `.nic.in` sources, then passes results to a second Gemini call to enrich the original answer.

---

### 8. Speech-to-Text

**Model:** OpenAI Whisper small (open source, CUDA, ~900 MB VRAM)

The only GPU-resident model in the stack. `condition_on_previous_text=False` prevents hallucinations on silent or low-quality rural audio. Language is fixed to `te` but the multilingual model handles English input correctly regardless.

| Size | VRAM | Latency |
|---|---|---|
| base | ~500 MB | ~1.5s |
| small | ~900 MB | ~2.5s |
| medium | ~3.0 GB | ~4.0s |

---

### 9. Text-to-Speech

**Model:** Facebook MMS-TTS `facebook/mms-tts-tel` (open source, CPU)

VITS-based neural TTS with native Telugu support. Runs on CPU to keep the GPU free for Whisper. Output waveform is peak-normalised to 0.95 full-scale before writing to WAV. TTS failures are non-fatal — the endpoint returns text with `audio_url: null` rather than an HTTP error. Generated files are auto-deleted after one hour.

---

## Model Summary

| Component | Model | Type | Device |
|---|---|---|---|
| STT | OpenAI Whisper small | Open source | CUDA |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 | Open source | CPU |
| Reranker | ms-marco-MiniLM-L-12-v2 (FlashRank) | Open source | CPU |
| LLM (primary) | Gemini 2.0 Flash | Closed source (API) | Cloud |
| LLM (fallback) | Mistral-7B-Instruct-v0.2 GGUF | Open source | CUDA |
| TTS | Facebook MMS-TTS (tel) | Open source | CPU |

---

## Project Structure

```
rurallegalaidbot/
|-- main.py                  # FastAPI app and route handlers
|-- requirements.txt
|-- Dockerfile
|-- docker-compose.yml
|-- .env.example
|
|-- modules/
|   |-- config.py            # All config loaded from environment variables
|   |-- gemini_qa.py         # Core RAG pipeline (retrieval, reranking, generation)
|   |-- agents.py            # Calculator and web search agents
|   |-- audio_processing.py  # Whisper STT and MMS-TTS
|   |-- doc_extraction.py    # PDF, DOCX, TXT, image extractors
|   |-- qa_system.py         # Document indexing and LlamaCpp fallback chain
|
|-- templates/
|   |-- index.html           # Bilingual Telugu/English web UI
|
|-- scripts/
|   |-- index_documents.py   # Batch document indexing CLI
|
|-- data/
    |-- legal_docs/          # Source documents (PDF, DOCX, TXT)
    |-- chroma_db/           # ChromaDB vector index (auto-created)
    |-- temp/                # Temporary audio files (1-hour TTL)
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | (required) | Google AI Studio API key |
| `GEMINI_MODEL` | auto-detect | Gemini model; auto-selects best available |
| `GEMINI_MAX_TOKENS` | 1024 | Max tokens per response |
| `GEMINI_TEMPERATURE` | 0.1 | Low temperature for factual legal answers |
| `EMBEDDING_MODEL` | paraphrase-multilingual-MiniLM-L12-v2 | HuggingFace embedding model |
| `EMBEDDING_DEVICE` | cpu | Embedding inference device |
| `CHROMA_COLLECTION` | rural_legal_docs | ChromaDB collection name |
| `RETRIEVAL_K` | 4 | Final chunks passed to LLM |
| `RETRIEVAL_FETCH_K` | 16 | Chunks fetched before reranking |
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `ENABLE_RERANKER` | true | FlashRank cross-encoder reranking |
| `WHISPER_MODEL_SIZE` | small | Whisper variant (base/small/medium/large) |
| `WHISPER_DEVICE` | cuda | Whisper device |
| `TTS_MODEL` | facebook/mms-tts-tel | HuggingFace TTS model |
| `ENABLE_WEB_SEARCH` | true | DuckDuckGo web search agent |
| `ENABLE_CALCULATOR` | true | AST calculator agent |
| `TEMP_MAX_AGE_SECONDS` | 3600 | Audio file TTL in seconds |
| `LLM_MODEL_PATH` | ./mistral-7b-instruct-v0.2.Q4_K_M.gguf | GGUF path for local fallback |
| `OCR_LANG` | tel+eng | Tesseract language pack |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web UI |
| POST | `/api/chat` | Chat — accepts `text` and/or `audio` form fields |
| POST | `/api/upload_doc` | Upload and index a document |
| GET | `/api/health` | Liveness check |
| GET | `/api/status` | Model load status and chunk count |

**Chat response schema:**
```json
{
  "query_original": "string",
  "answer_primary": "string",
  "answer_en":      "string",
  "sources":        ["string"],
  "audio_url":      "string or null",
  "used_web":       "boolean",
  "agents_used":    { "calculator": "string or null", "web_search": "boolean or null" },
  "processing_ms":  "integer"
}
```

---

## VRAM Budget (RTX 3050 6 GB)

| Component | Device | Memory |
|---|---|---|
| Whisper small | CUDA | ~900 MB VRAM |
| Gemini 2.0 Flash | Cloud | 0 MB |
| MiniLM embeddings | CPU | ~200 MB RAM |
| FlashRank reranker | CPU | ~120 MB RAM |
| MMS-TTS | CPU | ~250 MB RAM |

---

## Troubleshooting

**GEMINI_API_KEY not set** — Get a free key at https://aistudio.google.com/app/apikey, add to `.env`, restart.

**ChromaDB empty** — Run `python scripts/index_documents.py` after adding documents to `data/legal_docs/`.

**FlashRank not installed** — `pip install flashrank`. System falls back to unranked MMR results.

**Poor Telugu transcription** — Set `WHISPER_MODEL_SIZE=medium` (requires ~3 GB VRAM).

**Gemini 429 rate limit** — Auto-retried with backoff. For production, upgrade to a paid Gemini tier.

**Tesseract language pack missing** — `sudo apt-get install tesseract-ocr-tel`.