import logging
import os
import shutil
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ── Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── Local Modules 
from modules.config import TEMP_DIR, TEMP_MAX_AGE_SECONDS, ENABLE_WEB_SEARCH, ENABLE_CALCULATOR
from modules.audio_processing import (
    transcribe_telugu_audio,
    generate_telugu_speech,
    cleanup_old_temp_files,
)
from modules.doc_extraction import process_uploaded_file, get_supported_extensions
from modules.gemini_qa import get_qa_chain, query_legal_bot, add_document_to_index

#from modules.qa_system import get_qa_chain, query_legal_bot, add_document_to_index

from modules.audio_processing import _load_whisper, _load_tts



# ── FastAPI App 
app = FastAPI(
    title="RuralLegalAidBot",
    description="Multilingual RAG legal assistant for rural India — Telugu + English",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Templates & Static Files 
templates = Jinja2Templates(directory="templates")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/audio", StaticFiles(directory=str(TEMP_DIR)), name="audio")


# ── Startup 


@app.on_event("startup")
async def startup_event():
    logger.info("═" * 60)
    logger.info("  RuralLegalAidBot v2 starting...")
    logger.info(f"  Web search agent : {'ENABLED' if ENABLE_WEB_SEARCH else 'disabled'}")
    logger.info(f"  Calculator agent : {'ENABLED' if ENABLE_CALCULATOR else 'disabled'}")
    logger.info("═" * 60)

    chain = get_qa_chain()
    if chain:
        logger.info("Gemini RAG pipeline ready ✓")
    else:
        logger.warning(
            "⚠  Gemini not ready. Check GEMINI_API_KEY in .env\n"
            "   Get free key: https://aistudio.google.com/app/apikey"
        )

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "RuralLegalAidBot", "version": "2.0"}


@app.get("/api/status")
async def system_status():
    import torch
    from modules import audio_processing

    cuda_available = torch.cuda.is_available()

    try:
        from modules.gemini_qa import _gemini_client, _vectordb, _reranker
        gemini_loaded = _gemini_client is not None
        db_loaded     = _vectordb is not None
        reranker_on   = _reranker is not None
        db_count      = _vectordb._collection.count() if _vectordb else 0
    except Exception:
        gemini_loaded = db_loaded = reranker_on = False
        db_count = 0

    return {
        "version": "2.0",
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "vram_free_mb": (
            round(torch.cuda.mem_get_info()[0] / 1024 ** 2) if cuda_available else None
        ),
        "models": {
            "gemini_loaded":   gemini_loaded,
            "vectordb_loaded": db_loaded,
            "vectordb_chunks": db_count,
            "reranker_loaded": reranker_on,
            "whisper_loaded":  audio_processing._whisper_model is not None,
            "tts_loaded":      audio_processing._tts_model is not None,
        },
        "agents": {
            "web_search": ENABLE_WEB_SEARCH,
            "calculator": ENABLE_CALCULATOR,
        },
    }


@app.post("/api/chat")
async def process_chat(
    text:  str        = Form(None),
    audio: UploadFile = File(None),
):
    """
    Main chat endpoint — handles voice or text input in Telugu or English.

    Returns JSON:
      {
        "query_original": str,      # Original query (any language)
        "answer_primary": str,      # Answer in user's language
        "answer_en":      str,      # English answer (for UI toggle)
        "sources":        [str],    # Source document names
        "audio_url":      str|null, # URL to .wav audio
        "used_web":       bool,     # Web search agent triggered?
        "agents_used":    dict,     # Agent details
        "processing_ms":  int       # Total time
      }
    """
    t_start = time.time()
    cleanup_old_temp_files(str(TEMP_DIR), TEMP_MAX_AGE_SECONDS)

    # ── Step 1: Get user query ────────────────────────────────────────────────
    user_query = ""

    if audio and audio.filename:
        rand_hex   = os.urandom(4).hex()
        audio_path = str(TEMP_DIR / f"input_{rand_hex}_{audio.filename}")
        try:
            with open(audio_path, "wb") as f:
                shutil.copyfileobj(audio.file, f)
            user_query = transcribe_telugu_audio(audio_path)
        except Exception as e:
            logger.error(f"STT failed: {e}")
            raise HTTPException(status_code=500, detail=f"Audio transcription failed: {e}")
        finally:
            Path(audio_path).unlink(missing_ok=True)

    elif text and text.strip():
        user_query = text.strip()
    else:
        raise HTTPException(status_code=400, detail="Please provide text or audio input.")

    if not user_query:
        raise HTTPException(status_code=400, detail="Could not extract text from input.")

    # ── Step 2: RAG + Agents (Gemini handles translation internally) ──────────
    try:
        answer_primary, answer_en, sources, used_web, agents_used = query_legal_bot(user_query)
    except Exception as e:
        logger.error(f"RAG pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    # ── Step 3: TTS (non-fatal) ───────────────────────────────────────────────
    audio_url = None
    try:
        audio_filename = f"response_{os.urandom(6).hex()}.wav"
        audio_output   = str(TEMP_DIR / audio_filename)
        generate_telugu_speech(answer_primary, audio_output)
        audio_url = f"/audio/{audio_filename}"
    except Exception as e:
        logger.warning(f"TTS failed (non-fatal): {e}")

    processing_ms = int((time.time() - t_start) * 1000)
    logger.info(
        f"Chat complete in {processing_ms}ms | "
        f"web={used_web} | calc={agents_used.get('calculator') is not None} | "
        f"sources={sources}"
    )

    return {
        "query_original": user_query,
        "answer_primary": answer_primary,
        "answer_en":      answer_en,
        "sources":        sources,
        "audio_url":      audio_url,
        "used_web":       used_web,
        "agents_used":    agents_used,
        "processing_ms":  processing_ms,
    }


@app.post("/api/upload_doc")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a legal document to the knowledge base.
    Supported: PDF, DOCX, TXT, PNG, JPG
    """
    allowed_ext = set(get_supported_extensions())
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type '{ext}'. Allowed: {', '.join(sorted(allowed_ext))}"
        )

    safe_name = f"upload_{os.urandom(4).hex()}_{file.filename}"
    file_path = str(TEMP_DIR / safe_name)

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        extracted_text = process_uploaded_file(file_path)

        if not extracted_text.strip():
            return JSONResponse(
                status_code=422,
                content={"error": "No text could be extracted from this file."}
            )

        chunks_added = add_document_to_index(extracted_text, file.filename)

        return {
            "filename":             file.filename,
            "characters_extracted": len(extracted_text),
            "chunks_indexed":       chunks_added,
            "preview":              extracted_text[:500].strip() + ("..." if len(extracted_text) > 500 else ""),
            "message":              f"'{file.filename}' indexed — {chunks_added} chunks added.",
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        Path(file_path).unlink(missing_ok=True)


# ── Dev Entry Point
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
