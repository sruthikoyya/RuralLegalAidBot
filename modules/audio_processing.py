
import logging
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from modules.config import (
    WHISPER_MODEL_SIZE, WHISPER_DEVICE,
    TTS_MODEL, TTS_DEVICE,
)

logger = logging.getLogger(__name__)

_whisper_model = None
_tts_model     = None
_tts_tokenizer = None

def _load_whisper():
    """Load Whisper STT model (lazy, GPU if available)."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        device = WHISPER_DEVICE if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper '{WHISPER_MODEL_SIZE}' on {device} ...")
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)
        logger.info(f"Whisper loaded ✓  (device={device})")
    return _whisper_model

def transcribe_audio(audio_path: str, language: str = "te") -> str:
    
    model    = _load_whisper()
    use_fp16 = torch.cuda.is_available()

    logger.info(f"Transcribing audio: {os.path.basename(audio_path)}")
    try:
        result = model.transcribe(
            audio_path,
            language=language,
            fp16=use_fp16,
            condition_on_previous_text=False,  # prevents hallucinations on silence
        )
        transcript = result["text"].strip()
        detected_lang = result.get("language", "unknown")
        logger.info(
            f"Transcription complete: '{transcript[:80]}{'...' if len(transcript) > 80 else ''}' "
            f"(detected language: {detected_lang})"
        )
        return transcript
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        raise RuntimeError(f"Audio transcription failed: {e}") from e

def transcribe_telugu_audio(audio_path: str) -> str:
    """Alias for transcribe_audio with language='te' (backward compatible)."""
    return transcribe_audio(audio_path, language="te")

def _load_tts():
    """Load Facebook MMS-TTS model (lazy, CPU)."""
    global _tts_model, _tts_tokenizer
    if _tts_model is None:
        from transformers import VitsModel, AutoTokenizer
        logger.info(f"Loading TTS model ({TTS_MODEL}) on CPU ...")
        _tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL)
        _tts_model     = VitsModel.from_pretrained(TTS_MODEL)
        _tts_model.eval()
        logger.info("TTS loaded ✓")
    return _tts_model, _tts_tokenizer


def generate_telugu_speech(telugu_text: str, output_path: str) -> str:

    if not telugu_text.strip():
        logger.warning("Empty text passed to TTS — skipping")
        return output_path

    model, tokenizer = _load_tts()
    logger.info(f"Generating speech → {os.path.basename(output_path)}")

    try:
        inputs = tokenizer(telugu_text, return_tensors="pt")
        with torch.no_grad():
            audio_output = model(**inputs).waveform

        audio_data = audio_output.squeeze().cpu().numpy().astype(np.float32)

        # Normalize to ±0.95 to prevent clipping
        max_val = np.abs(audio_data).max()
        if max_val > 0:
            audio_data = audio_data / max_val * 0.95

        sf.write(output_path, audio_data, model.config.sampling_rate)
        duration = len(audio_data) / model.config.sampling_rate
        logger.info(f"Audio saved: {output_path} ({duration:.1f}s)")
        return output_path

    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise RuntimeError(f"Speech synthesis failed: {e}") from e


def cleanup_old_temp_files(temp_dir: str, max_age_seconds: int = 3600) -> int:
    
    now     = time.time()
    deleted = 0
    try:
        for f in Path(temp_dir).glob("*"):
            if f.is_file() and (now - f.stat().st_mtime) > max_age_seconds:
                try:
                    f.unlink()
                    deleted += 1
                except OSError as e:
                    logger.debug(f"Could not delete {f}: {e}")
    except Exception as e:
        logger.debug(f"Temp cleanup scan error: {e}")

    if deleted:
        logger.info(f"Cleaned up {deleted} old temp file(s) from {temp_dir}")
    return deleted
