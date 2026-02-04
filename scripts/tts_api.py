#!/usr/bin/env python3
"""
Chatterbox TTS API Server

A FastAPI-based REST API for text-to-speech synthesis using Chatterbox models.
Supports both Turbo and Original models with voice cloning capabilities.

Features:
- Turbo and Original model support
- Voice embedding precomputation and caching
- Multi-file reference audio upload
- Configurable generation parameters
- Health checks and status endpoints

Usage:
    python tts_api.py [--host 0.0.0.0] [--port 10050] [--preload-voice /path/to/audio.wav]
"""

import argparse
import base64
import hashlib
import io
import logging
import os
import random
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field
from scipy.io import wavfile

DEBUG = os.getenv("API_DEBUG", "0") == "1"
LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Global state
state = {
    "turbo_model": None,
    "original_model": None,
    "voices": {},  # name -> {"conds": Conditionals, "model_type": "turbo"|"original"}
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Constants
MIN_DURATION = 5.0
TARGET_DURATION = 10.0
S3GEN_SR = 24000


# ============================================================
# Pydantic Models
# ============================================================

class TTSRequest(BaseModel):
    """Request body for TTS generation."""
    text: str = Field(..., description="Text to synthesize", max_length=500)
    voice: Optional[str] = Field(None, description="Name of precomputed voice to use")
    model: str = Field("turbo", description="Model to use: 'turbo' or 'original'")
    
    # Common parameters
    temperature: float = Field(0.8, ge=0.05, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(1000, ge=0, le=2000, description="Top-k sampling (Turbo only)")
    min_p: float = Field(0.0, ge=0.0, le=1.0, description="Min-p sampling")
    repetition_penalty: float = Field(1.2, ge=1.0, le=2.0, description="Repetition penalty")
    seed: int = Field(0, description="Random seed (0 for random)")
    
    # Original model specific
    exaggeration: float = Field(0.5, ge=0.25, le=2.0, description="Exaggeration (Original only)")
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0, description="CFG weight (Original only)")
    
    # Turbo specific
    norm_loudness: bool = Field(True, description="Normalize loudness (Turbo only)")
    
    # Output format
    output_format: str = Field("wav", description="Output format: 'wav' or 'base64'")


class VoiceUploadResponse(BaseModel):
    """Response for voice upload."""
    name: str
    duration: float
    model_type: str
    message: str


class VoiceInfo(BaseModel):
    """Information about a precomputed voice."""
    name: str
    model_type: str


class StatusResponse(BaseModel):
    """Server status response."""
    status: str
    device: str
    turbo_loaded: bool
    original_loaded: bool
    voices: list[str]
    cuda_available: bool
    cuda_device: Optional[str]


# ============================================================
# Helper Functions
# ============================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    if seed != 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)


def get_turbo_model():
    """Get or load the Turbo model."""
    if state["turbo_model"] is None:
        logger.info("Loading Chatterbox Turbo model...")
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        state["turbo_model"] = ChatterboxTurboTTS.from_pretrained(state["device"])
        logger.info("Turbo model loaded")
    return state["turbo_model"]


def get_original_model():
    """Get or load the Original model."""
    if state["original_model"] is None:
        logger.info("Loading Chatterbox Original model...")
        from chatterbox.tts import ChatterboxTTS
        state["original_model"] = ChatterboxTTS.from_pretrained(state["device"])
        logger.info("Original model loaded")
    return state["original_model"]


def process_audio_files(files: list[tuple[str, bytes]]) -> tuple[str, float]:
    """
    Process uploaded audio files, concatenating if needed.
    Returns: (temp_file_path, total_duration)
    """
    file_info = []
    total_duration = 0.0
    
    for filename, content in files:
        # Save to temp file for librosa
        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            y, sr = librosa.load(temp_path, sr=None)
            duration = len(y) / sr
            if duration > 0:
                file_info.append((temp_path, duration, y, sr))
                total_duration += duration
        except Exception as e:
            logger.warning(f"Failed to load {filename}: {e}")
            os.unlink(temp_path)
    
    if not file_info:
        raise HTTPException(status_code=400, detail="No valid audio files provided")
    
    if total_duration < MIN_DURATION:
        # Clean up temp files
        for path, _, _, _ in file_info:
            os.unlink(path)
        raise HTTPException(
            status_code=400, 
            detail=f"Total audio duration {total_duration:.1f}s is less than minimum {MIN_DURATION}s"
        )
    
    # If single file, return it directly
    if len(file_info) == 1:
        return file_info[0][0], total_duration
    
    # Concatenate files
    combined = []
    current_duration = 0.0
    
    for path, duration, y, sr in file_info:
        # Resample to S3GEN_SR if needed
        if sr != S3GEN_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=S3GEN_SR)
        combined.append(y)
        current_duration += duration
        os.unlink(path)  # Clean up individual temp files
        if current_duration >= TARGET_DURATION:
            break
    
    # Save concatenated audio
    result = np.concatenate(combined)
    temp_path = tempfile.mktemp(suffix=".wav")
    wav_int16 = (result * 32767).astype(np.int16)
    wavfile.write(temp_path, S3GEN_SR, wav_int16)
    
    return temp_path, total_duration


def enable_debug() -> None:
    global DEBUG
    if DEBUG:
        return
    DEBUG = True
    logging.getLogger().setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)


def _maybe_debug(label: str, payload: object) -> None:
    if DEBUG:
        logger.debug("%s: %s", label, payload)


def generate_voice_name(files: list[tuple[str, bytes]]) -> str:
    """Generate a unique voice name from file contents."""
    hasher = hashlib.md5()
    for filename, content in files:
        hasher.update(filename.encode())
        hasher.update(content[:1024])  # First 1KB for speed
    return f"voice_{hasher.hexdigest()[:8]}"


# ============================================================
# Lifespan (startup/shutdown)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown."""
    logger.info(f"Starting TTS API server on device: {state['device']}")
    if DEBUG:
        routes = [f"{route.methods} {route.path}" for route in app.routes if hasattr(route, "methods")]
        logger.debug("Registered routes: %s", routes)
    
    # Preload models if requested
    preload_turbo = os.environ.get("PRELOAD_TURBO", "1") == "1"
    preload_original = os.environ.get("PRELOAD_ORIGINAL", "0") == "1"
    
    if preload_turbo:
        get_turbo_model()
    if preload_original:
        get_original_model()
    
    # Preload voice if specified
    preload_voice = os.environ.get("PRELOAD_VOICE")
    if preload_voice and os.path.exists(preload_voice):
        logger.info(f"Preloading voice from: {preload_voice}")
        try:
            model = get_turbo_model()
            model.prepare_conditionals(preload_voice)
            state["voices"]["default"] = {
                "conds": model.conds,
                "model_type": "turbo"
            }
            logger.info("Default voice preloaded")
        except Exception as e:
            logger.error(f"Failed to preload voice: {e}")
    
    yield
    
    logger.info("Shutting down TTS API server")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Chatterbox TTS API",
    description="Text-to-speech API with voice cloning support",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request, call_next):
    if not DEBUG:
        return await call_next(request)
    start_time = time.time()
    _maybe_debug("tts_request", request.model_dump())
    logger.debug("Request: %s %s", request.method, request.url.path)
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000
    logger.debug(
        "Response: %s %s -> %s (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# ============================================================
# Endpoints
# ============================================================

@app.get("/", response_model=StatusResponse)
async def get_status():
    """Get server status and loaded models."""
    cuda_device = None
    if torch.cuda.is_available():
        cuda_device = torch.cuda.get_device_name(0)
    
    return StatusResponse(
        status="ok",
        device=state["device"],
        turbo_loaded=state["turbo_model"] is not None,
        original_loaded=state["original_model"] is not None,
        voices=list(state["voices"].keys()),
        cuda_available=torch.cuda.is_available(),
        cuda_device=cuda_device,
    )


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}


@app.get("/debug/routes")
async def debug_routes():
    """List routes when debugging is enabled."""
    if not DEBUG:
        raise HTTPException(status_code=403, detail="Debug disabled")
    return [
        {"path": route.path, "methods": sorted(route.methods or [])}
        for route in app.routes
        if hasattr(route, "methods")
    ]


@app.get("/voices", response_model=list[VoiceInfo])
async def list_voices():
    """List all precomputed voices."""
    return [
        VoiceInfo(name=name, model_type=info["model_type"])
        for name, info in state["voices"].items()
    ]


@app.post("/voices/upload", response_model=VoiceUploadResponse)
async def upload_voice(
    files: list[UploadFile] = File(..., description="Audio files for voice cloning"),
    name: Optional[str] = Form(None, description="Voice name (auto-generated if not provided)"),
    model: str = Form("turbo", description="Model type: 'turbo' or 'original'"),
):
    """
    Upload reference audio files to create a precomputed voice.
    
    The voice embeddings will be computed and cached for fast generation.
    Supports WAV, OGG, MP3, FLAC, M4A formats.
    """
    _maybe_debug(
        "voice_upload_request",
        {
            "name": name,
            "model": model,
            "files": [f.filename for f in files],
        },
    )

    # Read file contents
    file_data = []
    for f in files:
        content = await f.read()
        file_data.append((f.filename, content))
    
    # Process audio files
    audio_path, duration = process_audio_files(file_data)
    
    try:
        # Generate name if not provided
        voice_name = name or generate_voice_name(file_data)
        
        # Get appropriate model
        if model == "turbo":
            m = get_turbo_model()
        elif model == "original":
            m = get_original_model()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
        
        # Compute embeddings
        logger.info(f"Computing embeddings for voice '{voice_name}'...")
        m.prepare_conditionals(audio_path)
        
        # Cache the voice
        state["voices"][voice_name] = {
            "conds": m.conds,
            "model_type": model,
        }
        
        logger.info(f"Voice '{voice_name}' cached successfully")
        
        return VoiceUploadResponse(
            name=voice_name,
            duration=duration,
            model_type=model,
            message=f"Voice '{voice_name}' created with {duration:.1f}s of audio",
        )
    
    finally:
        # Clean up temp file
        if os.path.exists(audio_path):
            os.unlink(audio_path)


@app.delete("/voices/{name}")
async def delete_voice(name: str):
    """Delete a precomputed voice."""
    if name not in state["voices"]:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    
    del state["voices"][name]
    return {"message": f"Voice '{name}' deleted"}


@app.post("/tts")
async def synthesize(request: TTSRequest):
    """
    Synthesize speech from text.
    
    Either provide a precomputed voice name, or upload reference audio separately first.
    """
    start_time = time.time()
    
    # Set seed
    set_seed(request.seed)
    
    # Get model
    if request.model == "turbo":
        model = get_turbo_model()
    elif request.model == "original":
        model = get_original_model()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")
    
    # Check voice
    audio_path = None
    if request.voice:
        if request.voice not in state["voices"]:
            _maybe_debug("available_voices", list(state["voices"].keys()))
            raise HTTPException(status_code=404, detail=f"Voice '{request.voice}' not found")
        
        voice_info = state["voices"][request.voice]
        if voice_info["model_type"] != request.model:
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{request.voice}' was created for {voice_info['model_type']} model, not {request.model}"
            )
        
        model.conds = voice_info["conds"]
    else:
        raise HTTPException(
            status_code=400,
            detail="No voice specified. Upload a voice first with POST /voices/upload"
        )
    
    # Generate
    try:
        if request.model == "turbo":
            wav = model.generate(
                request.text,
                audio_prompt_path=audio_path,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                min_p=request.min_p,
                repetition_penalty=request.repetition_penalty,
                norm_loudness=request.norm_loudness,
            )
        else:  # original
            wav = model.generate(
                request.text,
                audio_prompt_path=audio_path,
                temperature=request.temperature,
                top_p=request.top_p,
                min_p=request.min_p,
                repetition_penalty=request.repetition_penalty,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight,
            )
        
        # Convert to numpy
        wav_np = wav.squeeze().cpu().numpy()
        
        # Calculate duration
        audio_duration = len(wav_np) / model.sr
        generation_time = time.time() - start_time
        
        # Return based on format
        if request.output_format == "base64":
            buffer = io.BytesIO()
            wav_int16 = (wav_np * 32767).astype(np.int16)
            wavfile.write(buffer, model.sr, wav_int16)
            audio_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return JSONResponse({
                "audio": audio_base64,
                "sample_rate": model.sr,
                "duration": audio_duration,
                "generation_time": generation_time,
                "format": "wav",
            })
        else:
            # Return raw WAV
            buffer = io.BytesIO()
            wav_int16 = (wav_np * 32767).astype(np.int16)
            wavfile.write(buffer, model.sr, wav_int16)
            
            return Response(
                content=buffer.getvalue(),
                media_type="audio/wav",
                headers={
                    "X-Audio-Duration": str(audio_duration),
                    "X-Generation-Time": str(generation_time),
                    "X-Sample-Rate": str(model.sr),
                }
            )
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/with-audio")
async def synthesize_with_audio(
    text: str = Form(..., description="Text to synthesize"),
    files: list[UploadFile] = File(..., description="Reference audio files"),
    model: str = Form("turbo", description="Model: 'turbo' or 'original'"),
    temperature: float = Form(0.8),
    top_p: float = Form(0.95),
    top_k: int = Form(1000),
    min_p: float = Form(0.0),
    repetition_penalty: float = Form(1.2),
    seed: int = Form(0),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5),
    norm_loudness: bool = Form(True),
    cache_voice: bool = Form(False, description="Cache the voice for future use"),
    voice_name: Optional[str] = Form(None, description="Name for cached voice"),
):
    """
    Synthesize speech with inline reference audio.
    
    This is a convenience endpoint that combines voice upload and synthesis.
    Use cache_voice=true to save the voice for future requests.
    """
    start_time = time.time()
    
    # Set seed
    set_seed(seed)
    
    # Read file contents
    file_data = []
    for f in files:
        content = await f.read()
        file_data.append((f.filename, content))
    
    # Process audio files
    audio_path, duration = process_audio_files(file_data)
    
    try:
        # Get model
        if model == "turbo":
            m = get_turbo_model()
        elif model == "original":
            m = get_original_model()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
        
        # Generate
        if model == "turbo":
            wav = m.generate(
                text,
                audio_prompt_path=audio_path,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                norm_loudness=norm_loudness,
            )
        else:
            wav = m.generate(
                text,
                audio_prompt_path=audio_path,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
        
        # Cache voice if requested
        if cache_voice:
            name = voice_name or generate_voice_name(file_data)
            state["voices"][name] = {
                "conds": m.conds,
                "model_type": model,
            }
            logger.info(f"Voice '{name}' cached")
        
        # Convert to numpy
        wav_np = wav.squeeze().cpu().numpy()
        
        # Calculate duration
        audio_duration = len(wav_np) / m.sr
        generation_time = time.time() - start_time
        
        # Return raw WAV
        buffer = io.BytesIO()
        wav_int16 = (wav_np * 32767).astype(np.int16)
        wavfile.write(buffer, m.sr, wav_int16)
        
        return Response(
            content=buffer.getvalue(),
            media_type="audio/wav",
            headers={
                "X-Audio-Duration": str(audio_duration),
                "X-Generation-Time": str(generation_time),
                "X-Sample-Rate": str(m.sr),
                "X-Voice-Cached": str(cache_voice),
            }
        )
    
    finally:
        # Clean up temp file
        if os.path.exists(audio_path):
            os.unlink(audio_path)


@app.post("/models/load")
async def load_model(model: str = Form(..., description="Model to load: 'turbo' or 'original'")):
    """Explicitly load a model into memory."""
    if model == "turbo":
        get_turbo_model()
        return {"message": "Turbo model loaded"}
    elif model == "original":
        get_original_model()
        return {"message": "Original model loaded"}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")


@app.post("/models/unload")
async def unload_model(model: str = Form(..., description="Model to unload: 'turbo' or 'original'")):
    """Unload a model from memory to free GPU resources."""
    if model == "turbo":
        if state["turbo_model"] is not None:
            del state["turbo_model"]
            state["turbo_model"] = None
            torch.cuda.empty_cache()
            return {"message": "Turbo model unloaded"}
        return {"message": "Turbo model was not loaded"}
    elif model == "original":
        if state["original_model"] is not None:
            del state["original_model"]
            state["original_model"] = None
            torch.cuda.empty_cache()
            return {"message": "Original model unloaded"}
        return {"message": "Original model was not loaded"}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Chatterbox TTS API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=10050, help="Port to listen on")
    parser.add_argument("--preload-voice", help="Path to audio file for preloading default voice")
    parser.add_argument("--preload-turbo", action="store_true", default=True, help="Preload Turbo model")
    parser.add_argument("--preload-original", action="store_true", help="Preload Original model")
    parser.add_argument("--no-preload-turbo", action="store_true", help="Don't preload Turbo model")
    args = parser.parse_args()
    
    # Set environment variables for lifespan
    if args.preload_voice:
        os.environ["PRELOAD_VOICE"] = args.preload_voice
    if args.no_preload_turbo:
        os.environ["PRELOAD_TURBO"] = "0"
    if args.preload_original:
        os.environ["PRELOAD_ORIGINAL"] = "1"
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
