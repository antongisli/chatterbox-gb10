# Chatterbox TTS API Documentation

REST API for text-to-speech synthesis with voice cloning support.

**Base URL**: `http://localhost:10050`

---

## Quick Start

```bash
# Start the API server
python tts_api.py --port 10050

# Or with a preloaded voice
python tts_api.py --preload-voice /path/to/reference.wav
```

---

## Endpoints

### Status & Health

#### `GET /`
Get server status and loaded models.

**Response:**
```json
{
  "status": "ok",
  "device": "cuda",
  "turbo_loaded": true,
  "original_loaded": false,
  "voices": ["default", "jinx"],
  "cuda_available": true,
  "cuda_device": "NVIDIA GH200 480GB"
}
```

#### `GET /health`
Simple health check.

**Response:**
```json
{"status": "healthy"}
```

---

### Voice Management

API voices are stored persistently on disk at `/workspace/chatterbox/voices/`. To persist across container restarts, mount this directory:

```bash
-v /path/to/voices:/workspace/chatterbox/voices
```

#### `GET /voices`
List all precomputed voices.

**Response:**
```json
[
  {"name": "default", "model_type": "turbo"},
  {"name": "jinx", "model_type": "turbo"}
]
```

#### `POST /voices/upload`
Upload reference audio to create a precomputed voice.

**Form Data:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `files` | File[] | Yes | Audio files (WAV, OGG, MP3, FLAC, M4A) |
| `name` | string | No | Voice name (auto-generated if omitted) |
| `model` | string | No | Model type: `turbo` or `original` (default: `turbo`) |

**Example:**
```bash
curl -X POST http://localhost:10050/voices/upload \
  -F "files=@jinx_clip1.ogg" \
  -F "files=@jinx_clip2.ogg" \
  -F "name=jinx" \
  -F "model=turbo"
```

**Response:**
```json
{
  "name": "jinx",
  "duration": 12.5,
  "model_type": "turbo",
  "message": "Voice 'jinx' created with 12.5s of audio"
}
```

#### `DELETE /voices/{name}`
Delete a precomputed voice.

**Example:**
```bash
curl -X DELETE http://localhost:10050/voices/jinx
```

---

### Text-to-Speech

#### `POST /tts`
Synthesize speech using a precomputed voice.

**Request Body (JSON):**
```json
{
  "text": "Hello, this is a test!",
  "voice": "jinx",
  "model": "turbo",
  "temperature": 0.8,
  "top_p": 0.95,
  "top_k": 1000,
  "min_p": 0.0,
  "repetition_penalty": 1.2,
  "seed": 0,
  "norm_loudness": true,
  "output_format": "wav"
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | *required* | Text to synthesize (max 500 chars) |
| `voice` | string | *required* | Name of precomputed voice |
| `model` | string | `turbo` | Model: `turbo` or `original` |
| `temperature` | float | `0.8` | Sampling temperature (0.05-2.0) |
| `top_p` | float | `0.95` | Top-p sampling (0.0-1.0) |
| `top_k` | int | `1000` | Top-k sampling (Turbo only, 0-2000) |
| `min_p` | float | `0.0` | Min-p sampling (0.0-1.0) |
| `repetition_penalty` | float | `1.2` | Repetition penalty (1.0-2.0) |
| `seed` | int | `0` | Random seed (0 = random) |
| `exaggeration` | float | `0.5` | Exaggeration (Original only, 0.25-2.0) |
| `cfg_weight` | float | `0.5` | CFG weight (Original only, 0.0-1.0) |
| `norm_loudness` | bool | `true` | Normalize loudness (Turbo only) |
| `output_format` | string | `wav` | Output: `wav` or `base64` |

**Example (WAV response):**
```bash
curl -X POST http://localhost:10050/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "voice": "jinx"}' \
  --output output.wav
```

**Example (Base64 response):**
```bash
curl -X POST http://localhost:10050/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello!", "voice": "jinx", "output_format": "base64"}'
```

**Base64 Response:**
```json
{
  "audio": "UklGRv4...",
  "sample_rate": 24000,
  "duration": 1.5,
  "generation_time": 0.45,
  "format": "wav"
}
```

**WAV Response Headers:**
- `X-Audio-Duration`: Audio duration in seconds
- `X-Generation-Time`: Generation time in seconds
- `X-Sample-Rate`: Sample rate (24000)

---

#### `POST /tts/with-audio`
Synthesize speech with inline reference audio (one-shot).

**Form Data:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | *required* | Text to synthesize |
| `files` | File[] | *required* | Reference audio files |
| `model` | string | `turbo` | Model type |
| `temperature` | float | `0.8` | Sampling temperature |
| `top_p` | float | `0.95` | Top-p sampling |
| `top_k` | int | `1000` | Top-k sampling |
| `min_p` | float | `0.0` | Min-p sampling |
| `repetition_penalty` | float | `1.2` | Repetition penalty |
| `seed` | int | `0` | Random seed |
| `exaggeration` | float | `0.5` | Exaggeration (Original) |
| `cfg_weight` | float | `0.5` | CFG weight (Original) |
| `norm_loudness` | bool | `true` | Normalize loudness |
| `cache_voice` | bool | `false` | Cache voice for reuse |
| `voice_name` | string | *auto* | Name for cached voice |

**Example:**
```bash
curl -X POST http://localhost:10050/tts/with-audio \
  -F "text=Hello, this is Jinx speaking!" \
  -F "files=@jinx_clip.ogg" \
  -F "cache_voice=true" \
  -F "voice_name=jinx" \
  --output output.wav
```

---

### Model Management

#### `POST /models/load`
Explicitly load a model into GPU memory.

**Form Data:**
| Field | Type | Description |
|-------|------|-------------|
| `model` | string | `turbo` or `original` |

**Example:**
```bash
curl -X POST http://localhost:10050/models/load -F "model=original"
```

#### `POST /models/unload`
Unload a model to free GPU memory.

**Example:**
```bash
curl -X POST http://localhost:10050/models/unload -F "model=original"
```

---

## CLI Options

```bash
python tts_api.py [OPTIONS]

Options:
  --host TEXT           Host to bind to (default: 0.0.0.0)
  --port INT            Port to listen on (default: 10050)
  --preload-voice PATH  Preload a voice on startup
  --preload-turbo       Preload Turbo model (default: true)
  --preload-original    Preload Original model
  --no-preload-turbo    Don't preload Turbo model
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PRELOAD_VOICE` | Path to audio file for default voice |
| `PRELOAD_TURBO` | Set to `0` to disable Turbo preload |
| `PRELOAD_ORIGINAL` | Set to `1` to preload Original model |
| `HF_TOKEN` | HuggingFace token for model download |
| `API_DEBUG` | Set to `1` for verbose request/response logging |
| `DEBUG` | Set to `1` for startup script debug output |

---

## Docker Usage

The container uses a startup script that manages services. **Do not override the command** unless you want to run a specific script.

```bash
# Build
docker build -t chatterbox-tts .

# Run API server only (default)
docker run --gpus all -it --rm \
  -e HF_TOKEN=hf_your_token \
  -e HF_HOME=/cache/hf \
  -v /path/to/hf-cache:/cache/hf \
  -p 10050:10050 \
  chatterbox-tts

# Run API + Gradio Turbo UI
docker run --gpus all -it --rm \
  -e HF_TOKEN=hf_your_token \
  -e HF_HOME=/cache/hf \
  -v /path/to/hf-cache:/cache/hf \
  -e START_GRADIO_TURBO=1 \
  -e GRADIO_SHARE=1 \
  -p 10050:10050 -p 7860:7860 \
  chatterbox-tts

# Debug mode (verbose logging)
docker run --gpus all -it --rm \
  -e HF_TOKEN=hf_your_token \
  -e DEBUG=1 \
  -p 10050:10050 \
  chatterbox-tts
```

---

## Python Client Example

```python
import requests

API_URL = "http://localhost:10050"

# Upload a voice
with open("jinx_clip.ogg", "rb") as f:
    resp = requests.post(
        f"{API_URL}/voices/upload",
        files={"files": f},
        data={"name": "jinx", "model": "turbo"}
    )
    print(resp.json())

# Generate speech
resp = requests.post(
    f"{API_URL}/tts",
    json={
        "text": "Ready to cause some chaos?",
        "voice": "jinx",
        "temperature": 0.7,
    }
)

# Save audio
with open("output.wav", "wb") as f:
    f.write(resp.content)

print(f"Duration: {resp.headers['X-Audio-Duration']}s")
print(f"Generation time: {resp.headers['X-Generation-Time']}s")
```

---

## Performance Tips

1. **Precompute voices** - Upload voices once, reuse for all requests
2. **Use Turbo model** - Faster than Original, especially on ARM
3. **Keep model loaded** - First request loads model (~5-10s), subsequent requests are fast
4. **Batch text** - Longer text is more efficient than many short requests
5. **Use base64 format** - Avoids file I/O if you need to process audio in memory

---

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad request (invalid parameters, insufficient audio) |
| 404 | Voice not found |
| 500 | Internal server error (generation failed) |
