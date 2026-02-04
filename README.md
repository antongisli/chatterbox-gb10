# Chatterbox GB10

This repo contains our custom Chatterbox deployment for the GB10 stack. See **docs/API.md** for REST API usage and endpoints.

## Features

Your own text-to-speech service with:

- **🎤 Voice Cloning** - Clone any voice from 5-15 seconds of reference audio
- **🌐 Web UI (Gradio)** - Point-and-click interface with voice selection and generation controls
- **🔌 REST API** - HTTP endpoints for integration with other services
- **🎙️ Microphone Recording** - Record your voice directly in the browser (requires HTTPS)
- **💾 Voice Management** - Save voices in-memory or persist via API storage
- **🚀 Optimized for ARM** - GraceHopper/GB10 compatible with CUDA JIT fixes
- **⚡ Multiple Models** - Choose between Turbo (fast, distilled) or Original (CFG control)
- **🎭 Paralinguistic Tags** - Add expressions like `[laugh]`, `[sigh]`, `[clear throat]`

This guide documents how to set up Chatterbox TTS in the NVIDIA PyTorch container:
- **No watermarking** (skip `resemble-perth`)
- **No torchaudio dependency** (use stubs)
- **ARM/GraceHopper compatible** (CUDA JIT fixes for complex tensor ops)

---

## What's Modified

| Component | Issue | Fix |
|-----------|-------|-----|
| `torchaudio` | Not installed, but code imports it | Stub with `compliance.kaldi` and `transforms.Resample` |
| `resemble-perth` | Watermarking library | Not installed → watermarking disabled |
| `s3tokenizer.py` | `.abs()` on complex tensors fails on ARM | Patch to use `.real**2 + .imag**2` |
| `torch` | Container has optimized version | Use `--no-deps` to prevent override |

---

## Option A: Use the Dockerfile (Recommended)

The Dockerfile includes all patches automatically.

### Build the image

```bash
cd /path/to/chatterbox
docker build -t chatterbox-tts .
```

### Run interactive shell

```bash
# Persist Hugging Face downloads in local ./hf_cache folder
mkdir -p ./hf_cache

docker run --gpus all -it --rm \
  -e HF_TOKEN=hf_your_token_here \
  -e HF_HOME=/cache/hf \
  -v "$PWD/hf_cache":/cache/hf \
  -v "$PWD/voices/jinx":/workspace/audio \
  chatterbox-tts bash
```

### Run API server

```bash
# Persist Hugging Face downloads in local ./hf_cache folder
mkdir -p ./hf_cache

docker run --gpus all -it --rm \
  -e HF_TOKEN=hf_your_token_here \
  -e HF_HOME=/cache/hf \
  -v "$PWD/hf_cache":/cache/hf \
  -p 10050:10050 \
  -v "$PWD/voices/jinx":/workspace/audio \
  chatterbox-tts \
  python tts_api.py --port 10050

# With preloaded voice
docker run --gpus all -it --rm \
  -e HF_TOKEN=hf_your_token_here \
  -e HF_HOME=/cache/hf \
  -v "$PWD/hf_cache":/cache/hf \
  -p 10050:10050 \
  -v "$PWD/voices/jinx":/workspace/audio \
  chatterbox-tts \
  python tts_api.py --preload-voice /workspace/audio/reference.wav
```

The model will be downloaded on first use from HuggingFace.

---

## Optional: Start Enhanced Gradio UIs

The Docker image now includes a startup script that can run the API and/or
the enhanced Gradio apps. By default it starts the API only.

### Gradio Web UI (default: Turbo on port 7860)

By default, Gradio Turbo starts automatically. Use `GRADIO_MODE` to switch:

- `GRADIO_MODE=turbo` (default) - Fast distilled model, port 7860
- `GRADIO_MODE=original` - CFG/exaggeration controls, port 7861  
- `GRADIO_MODE=none` - API only, no Gradio

```bash
mkdir -p ./hf_cache

# Default: Turbo UI
docker run --gpus all -it --rm \
  -e HF_TOKEN=hf_your_token_here \
  -e HF_HOME=/cache/hf \
  -v "$PWD/hf_cache":/cache/hf \
  -p 10050:10050 -p 7860:7860 \
  -v "$PWD/voices/jinx":/workspace/audio \
  chatterbox-tts

# Original UI (with CFG/exaggeration sliders)
docker run --gpus all -it --rm \
  -e HF_TOKEN=hf_your_token_here \
  -e HF_HOME=/cache/hf \
  -v "$PWD/hf_cache":/cache/hf \
  -e GRADIO_MODE=original \
  -p 10050:10050 -p 7861:7861 \
  -v "$PWD/voices/jinx":/workspace/audio \
  chatterbox-tts

# API only (no Gradio)
docker run --gpus all -it --rm \
  -e HF_TOKEN=hf_your_token_here \
  -e HF_HOME=/cache/hf \
  -v "$PWD/hf_cache":/cache/hf \
  -e GRADIO_MODE=none \
  -p 10050:10050 \
  chatterbox-tts
```

### Debug Mode

Enable verbose logging with `DEBUG=1`:

```bash
docker run --gpus all -it --rm \
  -e HF_TOKEN=hf_your_token_here \
  -e HF_HOME=/cache/hf \
  -v "$PWD/hf_cache":/cache/hf \
  -e DEBUG=1 \
  -p 10050:10050 -p 7860:7860 \
  chatterbox-tts
```

Debug mode shows:
- Service startup/shutdown details
- Process PIDs
- API request/response logging (when `API_DEBUG=1`)

### Optional: Sync API voices into Gradio

If the API server is running, the enhanced Gradio apps can fetch voices from
`GET /voices` using these env vars:

```bash
  -e SYNC_API_VOICES=1 \
  -e API_BASE_URL=http://localhost:10050
```

Then click **Refresh API Voices** in the Gradio UI.

---

## Voice Management

### How Voices Work

Voices are stored as **embeddings** (numerical representations of a speaker's voice characteristics). There are three types of voice storage:

| Type | Storage | Persistence | Source |
|------|---------|-------------|--------|
| **Default Voice** | In-memory | Per-session | Loaded from `/workspace/audio` on startup |
| **Saved Voices** | In-memory | Per-session | Created via "Save Voice" button in Gradio |
| **API Voices** | Disk | Persistent | Uploaded via API `/voices/upload` endpoint |

### Default Voice (Recommended)

Mount a folder containing audio files to `/workspace/audio`. The voice will be named after the folder:

```bash
# Voice will be named "jinx" (from folder name)
-v "$PWD/voices/jinx":/workspace/audio

# Or specify a custom name
-e DEFAULT_VOICE_NAME=my_character
```

**Requirements:**
- Audio files: WAV, OGG, MP3, FLAC, or M4A
- Minimum 5 seconds total duration
- Up to 15 seconds will be used (files are concatenated alphabetically)
- Clean audio without background music or reverb

### Voice Priority in Gradio

When you click "Generate", the app uses voices in this priority order:

1. **Uploaded files** - If you've uploaded new audio files, those are used first
2. **Saved voice** - If a saved voice is selected in the dropdown
3. **Cached embeddings** - If "Keep Embeddings" is checked and you've generated before

The **Active Voice** indicator shows which voice will be used.

### Saving Voices in Gradio

1. Upload reference audio files or record via microphone
2. Enter a name in "Save Voice As" (e.g., "jinx")
3. Click "Save Voice"
4. The voice appears in the dropdown and persists for the session

**Note:** Saved voices are lost when the container restarts. For persistent voices, use the API.

### Persistent Voices via API

Upload voices to the API for persistent storage:

```bash
curl -X POST http://localhost:10050/voices/upload \
  -F "name=jinx" \
  -F "audio=@/path/to/reference.wav"
```

API voices are stored in `/workspace/chatterbox/voices/` inside the container.
Mount this directory to persist across restarts:

```bash
-v /path/to/voices:/workspace/chatterbox/voices
```

### Microphone Recording

Microphone recording requires **HTTPS**. Use the public Gradio link:

1. Start with `-e GRADIO_SHARE=1`
2. Use the `https://xxxxx.gradio.live` URL (not localhost)
3. Click the lock icon in Chrome and allow microphone access

---

## Option B: Manual Setup (inside existing container)

### Prerequisites

```bash
docker run --gpus all -it --rm \
  -v /path/to/your/data:/workspace \
  nvcr.io/nvidia/pytorch:25.01-py3 bash
```

### Step 1: Set HuggingFace Token

```bash
export HF_TOKEN=hf_your_token_here
# Or add to ~/.bashrc for persistence:
echo 'export HF_TOKEN=hf_your_token_here' >> ~/.bashrc
```

### Step 2: Install Dependencies (WITHOUT overriding PyTorch)

**IMPORTANT**: The container comes with PyTorch pre-installed and optimized for NVIDIA GPUs.
We must NOT let pip override it. Use `--no-deps` when installing chatterbox.

```bash
# Install dependencies manually (excludes torch, torchaudio, resemble-perth)
pip install --no-cache-dir \
  "numpy>=1.24.0,<1.26.0" \
  librosa==0.11.0 \
  s3tokenizer \
  transformers==4.46.3 \
  diffusers==0.29.0 \
  conformer==0.3.2 \
  safetensors==0.5.3 \
  spacy-pkuseg \
  pykakasi==2.3.0 \
  gradio==5.44.1 \
  pyloudnorm \
  omegaconf \
  scipy \
  huggingface_hub
```

**Skipped packages**:
- `torch==2.6.0` - container already has optimized PyTorch
- `torchaudio==2.6.0` - replaced with minimal stub (Step 3)
- `resemble-perth==1.0.1` - watermarking library (skipping disables watermarks)

### Step 3: Create torchaudio Stubs

The model uses `torchaudio.compliance.kaldi.fbank` and `torchaudio.transforms.Resample`. 
We provide pure-torch implementations:

```bash
TA_BASE="/usr/local/lib/python3.12/dist-packages/torchaudio"
mkdir -p "$TA_BASE/compliance"

# Make torchaudio a package (with transforms)
cat > "$TA_BASE/__init__.py" <<'PY'
# Minimal torchaudio stub
from . import compliance, transforms
PY

# Make compliance a package
cat > "$TA_BASE/compliance/__init__.py" <<'PY'
# torchaudio.compliance package (stub)
PY

# Minimal kaldi module with fbank/mfcc
cat > "$TA_BASE/compliance/kaldi.py" <<'PY'
"""
Minimal torchaudio.compliance.kaldi stub.

Provides enough surface for models that call torchaudio.compliance.kaldi.fbank/mfcc.
Implemented with torch only. Not bit-identical to torchaudio/kaldi, but usually sufficient
for inference embeddings.

If you hit a missing function, the traceback will tell us what to add next.
"""
from __future__ import annotations
import math
import torch

def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)

def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def _mel_filterbank(n_mels: int, n_fft: int, sample_frequency: int,
                    f_min: float = 20.0, f_max: float | None = None,
                    device=None, dtype=None) -> torch.Tensor:
    if f_max is None:
        f_max = sample_frequency / 2.0
    n_freqs = n_fft // 2 + 1
    freqs = torch.linspace(0, sample_frequency / 2.0, n_freqs, device=device, dtype=dtype)
    m_min = _hz_to_mel(torch.tensor(f_min, device=device, dtype=dtype))
    m_max = _hz_to_mel(torch.tensor(f_max, device=device, dtype=dtype))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2, device=device, dtype=dtype)
    hz_pts = _mel_to_hz(m_pts)
    bins = torch.floor((n_fft + 1) * hz_pts / sample_frequency).to(torch.int64)
    fb = torch.zeros((n_mels, n_freqs), device=device, dtype=dtype)
    for m in range(n_mels):
        left, center, right = bins[m].item(), bins[m+1].item(), bins[m+2].item()
        if center == left:
            center += 1
        if right == center:
            right += 1
        if center > left:
            fb[m, left:center] = (torch.arange(left, center, device=device) - left) / (center - left)
        if right > center:
            fb[m, center:right] = (right - torch.arange(center, right, device=device)) / (right - center)
    return fb

def fbank(waveform: torch.Tensor,
          sample_frequency: float = 16000.0,
          frame_length: float = 25.0,
          frame_shift: float = 10.0,
          num_mel_bins: int = 80,
          dither: float = 0.0,
          energy_floor: float = 0.0,
          snip_edges: bool = True,
          window_type: str = "povey",
          preemphasis_coefficient: float = 0.97,
          **kwargs) -> torch.Tensor:
    """
    Rough replacement for torchaudio.compliance.kaldi.fbank.
    waveform: (channels, time) or (time,)
    returns: (frames, num_mel_bins)
    """
    if waveform.dim() == 2:
        waveform = waveform[0]
    waveform = waveform.to(torch.float32)

    sr = int(sample_frequency)
    if dither and dither > 0:
        waveform = waveform + dither * torch.randn_like(waveform)

    if preemphasis_coefficient and preemphasis_coefficient != 0.0:
        waveform = torch.cat([waveform[:1], waveform[1:] - preemphasis_coefficient * waveform[:-1]])

    win_length = int(sr * frame_length / 1000.0)
    hop_length = int(sr * frame_shift / 1000.0)
    n_fft = 1
    while n_fft < win_length:
        n_fft *= 2

    if snip_edges:
        num_frames = 1 + (waveform.numel() - win_length) // hop_length
        waveform = waveform[: num_frames * hop_length + win_length]
    else:
        pad = win_length // 2
        waveform = torch.nn.functional.pad(waveform, (pad, pad), mode="reflect")
        num_frames = 1 + (waveform.numel() - win_length) // hop_length

    frames = waveform.unfold(0, win_length, hop_length)

    if window_type in ("povey", "hanning", "hann"):
        w = torch.hann_window(win_length, periodic=True, device=frames.device, dtype=frames.dtype)
        if window_type == "povey":
            w = w.pow(0.85)
    else:
        w = torch.hann_window(win_length, periodic=True, device=frames.device, dtype=frames.dtype)

    frames = frames * w

    spec = torch.fft.rfft(frames, n=n_fft)
    mag = (spec.real**2 + spec.imag**2).clamp_min(1e-10)

    fb = _mel_filterbank(num_mel_bins, n_fft, sr, device=mag.device, dtype=mag.dtype)
    mel = mag @ fb.t()
    mel = mel.clamp_min(1e-10).log()

    if energy_floor and energy_floor > 0:
        mel = torch.maximum(mel, torch.tensor(math.log(energy_floor), device=mel.device, dtype=mel.dtype))

    return mel

def mfcc(waveform: torch.Tensor,
         sample_frequency: float = 16000.0,
         num_ceps: int = 13,
         num_mel_bins: int = 23,
         cepstral_lifter: float = 22.0,
         **kwargs) -> torch.Tensor:
    """
    Simple MFCC from fbank + DCT.
    returns: (frames, num_ceps)
    """
    fb = fbank(waveform, sample_frequency=sample_frequency, num_mel_bins=num_mel_bins, **kwargs)
    n = fb.size(-1)
    k = torch.arange(num_ceps, device=fb.device).unsqueeze(1)
    n0 = torch.arange(n, device=fb.device).unsqueeze(0)
    dct = torch.cos(math.pi / n * (n0 + 0.5) * k).to(fb.dtype)
    cep = fb @ dct.t()
    if cepstral_lifter and cepstral_lifter > 0:
        lift = 1.0 + 0.5 * cepstral_lifter * torch.sin(math.pi * torch.arange(num_ceps, device=fb.device) / cepstral_lifter)
        cep = cep * lift.to(cep.dtype)
    return cep
PY

# transforms.py - sinc-based Resample for high-quality audio resampling
cat > "$TA_BASE/transforms.py" <<'PY'
"""
Improved torchaudio.transforms stub with proper sinc-based resampling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Resample(nn.Module):
    """Resample waveform using polyphase sinc interpolation."""
    
    def __init__(self, orig_freq: int = 16000, new_freq: int = 16000,
                 lowpass_filter_width: int = 6, rolloff: float = 0.99,
                 beta: float = 14.769656459379492):
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        
        if orig_freq == new_freq:
            self.kernel = None
            return
        
        gcd = math.gcd(orig_freq, new_freq)
        self.up = new_freq // gcd
        self.down = orig_freq // gcd
        self.kernel = self._build_kernel(lowpass_filter_width, rolloff, beta)
    
    def _build_kernel(self, width: int, rolloff: float, beta: float) -> torch.Tensor:
        cutoff = min(self.up, self.down) * rolloff / max(self.up, self.down)
        kernel_width = width * max(self.up, self.down)
        kernel_size = kernel_width * 2 + 1
        idx = torch.arange(-kernel_width, kernel_width + 1, dtype=torch.float64)
        
        kernels = []
        for phase in range(self.up):
            t = idx / self.up - phase / self.up
            sinc = torch.where(t == 0, torch.ones_like(t),
                torch.sin(math.pi * cutoff * t) / (math.pi * t)) * cutoff
            window = torch.kaiser_window(kernel_size, periodic=False, beta=beta, dtype=torch.float64)
            kernel = sinc * window
            kernel = kernel / kernel.sum()
            kernels.append(kernel.float())
        return torch.stack(kernels)
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.orig_freq == self.new_freq:
            return waveform
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        device, dtype = waveform.device, waveform.dtype
        kernel = self.kernel.to(device=device, dtype=dtype)
        channels, length = waveform.shape
        pad_size = kernel.shape[1] // 2
        waveform_padded = F.pad(waveform, (pad_size, pad_size), mode='reflect')
        
        if self.up > 1:
            upsampled = torch.zeros(channels, waveform_padded.shape[1] * self.up, device=device, dtype=dtype)
            upsampled[:, ::self.up] = waveform_padded
        else:
            upsampled = waveform_padded
        
        upsampled = upsampled.unsqueeze(1)
        avg_kernel = kernel.mean(dim=0, keepdim=True).unsqueeze(0)
        filtered = F.conv1d(upsampled, avg_kernel, padding='same').squeeze(1)
        
        if self.down > 1:
            out_length = int(math.ceil(length * self.up / self.down))
            start = pad_size * self.up
            indices = (torch.arange(0, out_length, device=device) * self.down + start).clamp(0, filtered.shape[1] - 1).long()
            result = filtered[:, indices]
        else:
            start = pad_size * self.up
            result = filtered[:, start:start + length * self.up]
        
        result = result * self.up
        return result.squeeze(0) if squeeze else result


class Spectrogram(nn.Module):
    """Basic spectrogram (ARM-compatible, avoids .abs() on complex)."""
    def __init__(self, n_fft=400, hop_length=None, power=2.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 2
        self.power = power
        self.register_buffer('window', torch.hann_window(n_fft))
    
    def forward(self, waveform):
        stft = torch.stft(waveform, self.n_fft, self.hop_length,
            window=self.window.to(waveform.device), return_complex=True)
        spec = stft.real**2 + stft.imag**2
        if self.power != 2.0:
            spec = spec.pow(self.power / 2.0)
        return spec
PY
```

### Step 4: Clone and Install Chatterbox

```bash
# Clone the repo
git clone https://github.com/resemble-ai/chatterbox.git /workspace/chatterbox
cd /workspace/chatterbox

# Install WITHOUT dependencies (critical - prevents torch override)
pip install --no-deps -e .
```

### Step 5: ARM/GraceHopper CUDA JIT Fix

On ARM-based NVIDIA systems (Jetson, GraceHopper), the `.abs()` operation on complex 
tensors triggers CUDA JIT compilation that fails. Patch the s3tokenizer:

```bash
# Patch s3tokenizer.py to avoid .abs() on complex tensors
sed -i 's/magnitudes = stft\[\.\.\.*, :-1\]\.abs()\*\*2/stft_slice = stft[..., :-1]; magnitudes = stft_slice.real**2 + stft_slice.imag**2/' \
    /workspace/chatterbox/src/chatterbox/models/s3tokenizer/s3tokenizer.py

# Verify the patch was applied
grep -n "stft_slice" /workspace/chatterbox/src/chatterbox/models/s3tokenizer/s3tokenizer.py
```

You should see output like:
```
161:        stft_slice = stft[..., :-1]; magnitudes = stft_slice.real**2 + stft_slice.imag**2
```

### Step 6: Add NGC CLI to PATH (optional)

If you need NGC CLI:

```bash
echo 'export PATH="$PATH:/home/anton/ngc-cli"' >> ~/.bashrc
source ~/.bashrc
```

---

## Verification

```bash
# Check PyTorch is still the container's version (should show NVIDIA build info)
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check chatterbox loads
python3 -c "from chatterbox.tts_turbo import ChatterboxTurboTTS; print('Chatterbox OK')"

# Check torchaudio stub works
python3 -c "import torchaudio.compliance.kaldi as k; print('torchaudio stub OK')"
```

---

## Quick Test (Voice Cloning)

```bash
cd /workspace/chatterbox

# Generate with default voice
python3 -c "
from chatterbox.tts_turbo import ChatterboxTurboTTS
import numpy as np
from scipy.io import wavfile

model = ChatterboxTurboTTS.from_pretrained(device='cuda')
wav = model.generate('Hello, this is a test!')
wav_np = wav.squeeze().numpy()
wavfile.write('test_output.wav', model.sr, (wav_np * 32767).astype(np.int16))
print('Saved test_output.wav')
"

# Voice cloning (requires >= 5 second reference audio)
python voice_clone_jinx.py --ref_audio /path/to/jinx_clip.wav --text "Ready to cause some chaos?"
```

---

## Notes

- **No watermarking**: By not installing `resemble-perth`, generated audio won't have watermarks
- **No torchaudio**: The stub provides only what's needed for the xvector speaker embedding model
- **PyTorch preserved**: Using `--no-deps` prevents pip from overriding the container's optimized PyTorch
- The NVIDIA PyTorch container already has torch, CUDA, and cuDNN configured

---

## Troubleshooting

### "No module named 'perth'"
This is expected and harmless. The code catches this and skips watermarking.

### PyTorch was overwritten
If you accidentally ran `pip install -e .` without `--no-deps`, reinstall the container or run:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```
(But it's better to start fresh with the container)

### Model download fails
Ensure `HF_TOKEN` is set:
```bash
export HF_TOKEN=hf_your_token_here
```
