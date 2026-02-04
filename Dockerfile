# Chatterbox TTS with NVIDIA PyTorch base image
# No watermarking, no torchaudio dependency
# Includes ARM/GraceHopper CUDA JIT fixes
FROM nvcr.io/nvidia/pytorch:25.01-py3

# Set working directory
WORKDIR /workspace

# Persist Hugging Face cache (bind-mount at runtime)
ENV HF_HOME=/cache/hf
RUN mkdir -p /cache/hf

# Install dependencies WITHOUT touching torch/torchaudio/torchvision
# We use --no-deps for chatterbox and install deps manually to avoid torch override
# NOTE: Don't constrain numpy - base image has numpy 2.x which works fine
# NOTE: tokenizers must be installed explicitly for transformers
# NOTE: Avoid installing torch from PyPI; keep NVIDIA torch from base image.
RUN pip install --no-cache-dir \
    librosa==0.11.0 \
    conformer==0.3.2 \
    safetensors==0.5.3 \
    spacy-pkuseg \
    pykakasi==2.3.0 \
    gradio==5.44.1 \
    pyloudnorm \
    omegaconf \
    scipy \
    huggingface_hub \
    "tokenizers>=0.21,<0.22"

# Install torch-dependent packages without deps to avoid pulling CPU torch
RUN pip install --no-cache-dir --no-deps \
    s3tokenizer \
    diffusers==0.29.0 \
    accelerate

# Install transformers without dependencies to avoid torchvision conflict
RUN pip install --no-cache-dir --no-deps transformers==4.48.0

# Remove incompatible torchvision from base image (not needed for Chatterbox)
RUN pip uninstall -y torchvision

# ============================================================
# TORCHAUDIO STUB
# Provides: compliance.kaldi.fbank/mfcc, transforms.Resample
# ============================================================
RUN mkdir -p /usr/local/lib/python3.12/dist-packages/torchaudio/compliance

# torchaudio __init__.py
RUN echo 'from . import compliance, transforms' > /usr/local/lib/python3.12/dist-packages/torchaudio/__init__.py

# compliance __init__.py
RUN echo '# torchaudio.compliance stub' > /usr/local/lib/python3.12/dist-packages/torchaudio/compliance/__init__.py

# compliance/kaldi.py - minimal fbank/mfcc implementation
COPY <<'EOF' /usr/local/lib/python3.12/dist-packages/torchaudio/compliance/kaldi.py
"""
Minimal torchaudio.compliance.kaldi stub.
Provides fbank/mfcc using pure torch. Not bit-identical to kaldi but sufficient for inference.
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
EOF

# transforms.py - sinc-based Resample for high-quality audio resampling
COPY <<'EOF' /usr/local/lib/python3.12/dist-packages/torchaudio/transforms.py
"""
Improved torchaudio.transforms stub with proper sinc-based resampling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Resample(nn.Module):
    """
    Resample waveform using polyphase sinc interpolation.
    """
    
    def __init__(
        self, 
        orig_freq: int = 16000, 
        new_freq: int = 16000,
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        beta: float = 14.769656459379492,
    ):
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
            sinc = torch.where(
                t == 0,
                torch.ones_like(t),
                torch.sin(math.pi * cutoff * t) / (math.pi * t)
            ) * cutoff
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
        
        device = waveform.device
        dtype = waveform.dtype
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
        filtered = F.conv1d(upsampled, avg_kernel, padding='same')
        filtered = filtered.squeeze(1)
        
        if self.down > 1:
            out_length = int(math.ceil(length * self.up / self.down))
            start = pad_size * self.up
            indices = torch.arange(0, out_length, device=device) * self.down + start
            indices = indices.clamp(0, filtered.shape[1] - 1).long()
            result = filtered[:, indices]
        else:
            start = pad_size * self.up
            end = start + length * self.up
            result = filtered[:, start:end]
        
        result = result * self.up
        
        if squeeze:
            result = result.squeeze(0)
        
        return result


class Spectrogram(nn.Module):
    """Basic spectrogram transform (ARM-compatible, avoids .abs() on complex)."""
    def __init__(self, n_fft=400, hop_length=None, power=2.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 2
        self.power = power
        self.register_buffer('window', torch.hann_window(n_fft))
    
    def forward(self, waveform):
        stft = torch.stft(
            waveform, 
            self.n_fft, 
            self.hop_length,
            window=self.window.to(waveform.device),
            return_complex=True
        )
        spec = stft.real**2 + stft.imag**2
        if self.power != 2.0:
            spec = spec.pow(self.power / 2.0)
        return spec
EOF

# ============================================================
# CLONE AND INSTALL CHATTERBOX
# Pin to specific commit to ensure patches remain compatible
# ============================================================
RUN git clone https://github.com/resemble-ai/chatterbox.git /workspace/chatterbox && \
    cd /workspace/chatterbox && \
    git checkout 14df766196b38221c3005c79183af8a8715128

# Install chatterbox WITHOUT dependencies (we already installed them above)
# This avoids pip trying to install torch==2.6.0 and torchaudio==2.6.0
RUN pip install --no-cache-dir --no-deps -e /workspace/chatterbox

# ============================================================
# ARM/GraceHopper CUDA JIT FIX
# Patch s3tokenizer to avoid .abs() on complex tensors
# ============================================================
RUN sed -i 's/magnitudes = stft\[\.\.\.*, :-1\]\.abs()\*\*2/stft_slice = stft[..., :-1]; magnitudes = stft_slice.real**2 + stft_slice.imag**2/' \
    /workspace/chatterbox/src/chatterbox/models/s3tokenizer/s3tokenizer.py

# ============================================================
# PATCH PERTH IMPORT (make watermarking optional)
# ============================================================
RUN sed -i 's/^import perth$/try:\n    import perth\nexcept ImportError:\n    perth = None/' \
    /workspace/chatterbox/src/chatterbox/tts.py \
    /workspace/chatterbox/src/chatterbox/tts_turbo.py \
    /workspace/chatterbox/src/chatterbox/vc.py \
    /workspace/chatterbox/src/chatterbox/mtl_tts.py

# Patch watermarker usage in TTS modules to be optional
RUN python - <<'PY'
from pathlib import Path

def patch_watermarker(path: str) -> None:
    fpath = Path(path)
    text = fpath.read_text()
    old = "        self.watermarker = perth.PerthImplicitWatermarker()"
    new = (
        "        try:\n"
        "            self.watermarker = perth.PerthImplicitWatermarker()\n"
        "            if self.watermarker is None:\n"
        "                raise TypeError(\"PerthImplicitWatermarker returned None\")\n"
        "        except Exception as e:\n"
        "            print(f\"Warning: Watermarker not available: {e}. Audio will not be watermarked.\")\n"
        "            self.watermarker = type(\"NoOpWatermarker\", (), {\"apply_watermark\": lambda _self, wav, sample_rate=None: wav})()"
    )
    if old in text:
        fpath.write_text(text.replace(old, new))

for rel in [
    "tts.py",
    "tts_turbo.py",
    "vc.py",
    "mtl_tts.py",
]:
    patch_watermarker(f"/workspace/chatterbox/src/chatterbox/{rel}")
PY

# Guard apply_watermark usage when watermarker is None (vc.py, mtl_tts.py)
RUN python - <<'PY'
from pathlib import Path

def patch_apply_watermark(path: str) -> None:
    fpath = Path(path)
    text = fpath.read_text()
    old = "            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)"
    new = (
        "            if self.watermarker is not None:\n"
        "                wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)\n"
        "            watermarked_wav = wav"
    )
    if old in text:
        fpath.write_text(text.replace(old, new))

patch_apply_watermark("/workspace/chatterbox/src/chatterbox/vc.py")
patch_apply_watermark("/workspace/chatterbox/src/chatterbox/mtl_tts.py")
PY

# HuggingFace token - pass at runtime via: docker run -e HF_TOKEN=hf_xxx ...
# (removed ENV HF_TOKEN to avoid Docker security warning)

# ============================================================
# COPY CUSTOM SCRIPTS (v2 - default voice, better URLs)
# ============================================================
COPY scripts/tts_api.py /workspace/chatterbox/tts_api.py
COPY docs/API.md /workspace/chatterbox/API.md
COPY scripts/voice_clone_jinx.py /workspace/chatterbox/voice_clone_jinx.py
COPY scripts/voice_clone_original.py /workspace/chatterbox/voice_clone_original.py
COPY gradio/gradio_tts_turbo_app_enhanced.py /workspace/chatterbox/gradio_tts_turbo_app_enhanced.py
COPY gradio/gradio_tts_app_enhanced.py /workspace/chatterbox/gradio_tts_app_enhanced.py
COPY scripts/docker_start.sh /workspace/chatterbox/docker_start.sh
RUN chmod +x /workspace/chatterbox/docker_start.sh

WORKDIR /workspace/chatterbox

# Expose API + Gradio ports
EXPOSE 10050 7860 7861

# Default command
CMD ["/workspace/chatterbox/docker_start.sh"]
