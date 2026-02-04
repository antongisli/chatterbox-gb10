# Chatterbox TTS - Agent Notes

This document captures everything needed to understand and maintain the modified Chatterbox setup.

---

## Environment

- **Base Container**: `nvcr.io/nvidia/pytorch:25.01-py3`
- **Architecture**: ARM64 (NVIDIA GraceHopper)
- **Python**: 3.12
- **PyTorch**: 2.6.0a0+ecf3bae40a.nv25.1 (NVIDIA optimized, DO NOT override)
- **Chatterbox Commit**: `14df766196b38221c3005c79183af8a8715128` (2026-02-03)
- **Chatterbox Repo**: https://github.com/resemble-ai/chatterbox.git

---

## Key Modifications

### 1. Watermark Removal

**Problem**: Chatterbox uses `resemble-perth` to embed watermarks in generated audio.

**Solution**: Simply don't install `resemble-perth`. The code gracefully handles this:
```python
# In tts.py and tts_turbo.py
try:
    self.watermarker = perth.PerthImplicitWatermarker()
except Exception as e:
    logger.warning(f"Watermarker not available: {e}. Audio will not be watermarked.")
    self.watermarker = None
```

### 2. Torchaudio Stub

**Problem**: Chatterbox imports `torchaudio` for:
- `torchaudio.compliance.kaldi.fbank` (speaker embedding extraction)
- `torchaudio.transforms.Resample` (audio resampling)

Installing full torchaudio would override the container's optimized PyTorch.

**Solution**: Create stub modules at `/usr/local/lib/python3.12/dist-packages/torchaudio/`:

| File | Purpose |
|------|---------|
| `__init__.py` | Package init, imports compliance and transforms |
| `compliance/__init__.py` | Empty package init |
| `compliance/kaldi.py` | Pure-torch `fbank()` and `mfcc()` implementations |
| `transforms.py` | Sinc-based `Resample` class with Kaiser window |

**Key implementation details**:
- `fbank()` uses `torch.fft.rfft()` and custom mel filterbank
- `Resample` uses polyphase sinc interpolation (not linear interpolation!)
- All complex tensor operations use `.real**2 + .imag**2` instead of `.abs()**2`

### 3. ARM CUDA JIT Fix

**Problem**: On ARM (GraceHopper/Jetson), calling `.abs()` on complex tensors triggers CUDA JIT compilation that fails with:
```
nvrtc: error: invalid value for --gpu-architecture (-arch)
```

**Affected file**: `chatterbox/models/s3tokenizer/s3tokenizer.py` line 161

**Original code**:
```python
magnitudes = stft[..., :-1].abs()**2
```

**Patched code**:
```python
stft_slice = stft[..., :-1]; magnitudes = stft_slice.real**2 + stft_slice.imag**2
```

**Patch command**:
```bash
sed -i 's/magnitudes = stft\[\.\.\.*, :-1\]\.abs()\*\*2/stft_slice = stft[..., :-1]; magnitudes = stft_slice.real**2 + stft_slice.imag**2/' \
    /workspace/chatterbox/src/chatterbox/models/s3tokenizer/s3tokenizer.py
```

### 4. PyTorch Preservation

**Problem**: Running `pip install -e .` installs dependencies including `torch==2.6.0`, overriding the container's NVIDIA-optimized version.

**Solution**: Install dependencies manually, then use `pip install --no-deps -e .`

---

## Model Comparison

| Feature | Chatterbox (Original) | Chatterbox-Turbo |
|---------|----------------------|------------------|
| Parameters | 500M | 350M |
| Diffusion steps | 10 | 1 (distilled) |
| Speed | Slower | Faster |
| Encoder conditioning | 6 seconds | 15 seconds |
| Decoder conditioning | 10 seconds | 10 seconds |
| CFG weight | ✅ Yes | ❌ No |
| Exaggeration | ✅ Yes | ❌ No |
| Paralinguistic tags | ❌ No | ✅ `[laugh]`, `[cough]`, etc. |

**Recommendation**: Use Turbo for ARM. Original model may produce garbled output due to additional numerical sensitivity in the 10-step diffusion.

---

## Voice Cloning Quality Tips

1. **Reference audio duration**: 10-15 seconds optimal (min 5s)
2. **Audio quality**: Clean, no background music/reverb/compression artifacts
3. **Game audio rips**: Often have compression artifacts - find cleaner sources
4. **Temperature**: 0.6-0.7 for stability, 0.9-1.0 for expressiveness
5. **Concatenation**: Joining different clips can confuse the model - prefer single long clip

---

## File Inventory

| File | Purpose |
|------|---------|
| `Dockerfile` | Complete build with all patches |
| `INSTALL.md` | Manual setup guide (6 steps) |
| `AGENTS.md` | This file - agent notes |
| `API.md` | REST API documentation |
| `tts_api.py` | FastAPI server (port 10050) |
| `voice_clone_jinx.py` | Turbo model voice cloning script |
| `voice_clone_original.py` | Original model with CFG/exaggeration |
| `gradio_tts_app.py` | Gradio UI for original model |
| `gradio_tts_turbo_app.py` | Gradio UI for Turbo model |
| `gradio_tts_turbo_app_enhanced.py` | Enhanced Gradio with multi-file upload |
| `gradio_tts_app_enhanced.py` | Enhanced Original Gradio |

---

## Common Issues

### "No module named 'perth'"
Expected and harmless. Watermarking disabled.

### "nvrtc: error: invalid value for --gpu-architecture"
ARM CUDA JIT issue. Apply the s3tokenizer patch.

### Garbled/distorted output
- Check if s3tokenizer patch is applied
- Try Turbo instead of Original
- Use longer/cleaner reference audio
- Lower temperature

### PyTorch version mismatch warnings
Expected. Container has `2.6.0a0+...nv25.1`, chatterbox wants `2.6.0`. Works fine.

---

## Rebuild Commands

```bash
# Full rebuild
cd /home/anton/src/chatterbox
docker build -t chatterbox-tts .

# Run container
docker run --gpus all -it --rm \
  -e HF_TOKEN=hf_your_token \
  -v /path/to/audio:/workspace/audio \
  chatterbox-tts bash

# Quick test
python voice_clone_jinx.py --ref_dir ./jinx --text "Hello!" --target_duration 15
```

---

## Code Locations

- **Watermark handling**: `src/chatterbox/tts.py:130-136`, `src/chatterbox/tts_turbo.py:130-136`
- **Speaker embedding (uses fbank)**: `src/chatterbox/models/s3gen/xvector.py`
- **Resampling (uses transforms.Resample)**: `src/chatterbox/models/s3gen/s3gen.py:44`
- **Complex abs issue**: `src/chatterbox/models/s3tokenizer/s3tokenizer.py:161`
- **Model conditioning lengths**: `tts.py:107-108`, `tts_turbo.py:111-112`

---

## Voice Storage Model

| Type | Storage | Persistence | Location |
|------|---------|-------------|----------|
| Default Voice | In-memory | Per-session | Loaded from `/workspace/audio` on startup |
| Gradio Saved | In-memory | Per-session | Created via "Save Voice" button |
| API Voices | Disk | Persistent | `/workspace/chatterbox/voices/` |

**Default voice naming**: Uses the mounted folder name (e.g., `-v /home/anton/jinx:/workspace/audio` → voice named "jinx")

**Voice priority in Gradio**: Uploaded files > Saved voice > Cached embeddings

---

## Session History (Feb 3, 2026)

1. Added NGC CLI to PATH
2. Analyzed watermark removal (perth not installed)
3. Created torchaudio compliance/kaldi stub
4. Created INSTALL.md
5. Created Dockerfile
6. Created voice_clone_jinx.py
7. Fixed missing S3GEN_SR import
8. Added torchaudio.transforms.Resample stub (linear interpolation)
9. Patched s3tokenizer.py for ARM CUDA JIT (.abs() issue)
10. Upgraded Resample to sinc interpolation with Kaiser window
11. Made reference duration configurable (--min_duration, --target_duration)
12. Created voice_clone_original.py for non-Turbo model
13. Updated Dockerfile and INSTALL.md with all patches
14. Created AGENTS.md (this file)

## Session History (Feb 4, 2026)

1. Fixed Docker startup script signal handling
2. Added DEBUG and API_DEBUG environment variables
3. Made Gradio public URLs display prominently after launch
4. Added default voice loading from `/workspace/audio`
5. Fixed voice naming to use directory name (e.g., "jinx" not "default")
6. Added "Active Voice" indicator in Gradio UI
7. Clarified voice priority (uploads > saved > cached)
8. Suppressed noisy FutureWarning/UserWarning in Docker output
9. Fixed `log: command not found` error in cleanup function
10. Added HTTPS note for microphone access
11. Updated INSTALL.md with Voice Management section
12. Updated AGENTS.md with voice storage model
