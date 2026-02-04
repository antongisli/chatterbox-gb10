#!/usr/bin/env python3
"""
Enhanced Chatterbox (Original) Gradio App

Features:
- Multi-file reference audio upload (OGG, WAV, MP3, FLAC, M4A)
- Automatic concatenation if total duration < 5 seconds
- Shows file count and total duration
- "Keep Embeddings" checkbox to cache voice conditionals (faster repeated generation)
- Pre-compute embeddings button for even faster workflow
- CFG weight and Exaggeration controls (Original model features)
"""

import io
import json
import os
import random
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import torch
import gradio as gr
from scipy.io import wavfile

from chatterbox.tts import ChatterboxTTS, Conditionals
from chatterbox.models.s3gen import S3GEN_SR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_DURATION = 5.0
TARGET_DURATION = 10.0
DEFAULT_VOICE_LABEL = "(use uploaded audio)"
API_DEFAULT_VOICE_LABEL = "(use local voice)"
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:10050").rstrip("/")
SYNC_API_VOICES = os.getenv("SYNC_API_VOICES", "0") == "1"

CUSTOM_CSS = """
/* minimal styling */
@media (max-width: 900px) {
    .gradio-container .gr-row {
        flex-direction: column !important;
        gap: 12px !important;
    }
    .gradio-container .gr-column {
        width: 100% !important;
    }
    .gradio-container .gr-button {
        width: 100% !important;
    }
}
"""


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model():
    print(f"Loading Chatterbox (Original) on {DEVICE}...")
    model = ChatterboxTTS.from_pretrained(DEVICE)
    return model


def get_audio_duration(filepath: str) -> float:
    """Get duration of audio file in seconds."""
    try:
        y, sr = librosa.load(filepath, sr=None)
        return len(y) / sr
    except Exception as e:
        print(f"Error getting duration for {filepath}: {e}")
        return 0.0


def process_reference_files(files: list, mic_audio: str | None = None) -> tuple[str | None, str]:
    """
    Process uploaded reference files.
    Returns: (path_to_use, status_message)
    """
    if not files and not mic_audio:
        return None, "No reference files uploaded"
    
    # Get durations for all files
    file_info = []
    total_duration = 0.0
    
    def add_file(path: str, label: str | None = None) -> None:
        nonlocal total_duration
        duration = get_audio_duration(path)
        if duration > 0:
            file_info.append((path, duration, label or Path(path).name))
            total_duration += duration

    if mic_audio:
        add_file(mic_audio, "mic_recording.wav")

    for f in files or []:
        filepath = f.name if hasattr(f, "name") else f
        add_file(filepath)
    
    if not file_info:
        return None, "ERROR: No valid audio files found"
    
    # Build status message
    status_lines = [f"{len(file_info)} file(s) | {total_duration:.1f}s total"]
    status_lines.append("-" * 30)
    for filepath, duration, label in file_info:
        status_lines.append(f"  {label} ({duration:.1f}s)")
    
    if total_duration < MIN_DURATION:
        status_lines.append("-" * 30)
        status_lines.append(f"Need {MIN_DURATION - total_duration:.1f}s more (min {MIN_DURATION}s)")
        return None, "\n".join(status_lines)
    
    # If single file with enough duration, use it directly
    if len(file_info) == 1:
        status_lines.append("-" * 30)
        status_lines.append("Ready to generate!")
        return file_info[0][0], "\n".join(status_lines)
    
    # Concatenate multiple files
    status_lines.append("-" * 30)
    status_lines.append("Concatenating files...")
    
    combined = []
    current_duration = 0.0
    
    for filepath, duration, label in file_info:
        try:
            audio, sr = librosa.load(filepath, sr=S3GEN_SR)
            combined.append(audio)
            current_duration += duration
            if current_duration >= TARGET_DURATION:
                break
        except Exception as e:
            status_lines.append(f"  Skip {label}: {e}")
    
    if not combined:
        return None, "\n".join(status_lines) + "\nERROR: Failed to load audio"
    
    # Save concatenated audio
    result = np.concatenate(combined)
    temp_path = tempfile.mktemp(suffix=".wav")
    wav_int16 = (result * 32767).astype(np.int16)
    wavfile.write(temp_path, S3GEN_SR, wav_int16)
    
    status_lines.append(f"Ready! Using {current_duration:.1f}s of audio")
    return temp_path, "\n".join(status_lines)


def update_file_status(files, mic_audio):
    """Update the file status display when files change."""
    _, status = process_reference_files(files, mic_audio)
    return status


def build_voice_choices(voice_library: dict) -> list[str]:
    return [DEFAULT_VOICE_LABEL] + sorted(voice_library.keys())


def fetch_api_voices(model_type: str) -> tuple[list[str], str]:
    if not SYNC_API_VOICES:
        return [], "API voice sync disabled"
    try:
        with urllib.request.urlopen(f"{API_BASE_URL}/voices") as resp:
            data = json.load(resp)
        voices = sorted([v["name"] for v in data if v.get("model_type") == model_type])
        if not voices:
            return [], "No API voices available"
        return voices, f"Loaded {len(voices)} API voices"
    except Exception as exc:
        return [], f"Failed to fetch API voices: {exc}"


def generate_via_api(
    text: str,
    voice: str,
    temperature: float,
    min_p: float,
    top_p: float,
    repetition_penalty: float,
    exaggeration: float,
    cfg_weight: float,
):
    if not SYNC_API_VOICES:
        raise gr.Error("API voice sync disabled.")
    payload = json.dumps(
        {
            "text": text,
            "voice": voice,
            "model": "original",
            "temperature": temperature,
            "min_p": min_p,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE_URL}/tts",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            wav_bytes = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise gr.Error(f"API request failed: {exc.code} {detail}")
    sr, audio = wavfile.read(io.BytesIO(wav_bytes))
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / 32767.0
    return sr, audio


def generate(
    model,
    cached_conds,
    voice_library,
    selected_voice,
    api_voice,
    text,
    ref_files,
    mic_audio,
    keep_embeddings,
    exaggeration,
    cfg_weight,
    temperature,
    seed_num,
    min_p,
    top_p,
    repetition_penalty,
):
    if api_voice and api_voice != API_DEFAULT_VOICE_LABEL:
        audio = generate_via_api(
            text,
            api_voice,
            temperature,
            min_p,
            top_p,
            repetition_penalty,
            exaggeration,
            cfg_weight,
        )
        return audio, model, cached_conds, f"🎤 **{api_voice}** (API) ✓"

    if model is None:
        model = _default_model or ChatterboxTTS.from_pretrained(DEVICE)
    
    if seed_num != 0:
        set_seed(int(seed_num))
    
    use_saved_voice = selected_voice and selected_voice != DEFAULT_VOICE_LABEL
    use_cached = keep_embeddings and cached_conds is not None
    
    # Determine voice source (priority: uploaded files > saved voice > cached)
    audio_path = None
    voice_used = None
    
    has_uploads = ref_files or mic_audio
    if has_uploads:
        audio_path, status = process_reference_files(ref_files, mic_audio)
        if audio_path:
            voice_used = "uploaded audio"
    
    # Merge module-level default voice library with session voice library
    merged_library = {**(_default_voice_library or {}), **(voice_library or {})}
    
    if not voice_used and use_saved_voice:
        if selected_voice not in merged_library:
            raise gr.Error(f"Voice '{selected_voice}' not found. Please save a voice first.")
        model.conds = merged_library[selected_voice]
        voice_used = selected_voice
    
    if not voice_used and use_cached:
        model.conds = cached_conds
        voice_used = "cached"
    
    if not voice_used:
        raise gr.Error(
            "⚠️ No voice selected!\n\n"
            "Please do ONE of the following:\n"
            "• Upload reference audio file(s)\n"
            "• Record your voice using the microphone\n"
            "• Select a saved voice from the dropdown"
        )
    
    # Generate
    wav = model.generate(
        text,
        audio_prompt_path=audio_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
        min_p=min_p,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    
    # Cache conditionals if requested
    if voice_used not in ("uploaded audio", "cached"):
        new_conds = cached_conds
    else:
        new_conds = model.conds if keep_embeddings else None
    
    active_display = f"🎤 **{voice_used}** ✓"
    return (model.sr, wav.squeeze(0).numpy()), model, new_conds, active_display


def precompute_embeddings(model, ref_files, mic_audio, exaggeration):
    """Pre-compute embeddings from reference files."""
    if model is None:
        model = _default_model or ChatterboxTTS.from_pretrained(DEVICE)
    
    audio_path, status = process_reference_files(ref_files, mic_audio)
    if audio_path is None:
        return model, None, f"❌ Cannot pre-compute: {status}"
    
    # Prepare conditionals (this is the expensive part)
    model.prepare_conditionals(audio_path, exaggeration=exaggeration)
    
    return model, model.conds, f"✅ Embeddings cached! Future generations will be faster.\n\n{status}"


def save_voice(model, ref_files, mic_audio, exaggeration, voice_name, voice_library):
    if model is None:
        model = _default_model or ChatterboxTTS.from_pretrained(DEVICE)

    audio_path, status = process_reference_files(ref_files, mic_audio)
    if audio_path is None:
        return model, voice_library, gr.update(), f"❌ Cannot save voice: {status}"

    model.prepare_conditionals(audio_path, exaggeration=exaggeration)
    # Merge with default library to preserve default voices in choices
    merged_library = {**(_default_voice_library or {}), **(voice_library or {})}
    cleaned_name = (voice_name or "").strip()
    if not cleaned_name:
        cleaned_name = f"voice_{len(merged_library) + 1}"
    merged_library[cleaned_name] = model.conds
    # Return only the session-added voices (exclude defaults to avoid deepcopy issues)
    session_library = {k: v for k, v in merged_library.items() if k not in (_default_voice_library or {})}
    session_library[cleaned_name] = model.conds
    return (
        model,
        session_library,
        gr.update(choices=build_voice_choices(merged_library), value=cleaned_name),
        f"✅ Saved voice '{cleaned_name}' ({status.splitlines()[0]})",
    )


DEFAULT_AUDIO_DIR = os.getenv("DEFAULT_AUDIO_DIR", "/workspace/audio")
DEFAULT_VOICE_NAME = os.getenv("DEFAULT_VOICE_NAME", "")


def load_default_voice():
    """Load default voice from audio directory if available."""
    if not os.path.isdir(DEFAULT_AUDIO_DIR):
        print(f"[Voice] No audio directory at {DEFAULT_AUDIO_DIR}", flush=True)
        return None, {}, None
    
    voice_name = DEFAULT_VOICE_NAME or Path(DEFAULT_AUDIO_DIR).name
    if voice_name in ("audio", "workspace"):
        voice_name = "default"
    
    audio_extensions = {".wav", ".ogg", ".oga", ".mp3", ".flac", ".m4a"}
    audio_files = []
    for f in os.listdir(DEFAULT_AUDIO_DIR):
        if Path(f).suffix.lower() in audio_extensions:
            audio_files.append(os.path.join(DEFAULT_AUDIO_DIR, f))
    
    if not audio_files:
        print(f"[Voice] No audio files found in {DEFAULT_AUDIO_DIR}", flush=True)
        return None, {}, None
    
    audio_files.sort()
    print(f"[Voice] Loading '{voice_name}' from {len(audio_files)} files...", flush=True)
    
    try:
        model = ChatterboxTTS.from_pretrained(DEVICE)
        
        total_duration = 0.0
        combined = []
        for filepath in audio_files:
            try:
                y, sr = librosa.load(filepath, sr=S3GEN_SR)
                duration = len(y) / sr
                combined.append(y)
                total_duration += duration
                if total_duration >= 15.0:
                    break
            except Exception as e:
                print(f"[Voice]   Skip {Path(filepath).name}: {e}", flush=True)
        
        if total_duration < MIN_DURATION:
            print(f"[Voice] ✗ Not enough audio ({total_duration:.1f}s < {MIN_DURATION}s)", flush=True)
            return model, {}, None
        
        result = np.concatenate(combined)
        temp_path = tempfile.mktemp(suffix=".wav")
        wav_int16 = (result * 32767).astype(np.int16)
        wavfile.write(temp_path, S3GEN_SR, wav_int16)
        
        model.prepare_conditionals(temp_path)
        os.unlink(temp_path)
        
        voice_library = {voice_name: model.conds}
        print(f"[Voice] ✓ Loaded '{voice_name}' ({total_duration:.1f}s)", flush=True)
        return model, voice_library, voice_name
        
    except Exception as e:
        print(f"[Voice] ✗ Failed to load: {e}", flush=True)
        return None, {}, None


_default_model, _default_voice_library, _default_voice_name = load_default_voice()


with gr.Blocks(title="Chatterbox Original Enhanced", css=CUSTOM_CSS) as demo:
    gr.Markdown("# 🎭 Chatterbox Original - Enhanced")
    gr.Markdown("*Multi-file upload, OGG support, embedding caching, CFG & Exaggeration controls*")
    
    # State - avoid deepcopy of torch tensors, access module-level _default_* vars directly
    model_state = gr.State(None)
    cached_conds_state = gr.State(None)
    voice_library_state = gr.State({})  # Start empty, merge with _default_voice_library in functions
    
    with gr.Row():
        with gr.Column():
            # Text input
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit.",
                label="Text to synthesize (max chars 300)",
                max_lines=5,
            )
            
            # ===== SECTION 1: Select Voice =====
            with gr.Group():
                gr.Markdown("### 🎤 Select Voice")
                _initial_voice_choices = build_voice_choices(_default_voice_library)
                _initial_voice_value = _default_voice_name if _default_voice_name else DEFAULT_VOICE_LABEL
                voice_select = gr.Dropdown(
                    label="Voice",
                    choices=_initial_voice_choices,
                    value=_initial_voice_value,
                )
                # Active voice indicator
                _initial_active = f"✓ Using **{_default_voice_name}**" if _default_voice_name else "⚠️ No voice selected"
                active_voice_display = gr.Markdown(value=_initial_active, elem_id="active_voice")
                
                # API voices (if enabled)
                api_voice_select = gr.Dropdown(
                    label="Or use API Voice",
                    choices=[API_DEFAULT_VOICE_LABEL],
                    value=API_DEFAULT_VOICE_LABEL,
                    visible=SYNC_API_VOICES,
                )
                refresh_api_btn = gr.Button(
                    "Refresh API Voices",
                    variant="secondary",
                    size="sm",
                    visible=SYNC_API_VOICES,
                )
                api_voice_status = gr.Markdown(value="", visible=False)
            
            # ===== SECTION 2: Add New Voice =====
            with gr.Accordion("➕ Add New Voice", open=False):
                gr.Markdown("*Record or upload audio, then save with a name*")
                mic_audio = gr.Audio(
                    label="Record Voice (requires HTTPS/public link)",
                    sources=["microphone"],
                    type="filepath",
                )
                ref_files = gr.File(
                    label="Or Upload Audio Files (WAV, OGG, MP3, FLAC, M4A)",
                    file_count="multiple",
                    file_types=[".wav", ".ogg", ".oga", ".mp3", ".flac", ".m4a"],
                )
                file_status = gr.Textbox(
                    value="",
                    label="Audio Status",
                    interactive=False,
                    lines=2,
                    visible=False,
                )
                with gr.Row():
                    voice_name = gr.Textbox(
                        label="Voice Name",
                        placeholder="my_voice",
                        scale=2,
                    )
                    save_voice_btn = gr.Button("💾 Save Voice", variant="secondary", scale=1)
                save_voice_status = gr.Markdown(value="", visible=False)
            
            # Original model specific controls
            exaggeration = gr.Slider(
                0.25, 2, step=.05, 
                label="Exaggeration (0.5 = neutral, higher = more expressive)", 
                value=0.5
            )
            cfg_weight = gr.Slider(
                0.0, 1, step=.05, 
                label="CFG Weight (higher = more faithful to voice)", 
                value=0.5
            )
            
            # Hidden state for keep_embeddings (always true)
            keep_embeddings = gr.State(True)
            
            run_btn = gr.Button("Generate 🎭", variant="primary")
        
        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")
            
            with gr.Accordion("Advanced Options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=0.8)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="Min P (0.02-0.1 recommended, 0 to disable)", value=0.05)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="Top P (1.0 to disable)", value=1.00)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="Repetition Penalty", value=1.2)
    
    # Load model on startup
    demo.load(fn=load_model, inputs=[], outputs=model_state)
    
    # Update active voice display when dropdown changes
    def on_voice_select(voice):
        if voice and voice != DEFAULT_VOICE_LABEL:
            return f"✓ Using **{voice}**"
        return "⚠️ No voice selected"
    
    voice_select.change(
        fn=on_voice_select,
        inputs=[voice_select],
        outputs=[active_voice_display]
    )
    
    # Update file status when files change in Add Voice section
    def on_files_change(files, mic_audio):
        audio_path, status = process_reference_files(files, mic_audio)
        if audio_path or files or mic_audio:
            return gr.update(value=status, visible=True)
        return gr.update(visible=False)
    
    ref_files.change(
        fn=on_files_change,
        inputs=[ref_files, mic_audio],
        outputs=[file_status]
    )
    mic_audio.change(
        fn=on_files_change,
        inputs=[ref_files, mic_audio],
        outputs=[file_status]
    )

    def do_save_voice(model, ref_files, mic_audio, exaggeration, voice_name, voice_library):
        model, voice_library, dropdown_update, status = save_voice(
            model, ref_files, mic_audio, exaggeration, voice_name, voice_library
        )
        saved_name = voice_name.strip() if voice_name else f"voice_{len(voice_library)}"
        active_display = f"✓ Using **{saved_name}**"
        return model, voice_library, dropdown_update, status, gr.update(visible=True), active_display

    save_voice_btn.click(
        fn=do_save_voice,
        inputs=[model_state, ref_files, mic_audio, exaggeration, voice_name, voice_library_state],
        outputs=[model_state, voice_library_state, voice_select, save_voice_status, save_voice_status, active_voice_display],
    )

    def do_refresh_api_voices():
        voices, status = fetch_api_voices("original")
        choices = [API_DEFAULT_VOICE_LABEL] + voices
        return gr.update(choices=choices, value=API_DEFAULT_VOICE_LABEL), status, gr.update(visible=True)

    refresh_api_btn.click(
        fn=do_refresh_api_voices,
        inputs=[],
        outputs=[api_voice_select, api_voice_status, api_voice_status],
    )
    
    # Generate
    def do_generate(model, cached_conds, voice_library, selected_voice, api_voice, text, ref_files, mic_audio, keep_embeddings, exaggeration, cfg_weight, temp, seed_num, min_p, top_p, repetition_penalty):
        audio, model, new_conds, active_display = generate(
            model, cached_conds, voice_library, selected_voice, api_voice, text, ref_files, mic_audio, keep_embeddings,
            exaggeration, cfg_weight, temp, seed_num, min_p, top_p, repetition_penalty
        )
        return audio, model, new_conds, active_display
    
    run_btn.click(
        fn=do_generate,
        inputs=[
            model_state,
            cached_conds_state,
            voice_library_state,
            voice_select,
            api_voice_select,
            text,
            ref_files,
            mic_audio,
            keep_embeddings,
            exaggeration,
            cfg_weight,
            temp,
            seed_num,
            min_p,
            top_p,
            repetition_penalty,
        ],
        outputs=[audio_output, model_state, cached_conds_state, active_voice_display],
    )


def launch_app():
    share = os.getenv("GRADIO_SHARE", "0") == "1"
    host = os.getenv("GRADIO_HOST", "0.0.0.0")
    port = int(os.getenv("GRADIO_ORIGINAL_PORT", "7861"))
    
    app, local_url, share_url = demo.queue(max_size=50, default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        share=share,
    )
    
    # Print URLs prominently
    print("\n" + "="*50, flush=True)
    print("🎭 GRADIO ORIGINAL READY", flush=True)
    print("="*50, flush=True)
    print(f"   Local:  {local_url}", flush=True)
    if share_url:
        print(f"   Public: {share_url}", flush=True)
    print("="*50 + "\n", flush=True)


if __name__ == "__main__":
    launch_app()
