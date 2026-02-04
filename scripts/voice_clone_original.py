#!/usr/bin/env python3
"""Voice Cloning Script using Original Chatterbox (non-Turbo)

The original model has:
- CFG (Classifier-Free Guidance) for better prompt adherence
- Exaggeration control for expressiveness
- 10 diffusion steps (vs 1 in Turbo) - potentially higher quality
- 500M params vs 350M

Trade-off: Slower but potentially higher quality output.

Usage:
    python voice_clone_original.py --ref_audio jinx.wav --text "Hello!"
    python voice_clone_original.py --ref_dir ./jinx --text "Hello!" --cfg_weight 0.5 --exaggeration 0.5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from chatterbox.tts import ChatterboxTTS
from chatterbox.models.s3gen import S3GEN_SR


def find_audio_files(directory: Path) -> list[Path]:
    """Find all audio files in a directory."""
    extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.oga'}
    files = []
    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))
    return sorted(files)


def get_audio_duration(filepath: Path) -> float:
    """Get duration of audio file in seconds."""
    import librosa
    y, sr = librosa.load(filepath, sr=None)
    return len(y) / sr


def concatenate_audio_files(files: list[Path], output_path: Path, target_duration: float = 10.0) -> Path:
    """Concatenate multiple audio files to reach target duration."""
    import librosa
    
    print(f"\nConcatenating {len(files)} files to reach {target_duration}s...")
    
    combined = []
    current_duration = 0.0
    
    for f in files:
        try:
            audio, sr = librosa.load(f, sr=S3GEN_SR)
            duration = len(audio) / sr
            combined.append(audio)
            current_duration += duration
            print(f"  + {f.name} ({duration:.1f}s) -> total: {current_duration:.1f}s")
            
            if current_duration >= target_duration:
                break
        except Exception as e:
            print(f"  Skipping {f.name}: {e}")
    
    if not combined:
        raise ValueError("No audio files could be loaded")
    
    result = np.concatenate(combined)
    wav_int16 = (result * 32767).astype(np.int16)
    wavfile.write(str(output_path), S3GEN_SR, wav_int16)
    
    print(f"Saved concatenated audio: {output_path} ({current_duration:.1f}s)")
    return output_path


def select_best_reference(audio_files: list[Path], temp_dir: Path, min_duration: float = 5.0, target_duration: float = 10.0) -> Path | None:
    """Select best reference, concatenating if needed."""
    valid_files = []
    for f in audio_files:
        try:
            duration = get_audio_duration(f)
            valid_files.append((f, duration))
            print(f"  Found: {f.name} ({duration:.1f}s)")
        except Exception as e:
            print(f"  Error with {f.name}: {e}")
    
    # Use longest single file if >= target_duration
    long_files = [(f, d) for f, d in valid_files if d >= target_duration]
    if long_files:
        long_files.sort(key=lambda x: x[1], reverse=True)
        print(f"\nUsing single file: {long_files[0][1]:.1f}s")
        return long_files[0][0]
    
    # Concatenate multiple files
    total_duration = sum(d for _, d in valid_files)
    if total_duration < min_duration:
        print(f"Error: Total duration {total_duration:.1f}s < {min_duration}s required")
        return None
    
    temp_concat = temp_dir / "concatenated_reference.wav"
    return concatenate_audio_files([f for f, _ in valid_files], temp_concat, target_duration)


def clone_voice(
    ref_audio: Path,
    text: str,
    output_path: Path,
    temperature: float = 0.8,
    cfg_weight: float = 0.5,
    exaggeration: float = 0.5,
    repetition_penalty: float = 1.2,
    top_p: float = 1.0,
    min_p: float = 0.05,
) -> None:
    """Generate speech using voice cloning with original Chatterbox."""
    
    print(f"\nLoading Chatterbox (Original) model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTTS.from_pretrained(device=device)
    print(f"Model loaded on {device}")
    
    print(f"\nReference audio: {ref_audio}")
    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"Settings: cfg_weight={cfg_weight}, exaggeration={exaggeration}, temp={temperature}")
    
    print("\nGenerating speech...")
    wav = model.generate(
        text=text,
        audio_prompt_path=str(ref_audio),
        temperature=temperature,
        cfg_weight=cfg_weight,
        exaggeration=exaggeration,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        min_p=min_p,
    )
    
    wav_np = wav.squeeze().numpy()
    wav_int16 = (wav_np * 32767).astype(np.int16)
    wavfile.write(str(output_path), model.sr, wav_int16)
    
    print(f"\nSaved to: {output_path}")
    print(f"Duration: {len(wav_np) / model.sr:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Clone a voice using Original Chatterbox (non-Turbo) with CFG and exaggeration controls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quality Tips:
  - cfg_weight 0.3-0.7: Higher = more faithful to voice, lower = more natural
  - exaggeration 0.3-0.7: Higher = more expressive, lower = more neutral
  - temperature 0.6-0.8: Lower = more stable, higher = more varied

Examples:
  python voice_clone_original.py --ref_dir ./jinx --text "Hello!" --cfg_weight 0.5 --exaggeration 0.5
  python voice_clone_original.py --ref_audio jinx.wav --text "Boom!" --cfg_weight 0.3 --exaggeration 0.7
        """
    )
    
    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument('--ref_audio', type=Path, help='Path to reference audio file')
    ref_group.add_argument('--ref_dir', type=Path, help='Directory containing reference audio files')
    
    parser.add_argument('--text', type=str, required=True, help='Text to synthesize')
    parser.add_argument('--output', type=Path, default=Path('output_original.wav'), help='Output WAV file')
    parser.add_argument('--temp_dir', type=Path, default=Path('/tmp/voice_clone'), help='Temp directory')
    
    # Original model specific
    parser.add_argument('--cfg_weight', type=float, default=0.5, help='CFG weight 0.0-1.0 (default: 0.5)')
    parser.add_argument('--exaggeration', type=float, default=0.5, help='Exaggeration 0.0-1.0 (default: 0.5)')
    
    # Common params
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature (default: 0.8)')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p sampling (default: 1.0)')
    parser.add_argument('--min_p', type=float, default=0.05, help='Min-p sampling (default: 0.05)')
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help='Repetition penalty (default: 1.2)')
    parser.add_argument('--min_duration', type=float, default=5.0, help='Min reference duration (default: 5.0)')
    parser.add_argument('--target_duration', type=float, default=10.0, help='Target reference duration (default: 10.0)')
    
    args = parser.parse_args()
    
    if args.ref_audio:
        if not args.ref_audio.exists():
            print(f"Error: Reference audio not found: {args.ref_audio}")
            sys.exit(1)
        ref_audio = args.ref_audio
    else:
        if not args.ref_dir.exists() or not args.ref_dir.is_dir():
            print(f"Error: Reference directory not found: {args.ref_dir}")
            sys.exit(1)
        
        print(f"Scanning {args.ref_dir} for audio files...")
        audio_files = find_audio_files(args.ref_dir)
        
        if not audio_files:
            print(f"Error: No audio files found in {args.ref_dir}")
            sys.exit(1)
        
        args.temp_dir.mkdir(parents=True, exist_ok=True)
        ref_audio = select_best_reference(audio_files, args.temp_dir, args.min_duration, args.target_duration)
        if ref_audio is None:
            print(f"Error: Total audio < {args.min_duration}s")
            sys.exit(1)
    
    clone_voice(
        ref_audio=ref_audio,
        text=args.text,
        output_path=args.output,
        temperature=args.temperature,
        cfg_weight=args.cfg_weight,
        exaggeration=args.exaggeration,
        repetition_penalty=args.repetition_penalty,
        top_p=args.top_p,
        min_p=args.min_p,
    )


if __name__ == '__main__':
    main()
