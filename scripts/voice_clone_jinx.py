#!/usr/bin/env python3
"""Voice Cloning Script for Jinx (Arcane)

This script uses Chatterbox TTS Turbo to clone a voice from reference audio files.
Place your Jinx/Arcane voice samples in a directory and point this script to them.

Requirements:
- Reference audio should be at least 5 seconds (minimum), 10-15 seconds recommended
- Clean audio without background music/noise works best
- WAV format preferred (MP3/other formats also work via librosa)

Quality Tips:
- Longer reference = better quality (up to 15s for encoder, 10s for decoder)
- Use clean, noise-free audio (no background music, reverb, or compression artifacts)
- Consistent speaking style in reference helps
- Game audio rips often have compression artifacts - try to find cleaner sources
- Lower temperature (0.6-0.7) = more stable but less expressive
- Higher temperature (0.9-1.0) = more expressive but may introduce artifacts

Usage:
    python voice_clone_jinx.py --ref_audio jinx_samples/jinx_clip.wav --text "Your text here"
    python voice_clone_jinx.py --ref_dir jinx_samples/ --text "Your text here"  # concatenates files
    python voice_clone_jinx.py --ref_dir jinx_samples/ --text "Hello" --min_duration 10  # use 10s
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.models.s3gen import S3GEN_SR


def find_audio_files(directory: Path) -> list[Path]:
    """Find all audio files in a directory."""
    extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.oga'}  # .oga is OGG audio
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
    import numpy as np
    
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
    
    # Concatenate
    result = np.concatenate(combined)
    
    # Save
    wav_int16 = (result * 32767).astype(np.int16)
    wavfile.write(str(output_path), S3GEN_SR, wav_int16)
    
    print(f"Saved concatenated audio: {output_path} ({current_duration:.1f}s)")
    return output_path

def select_best_reference_with_concat(
    audio_files: list[Path],
    temp_dir: Path,
    min_duration: float = 5.0,
    target_duration: float = 10.0
) -> Path | None:
    """Select best reference, concatenating if needed."""
    
    # Check individual files first
    valid_files = []
    for f in audio_files:
        try:
            duration = get_audio_duration(f)
            valid_files.append((f, duration))
            print(f"  Found: {f.name} ({duration:.1f}s)")
        except Exception as e:
            print(f"  Error with {f.name}: {e}")
    
    # Use longest single file if >= min_duration
    long_files = [(f, d) for f, d in valid_files if d >= min_duration]
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
    top_k: int = 1000,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
) -> None:
    """Generate speech using voice cloning."""
    
    print(f"\nLoading Chatterbox Turbo model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    print(f"Model loaded on {device}")
    
    print(f"\nReference audio: {ref_audio}")
    print(f"Text to synthesize: {text[:100]}{'...' if len(text) > 100 else ''}")
    
    # Generate audio
    print("\nGenerating speech...")
    wav = model.generate(
        text=text,
        audio_prompt_path=str(ref_audio),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    
    # Save output
    wav_np = wav.squeeze().numpy()
    wav_int16 = (wav_np * 32767).astype(np.int16)
    wavfile.write(str(output_path), model.sr, wav_int16)
    
    print(f"\nSaved to: {output_path}")
    print(f"Sample rate: {model.sr} Hz")
    print(f"Duration: {len(wav_np) / model.sr:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Clone a voice (e.g., Jinx from Arcane) using Chatterbox TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single reference file
  python voice_clone_jinx.py --ref_audio jinx_samples/jinx_monologue.wav \\
      --text "Hey there! Ready to cause some chaos?"

  # Directory of samples (picks best one)
  python voice_clone_jinx.py --ref_dir jinx_samples/ \\
      --text "Pow pow says hello!"

  # Custom output and parameters
  python voice_clone_jinx.py --ref_audio jinx.wav --text "Boom!" \\
      --output jinx_boom.wav --temperature 0.9
        """
    )
    
    # Reference audio source (mutually exclusive)
    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument(
        '--ref_audio', type=Path,
        help='Path to reference audio file (must be >= 5 seconds)'
    )
    ref_group.add_argument(
        '--ref_dir', type=Path,
        help='Directory containing reference audio files'
    )
    
    # Required
    parser.add_argument(
        '--text', type=str, required=True,
        help='Text to synthesize with the cloned voice'
    )
    
    # Optional
    parser.add_argument(
        '--output', type=Path, default=Path('output_jinx.wav'),
        help='Output WAV file path (default: output_jinx.wav)'
    )
    parser.add_argument(
        '--temp_dir', type=Path, default=Path('/tmp/voice_clone'),
        help='Temp directory for concatenated audio (default: /tmp/voice_clone)'
    )
    parser.add_argument(
        '--temperature', type=float, default=0.8,
        help='Sampling temperature (default: 0.8)'
    )
    parser.add_argument(
        '--top_k', type=int, default=1000,
        help='Top-k sampling (default: 1000)'
    )
    parser.add_argument(
        '--top_p', type=float, default=0.95,
        help='Top-p (nucleus) sampling (default: 0.95)'
    )
    parser.add_argument(
        '--repetition_penalty', type=float, default=1.2,
        help='Repetition penalty (default: 1.2)'
    )
    parser.add_argument(
        '--min_duration', type=float, default=5.0,
        help='Minimum reference audio duration in seconds (default: 5.0)'
    )
    parser.add_argument(
        '--target_duration', type=float, default=10.0,
        help='Target reference audio duration when concatenating (default: 10.0, max useful: 15.0)'
    )
    
    args = parser.parse_args()
    
    # Resolve reference audio
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
        ref_audio = select_best_reference_with_concat(
            audio_files, 
            args.temp_dir,
            min_duration=args.min_duration,
            target_duration=args.target_duration
        )
        if ref_audio is None:
            print(f"Error: Total audio duration < {args.min_duration}s. Turbo model requires >= 5s reference.")
            sys.exit(1)
    
    # Run voice cloning
    clone_voice(
        ref_audio=ref_audio,
        text=args.text,
        output_path=args.output,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )


if __name__ == '__main__':
    main()
