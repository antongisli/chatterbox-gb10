import numpy as np
from scipy.io import wavfile
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Load the Turbo model
model = ChatterboxTurboTTS.from_pretrained(device="cuda")

# Generate with Paralinguistic Tags
text = "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. It's the SkyNet T-800 series and it's got basically everything. Including AI integration with ChatGPT and all that jazz. Would you like me to get some prices for you?"

# Generate audio (requires a reference clip for voice cloning)
# wav = model.generate(text, audio_prompt_path="your_10s_ref_clip.wav")
wav = model.generate(text)

# Save using scipy instead of torchaudio
wav_np = wav.squeeze().numpy()
wav_int16 = (wav_np * 32767).astype(np.int16)
wavfile.write("test-turbo.wav", model.sr, wav_int16)
