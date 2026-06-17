import whisperx
import soundfile as sf
import numpy as np
import time

AUDIO_PATH = "snrt_test.wav"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

print("Loading WhisperX turbo...")
model = whisperx.load_model("turbo", DEVICE, compute_type=COMPUTE_TYPE, language="ar")

# Load audio manually with soundfile instead of whisperx.load_audio
print("Loading audio...")
audio_np, sr = sf.read(AUDIO_PATH, dtype="float32")
if audio_np.ndim > 1:
    audio_np = audio_np.mean(axis=1)

print("Transcribing...")
start = time.time()

result = model.transcribe(audio_np, batch_size=4, language="ar")

elapsed = time.time() - start
print(f"\n--- TRANSCRIPTION ---\n")

for seg in result["segments"]:
    line = f"[{seg['start']:.1f}s - {seg['end']:.1f}s]  {seg['text'].strip()}"
    print(line)

print(f"\n--- DONE in {elapsed:.1f} seconds ---")

with open("output_whisperx.txt", "w", encoding="utf-8") as f:
    for seg in result["segments"]:
        f.write(f"[{seg['start']:.1f}s - {seg['end']:.1f}s]  {seg['text'].strip()}\n")

print("Saved to output_whisperx.txt")
