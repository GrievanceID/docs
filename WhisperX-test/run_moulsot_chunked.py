from qwen_asr import Qwen3ASRModel
import torch
import soundfile as sf
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps
import time
import os

AUDIO_PATH = "snrt_test.wav"
TARGET_SR = 16000
MAX_CHUNK = 28

print("Loading MoulSot v0.3...")
model = Qwen3ASRModel.from_pretrained(
    "atlasia/moulsot.v0.3",
    dtype="float32",
    device_map="cpu"
)
print("Loading Silero VAD...")
vad = load_silero_vad()
print("All models loaded.")

# Load audio using soundfile instead of torchaudio
print(f"\nLoading audio: {AUDIO_PATH}")
audio_np, sr = sf.read(AUDIO_PATH, dtype="float32")
if audio_np.ndim > 1:
    audio_np = audio_np.mean(axis=1)
audio_1d = torch.tensor(audio_np)
duration = len(audio_1d) / TARGET_SR
print(f"Audio duration: {duration:.1f}s")

# VAD
print("\nRunning VAD...")
timestamps = get_speech_timestamps(
    audio_1d,
    vad,
    sampling_rate=TARGET_SR,
    max_speech_duration_s=MAX_CHUNK,
    min_silence_duration_ms=800,
    return_seconds=False,
)
print(f"VAD found {len(timestamps)} speech segments")

# Transcribe each chunk
print("\nTranscribing chunks...\n")
results = []
start = time.time()

for i, ts in enumerate(timestamps):
    chunk = audio_1d[ts["start"]:ts["end"]].numpy()
    chunk_dur = len(chunk) / TARGET_SR

    if chunk_dur < 0.5:
        continue

    start_sec = round(ts["start"] / TARGET_SR, 1)
    end_sec = round(ts["end"] / TARGET_SR, 1)

    temp_path = f"temp_chunk_{i}.wav"
    sf.write(temp_path, chunk, TARGET_SR)

    result = model.transcribe(audio=temp_path, language="Arabic")

    if isinstance(result, list):
        text = " ".join([r.text if hasattr(r, "text") else str(r) for r in result])
    else:
        text = str(result)

    text = text.strip()
    if text:
        line = f"[{start_sec}s - {end_sec}s]  {text}"
        print(line)
        results.append(line)

    os.remove(temp_path)

elapsed = time.time() - start
print(f"\n--- DONE in {elapsed:.1f} seconds ---")

output = "\n".join(results)
with open("output_moulsot_chunked.txt", "w", encoding="utf-8") as f:
    f.write(output)
print("Saved to output_moulsot_chunked.txt")
