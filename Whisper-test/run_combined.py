from qwen_asr import Qwen3ASRModel
from faster_whisper import WhisperModel
import soundfile as sf
import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
import time
import os

AUDIO_PATH = "snrt_test.wav"
TARGET_SR = 16000
MAX_CHUNK = 28
CONFIDENCE_THRESHOLD = -0.15

print("Loading MoulSot v0.3...")
moulsot = Qwen3ASRModel.from_pretrained(
    "atlasia/moulsot.v0.3",
    dtype="float32",
    device_map="cpu"
)
print("Loading Whisper turbo...")
turbo = WhisperModel("turbo", device="cpu", compute_type="int8")
print("Loading Silero VAD...")
vad = load_silero_vad()
print("All models loaded.\n")

print(f"Loading audio: {AUDIO_PATH}")
audio_np, sr = sf.read(AUDIO_PATH, dtype="float32")
if audio_np.ndim > 1:
    audio_np = audio_np.mean(axis=1)
audio_1d = torch.tensor(audio_np)
duration = len(audio_1d) / TARGET_SR
print(f"Audio duration: {duration:.1f}s")

print("\nRunning VAD...")
timestamps = get_speech_timestamps(
    audio_1d,
    vad,
    sampling_rate=TARGET_SR,
    max_speech_duration_s=MAX_CHUNK,
    min_silence_duration_ms=800,
    return_seconds=False,
)
print(f"VAD found {len(timestamps)} segments\n")
print("Transcribing...\n")

results = []
start = time.time()

for i, ts in enumerate(timestamps):
    chunk = audio_1d[ts["start"]:ts["end"]].numpy()
    chunk_dur = len(chunk) / TARGET_SR
    start_sec = round(ts["start"] / TARGET_SR, 1)
    end_sec = round(ts["end"] / TARGET_SR, 1)

    if chunk_dur < 0.5:
        continue

    temp_path = f"temp_chunk_{i}.wav"
    sf.write(temp_path, chunk, TARGET_SR)

    # Step 1 — always run Turbo first (fast, gives confidence)
    segs, _ = turbo.transcribe(temp_path, language="ar", beam_size=5)
    seg_list = list(segs)
    text_turbo = " ".join([s.text.strip() for s in seg_list])
    avg_logprob = (
        sum(s.avg_logprob for s in seg_list) / len(seg_list)
        if seg_list else -999
    )

    if avg_logprob >= CONFIDENCE_THRESHOLD:
        # Turbo confident → MSA → use Turbo
        chosen_text = text_turbo
        chosen_model = "Turbo"
    else:
        # Turbo not confident → Darija → use MoulSot
        result_m = moulsot.transcribe(audio=temp_path, language="Arabic")
        if isinstance(result_m, list):
            chosen_text = " ".join(
                [r.text if hasattr(r, "text") else str(r) for r in result_m]
            ).strip()
        else:
            chosen_text = str(result_m).strip()
        chosen_model = "MoulSot"

    os.remove(temp_path)

    if chosen_text:
        line = f"[{start_sec}s - {end_sec}s] [{chosen_model} | conf={avg_logprob:.2f}]  {chosen_text}"
        print(line)
        results.append(line)

elapsed = time.time() - start
print(f"\n--- DONE in {elapsed:.1f} seconds ---")

with open("output_combined.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))
print("Saved to output_combined.txt")

moulsot_count = sum(1 for r in results if "[MoulSot" in r)
turbo_count = sum(1 for r in results if "[Turbo" in r)
print(f"\nMoulSot used: {moulsot_count} chunks")
print(f"Turbo used:   {turbo_count} chunks")
