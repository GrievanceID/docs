import whisperx
from qwen_asr import Qwen3ASRModel
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

print("Loading WhisperX turbo...")
wx_model = whisperx.load_model("turbo", "cpu", compute_type="int8", language="ar")

print("Loading MoulSot v0.3...")
moulsot = Qwen3ASRModel.from_pretrained(
    "atlasia/moulsot.v0.3",
    dtype="float32",
    device_map="cpu"
)

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

    # Step 1 — WhisperX for confidence score
    wx_result = wx_model.transcribe(chunk, batch_size=4, language="ar")
    seg_list = wx_result.get("segments", [])
    text_wx = " ".join([s["text"].strip() for s in seg_list])
    avg_logprob = (
        sum(s.get("avg_logprob", -999) for s in seg_list) / len(seg_list)
        if seg_list else -999
    )

    if avg_logprob >= CONFIDENCE_THRESHOLD:
        chosen_text = text_wx
        chosen_model = "WhisperX"
    else:
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

with open("output_combined_whisperx.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))
print("Saved to output_combined_whisperx.txt")

wx_count = sum(1 for r in results if "[WhisperX" in r)
m_count = sum(1 for r in results if "[MoulSot" in r)
print(f"\nWhisperX used: {wx_count} chunks")
print(f"MoulSot used:  {m_count} chunks")
