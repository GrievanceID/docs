from faster_whisper import WhisperModel
import time

print("Loading Whisper turbo...")
model = WhisperModel("turbo", device="cpu", compute_type="int8")

print("Model loaded. Starting transcription...")
start = time.time()

segments, info = model.transcribe(
    "snrt_test.wav",
    language="ar",
    beam_size=5,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500)
)

print(f"Detected language: {info.language} (confidence: {info.language_probability:.0%})")
print("\n--- TRANSCRIPTION ---\n")

output_lines = []
for segment in segments:
    line = f"[{segment.start:.1f}s - {segment.end:.1f}s]  {segment.text.strip()}"
    print(line)
    output_lines.append(line)

elapsed = time.time() - start
print(f"\n--- DONE in {elapsed:.1f} seconds ---")

with open("output_turbo.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))
print("Saved to output_turbo.txt")
