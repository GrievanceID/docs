from faster_whisper import WhisperModel
import time

print("Loading Whisper large-v3...")
print("(First run downloads the model ~3GB, be patient)")

# Load model - int8 makes it faster and lighter on CPU
model = WhisperModel(
    "large-v3",
    device="cpu",
    compute_type="int8"
)

print("Model loaded. Starting transcription...")
start = time.time()

segments, info = model.transcribe(
    "snrt_test.wav",
    language="ar",          # Arabic - forces Arabic, don't let it guess
    beam_size=5,            # accuracy vs speed tradeoff - 5 is standard
    vad_filter=True,        # skip silence automatically
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

# Save to file
with open("output_largev3.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("Saved to output_largev3.txt")
