from qwen_asr import Qwen3ASRModel
import time

print("Loading MoulSot v0.3...")

model = Qwen3ASRModel.from_pretrained(
    "atlasia/moulsot.v0.3",
    dtype="float32",
    device_map="cpu"
)

print("Model loaded. Transcribing...")
start = time.time()

result = model.transcribe(
    audio="snrt_test.wav",
    language="Arabic"
)

elapsed = time.time() - start
print(f"\n--- TRANSCRIPTION ---\n")

# result is a list — print it properly
if isinstance(result, list):
    full_text = " ".join([r.text if hasattr(r, 'text') else str(r) for r in result])
else:
    full_text = str(result)

print(full_text)
print(f"\n--- DONE in {elapsed:.1f} seconds ---")

with open("output_moulsot_nochunk.txt", "w", encoding="utf-8") as f:
    f.write(full_text)
print("Saved to output_moulsot_nochunk.txt")
