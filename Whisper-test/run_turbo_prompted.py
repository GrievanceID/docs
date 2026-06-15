import torch
from transformers import pipeline, WhisperProcessor
import time

print("Loading Whisper large-v3-turbo...")
model_id = "openai/whisper-large-v3-turbo"

processor = WhisperProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    device="cpu",
    dtype=torch.float32
)

# Tokenize the prompt properly
code_switch_prompt = "بسم الله، المتهم نتا متهم من أجل جنحة غسيل الأموال، الحسابات ديالك فاتت خمسة د المليار، أنا بريء مادرت والو، كلشي ديال السي فؤاد."
prompt_ids = processor.get_prompt_ids(code_switch_prompt, return_tensors="pt")

print("Transcribing full audio with code-switched prompt...\n")
start = time.time()

result = pipe(
    "snrt_test.wav",
    generate_kwargs={
        "prompt_ids": prompt_ids,
        "language": "ar",
        "task": "transcribe",
    },
    return_timestamps=True,
)

elapsed = time.time() - start
print("--- TRANSCRIPTION ---\n")

for chunk in result["chunks"]:
    ts = chunk["timestamp"]
    text = chunk["text"].strip()
    if text:
        line = f"[{ts[0]:.1f}s - {ts[1]:.1f}s]  {text}"
        print(line)

print(f"\n--- DONE in {elapsed:.1f} seconds ---")

with open("output_turbo_prompted.txt", "w", encoding="utf-8") as f:
    for chunk in result["chunks"]:
        ts = chunk["timestamp"]
        text = chunk["text"].strip()
        if text:
            f.write(f"[{ts[0]:.1f}s - {ts[1]:.1f}s]  {text}\n")

print("Saved to output_turbo_prompted.txt")
