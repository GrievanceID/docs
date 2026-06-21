import os
import json
import time
from google import genai
from google.genai import errors as genai_errors

API_KEY = "YOUR API KEY"

client = genai.Client(api_key=API_KEY)

CLIPS_DIR = r"PATH OF  YOUR CLIPS FOLDER"
OUTPUT_FILE = "transcripts_draft.json"
PROGRESS_FILE = "transcripts_draft_progress.json"

REQUEST_DELAY = 0.5
MAX_RETRIES = 5

results = []
done_files = set()
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
    done_files = {r["file"] for r in results}
    print(f"Resuming — {len(done_files)} clips already done\n")

wav_files = sorted([f for f in os.listdir(CLIPS_DIR) if f.endswith(".wav")])
remaining = [f for f in wav_files if f not in done_files]
print(f"Total clips: {len(wav_files)} | Remaining: {len(remaining)}\n")

errors = []

for i, fname in enumerate(remaining):
    path = os.path.join(CLIPS_DIR, fname)

    for attempt in range(MAX_RETRIES):
        try:
            audio_file = client.files.upload(file=path)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    audio_file,
                    "Transcribe this Moroccan Darija audio exactly as spoken, in Arabic script. Output only the transcription text, nothing else, no quotes, no explanation, no translation."
                ]
            )
            text = response.text.strip()
            results.append({"file": fname, "text": text, "verified": False})
            print(f"[{i+1}/{len(remaining)}] {fname}: {text[:60]}")
            break

        except genai_errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                wait = 5 * (attempt + 1)
                print(f"  Rate limited on {fname}, waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue
            else:
                print(f"  ERROR on {fname}: {e}")
                errors.append(fname)
                break

        except genai_errors.ServerError as e:
            wait = 8 * (attempt + 1)
            print(f"  Server overloaded on {fname} (503), waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
            time.sleep(wait)
            continue

        except Exception as e:
            print(f"  UNEXPECTED ERROR on {fname}: {e}")
            errors.append(fname)
            break
    else:
        print(f"  FAILED after {MAX_RETRIES} retries: {fname}")
        errors.append(fname)

    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    time.sleep(REQUEST_DELAY)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n--- DONE ---")
print(f"Transcribed: {len(results)}")
print(f"Errors: {len(errors)}")
if errors:
    print("Failed files:", errors)
print(f"Saved to {OUTPUT_FILE}")
