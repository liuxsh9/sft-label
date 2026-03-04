"""One-off: Convert commitpackft to Pangu format.
Samples across multiple languages, framing as code-modification tasks.
"""
import json, random, os, glob

BASE = "/Volumes/MOVESPEED/datasets/open-source-sft/bigcode__commitpackft/data"
OUTPUT = "tests/fixtures/e2e_folder_test/code/commitpackft_multilang.json"
N = 500
SEED = 42

# Pick popular programming languages (not markup/config)
TARGET_LANGS = [
    "python", "javascript", "typescript", "go", "rust", "java", "c++",
    "ruby", "c#", "swift", "kotlin", "scala", "php", "c", "haskell",
    "elixir", "lua", "r", "julia", "dart",
]

# Load all samples from target languages
all_records = []
lang_counts = {}
for lang in TARGET_LANGS:
    lang_dir = os.path.join(BASE, lang if lang != "c++" else "cpp" if lang != "c#" else lang)
    # Handle language directory name mapping
    candidates = [lang, lang.replace("+", "plus"), lang.replace("#", "sharp")]
    if lang == "c++":
        candidates = ["cpp", "c++"]
    elif lang == "c#":
        candidates = ["csharp", "c#"]

    found = False
    for cand in candidates:
        path = os.path.join(BASE, cand, "data.jsonl")
        if os.path.exists(path):
            count = 0
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    r["_lang_key"] = lang
                    all_records.append(r)
                    count += 1
            lang_counts[lang] = count
            found = True
            break
    if not found:
        print(f"  [skip] {lang} — dir not found")

print(f"Loaded {len(all_records)} records from {len(lang_counts)} languages")
for lang, cnt in sorted(lang_counts.items(), key=lambda x: -x[1])[:15]:
    print(f"  {lang}: {cnt}")

# Sample N records, trying to get language diversity
random.seed(SEED)
# Stratified sample: at least a few from each language, rest random
sampled = []
per_lang = max(3, N // len(lang_counts))  # minimum per language
for lang in lang_counts:
    lang_recs = [r for r in all_records if r["_lang_key"] == lang]
    n_take = min(per_lang, len(lang_recs))
    sampled.extend(random.sample(lang_recs, n_take))

# Fill remaining with random from all
remaining = N - len(sampled)
if remaining > 0:
    pool = [r for r in all_records if r not in sampled]
    sampled.extend(random.sample(pool, min(remaining, len(pool))))

random.shuffle(sampled)
sampled = sampled[:N]

# Convert to Pangu format
out = []
for i, r in enumerate(sampled):
    # Frame as a code modification task
    subject = r.get("subject", "").strip()
    old = r.get("old_contents", "").strip()
    new = r.get("new_contents", "").strip()
    fname = r.get("new_file", r.get("old_file", "file"))
    lang_label = r.get("lang", r.get("_lang_key", "unknown"))

    # User prompt: show the current code and the modification instruction
    if old:
        user_content = (
            f"Given the following {lang_label} code in `{fname}`:\n\n"
            f"```\n{old}\n```\n\n"
            f"Task: {subject}"
        )
    else:
        # New file creation
        user_content = (
            f"Create a new {lang_label} file `{fname}`.\n\n"
            f"Task: {subject}"
        )

    # Assistant: provide the modified code
    assistant_content = f"```\n{new}\n```"

    out.append({
        "id": f"commitpack-{i}",
        "source": "bigcode/commitpackft",
        "data": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    })

with open(OUTPUT, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# Final stats
final_langs = {}
for r in sampled:
    l = r.get("_lang_key", "unknown")
    final_langs[l] = final_langs.get(l, 0) + 1
top = sorted(final_langs.items(), key=lambda x: -x[1])
print(f"\nOutput: {len(out)} samples → {OUTPUT}")
print(f"Language distribution: {top}")
