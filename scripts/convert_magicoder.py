"""One-off: Convert Magicoder-OSS-Instruct JSONL to Pangu format."""
import json, random, sys

INPUT = "/Volumes/MOVESPEED/datasets/open-source-sft/ise-uiuc__Magicoder-OSS-Instruct-75K/data-oss_instruct-decontaminated.jsonl"
OUTPUT = "tests/fixtures/e2e_folder_test/code/magicoder_oss_instruct.json"
N = 500
SEED = 42

records = []
with open(INPUT) as f:
    for line in f:
        records.append(json.loads(line))

print(f"Total records: {len(records)}")
random.seed(SEED)
sampled = random.sample(records, min(N, len(records)))

out = []
for i, r in enumerate(sampled):
    out.append({
        "id": f"magicoder-{i}",
        "source": "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "data": [
            {"role": "user", "content": r["problem"]},
            {"role": "assistant", "content": r["solution"]},
        ],
    })

with open(OUTPUT, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# Stats
langs = {}
for r in sampled:
    lang = r.get("lang", "unknown")
    langs[lang] = langs.get(lang, 0) + 1
top = sorted(langs.items(), key=lambda x: -x[1])[:10]
print(f"Sampled: {len(out)} → {OUTPUT}")
print(f"Top languages: {top}")
