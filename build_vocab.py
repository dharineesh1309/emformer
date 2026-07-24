from pathlib import Path
from textgrid import TextGrid
import json

ROOT = Path(
    "/host/home/dc/lv02-server/accent_conversion/rawdataset/l2artic_release_v5"
)

PHONE_FIXES = {
    "d": "D",
    "l": "L",
    "D_": "D",
    "ER)": "ER",
    "V``": "V",
    "W`": "W",
    "Y_": "Y",
}

phones = set()

for tg_file in ROOT.rglob("annotation/*.TextGrid"):

    try:
        tg = TextGrid.fromFile(str(tg_file))

        for tier in tg.tiers:

            if tier.name.lower() != "phones":
                continue

            for interval in tier:

                p = interval.mark.strip()

                if not p:
                    continue

                if "," in p:
                    p = p.split(",")[0].strip()

                p = PHONE_FIXES.get(p, p)

                phones.add(p)

    except Exception:
        pass

phones = sorted(phones)

vocab = {
    "<blank>": 0
}

for idx, phone in enumerate(phones, start=1):
    vocab[phone] = idx

Path("vocab").mkdir(exist_ok=True)

with open("vocab/phoneme_vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)

print("Vocabulary size:", len(vocab))
print("Saved vocab/phoneme_vocab.json")
