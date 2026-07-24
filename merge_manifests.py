import json
import os

L2 = "manifests/l2arctic_manifest.json"
LIBRI = "manifests/librispeech_manifest.json"

OUT = "manifests/combined_manifest.json"

with open(L2) as f:
    l2 = json.load(f)

with open(LIBRI) as f:
    libri = json.load(f)

combined = libri + l2

os.makedirs(
    "manifests",
    exist_ok=True
)

with open(
    OUT,
    "w"
) as f:

    json.dump(
        combined,
        f
    )

print(
    f"LibriSpeech: {len(libri)}"
)

print(
    f"L2Arctic: {len(l2)}"
)

print(
    f"Combined: {len(combined)}"
)
