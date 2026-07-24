python - <<'PY'
import json

with open("manifests/librispeech_manifest.json") as f:
    m = json.load(f)

print("Entries:", len(m))

phones = [len(x["phonemes"]) for x in m]

print("Min phones:", min(phones))
print("Max phones:", max(phones))
print("Avg phones:", sum(phones)/len(phones))
PY