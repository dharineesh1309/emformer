from pathlib import Path
import json

from textgrid import TextGrid

ROOT = Path(
    "/host/home/dc/lv02-server/accent_conversion/rawdataset/l2artic_release_v5"
)

manifest = []

bad_textgrids = 0
missing_wavs = 0

for speaker_dir in ROOT.iterdir():

    if not speaker_dir.is_dir():
        continue

    if speaker_dir.name == "suitcase_corpus":
        continue

    wav_dir = speaker_dir / "wav"
    tg_dir = speaker_dir / "annotation"

    if not wav_dir.exists():
        continue

    if not tg_dir.exists():
        continue

    for tg in tg_dir.glob("*.TextGrid"):

        wav = wav_dir / f"{tg.stem}.wav"

        if not wav.exists():
            missing_wavs += 1
            continue

        # Validate TextGrid
        try:
            TextGrid.fromFile(str(tg))

        except Exception as e:

            bad_textgrids += 1

            print(
                f"BAD TEXTGRID: {tg}"
            )

            print(
                f"ERROR: {e}"
            )

            continue

        manifest.append(
            {
                "speaker": speaker_dir.name,
                "utt_id": tg.stem,
                "wav": str(wav),
                "textgrid": str(tg),
            }
        )

print()
print("Pairs:", len(manifest))
print("Bad TextGrids:", bad_textgrids)
print("Missing wavs:", missing_wavs)

Path(
    "manifests"
).mkdir(
    exist_ok=True
)

with open(
    "manifests/l2arctic_manifest.json",
    "w"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )

print("Saved manifest.")