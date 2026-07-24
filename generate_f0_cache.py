import os
import json
import numpy as np
import soundfile as sf
import torch

from torchaudio.functional import resample

from amfm_decompy.basic_tools import SignalObj
from amfm_decompy.pYAAPT import yaapt


# ==========================================
# CONFIG
# ==========================================

MANIFEST = "manifests/l2arctic_manifest.json"

CACHE_DIR = "f0_cache"

TARGET_SR = 16000

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)


# ==========================================
# LOAD MANIFEST
# ==========================================

with open(MANIFEST, "r") as f:
    manifest = json.load(f)

print(
    f"Found {len(manifest)} utterances"
)


# ==========================================
# PROCESS
# ==========================================

for idx, item in enumerate(manifest):

    speaker = item["speaker"]
    utt_id = item["utt_id"]

    save_path = os.path.join(
        CACHE_DIR,
        f"{speaker}_{utt_id}.npy"
    )

    if os.path.exists(save_path):
        continue

    try:

        audio, sr = sf.read(
            item["wav"]
        )

        wav = torch.tensor(
            audio,
            dtype=torch.float32
        )

        if wav.ndim > 1:
            wav = wav.mean(
                dim=1
            )

        if sr != TARGET_SR:

            wav = resample(
                wav.unsqueeze(0),
                sr,
                TARGET_SR
            ).squeeze(0)

        wav_np = wav.numpy()

        signal = SignalObj(
            wav_np,
            TARGET_SR
        )

        pitch = yaapt(
            signal,
            frame_space=20.0
        )

        f0 = pitch.samp_values

        f0 = np.asarray(
            f0,
            dtype=np.float32
        )

        # ----------------------------------
        # log-F0
        # ----------------------------------

        voiced = f0 > 0

        log_f0 = np.zeros_like(
            f0,
            dtype=np.float32
        )

        log_f0[voiced] = np.log(
            f0[voiced]
        )

        np.save(
            save_path,
            log_f0
        )

        if idx % 100 == 0:

            print(
                f"[{idx}/{len(manifest)}] "
                f"{speaker}/{utt_id} "
                f"frames={len(log_f0)}"
            )

    except Exception as e:

        print(
            f"FAILED: "
            f"{speaker}/{utt_id}"
        )

        print(e)

print("\nDone.")
