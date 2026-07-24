import json
import torch
import os
import soundfile as sf
import numpy as np
from textgrid import TextGrid
from torch.utils.data import Dataset
from torchaudio.functional import resample
PHONE_FIXES = {
    "d": "D",
    "l": "L",
    "D_": "D",
    "ER)": "ER",
    "V``": "V",
    "W`": "W",
    "Y_": "Y",
}


class L2ArcticDataset(Dataset):

    def __init__(
        self,
        manifest_path,
        vocab_path,
        sample_rate=16000,
    ):

        with open(manifest_path, "r") as f:
            self.items = json.load(f)

        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)

        self.sample_rate = sample_rate
        self.f0_dir_l2 = "f0_cache"
        self.f0_dir_libri = "f0_cache_librispeech"

        speakers = sorted(
            list(
                {
                    item["speaker"]
                    for item in self.items
                }
            )
        )

        self.spk2id = {
            spk: idx
            for idx, spk in enumerate(speakers)
        }

    def __len__(self):
        return len(self.items)

    def _extract_phonemes(self, textgrid_path):

        tg = TextGrid.fromFile(textgrid_path)

        phones = []

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

                if p not in self.vocab:
                    continue

                phones.append(self.vocab[p])

        return torch.tensor(
            phones,
            dtype=torch.long
        )

    def __getitem__(self, idx):

        item = self.items[idx]

        audio, sr = sf.read(item["wav"])

        

        # Convert to tensor
        wav = torch.tensor(
            audio,
            dtype=torch.float32
        )

        # Ensure shape = [channels, time]
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.transpose(0, 1)

        

        # Resample 44.1kHz -> 16kHz
        if sr != self.sample_rate:
            wav = resample(
                wav,
                orig_freq=sr,
                new_freq=self.sample_rate
            )

      

   

        # ----------------------------------
        # L2Arctic: TextGrid phonemes
        # LibriSpeech: precomputed phonemes
        # ----------------------------------

        if "phonemes" in item:

            phonemes = torch.tensor(
                item["phonemes"],
                dtype=torch.long
            )

        else:

            # ----------------------------------
            # Phonemes
            # ----------------------------------

            if "phonemes" in item:

                phonemes = torch.tensor(
                    item["phonemes"],
                    dtype=torch.long
                )

            else:

                phonemes = self._extract_phonemes(
                    item["textgrid"]
                )

        speaker_id = self.spk2id[
            item["speaker"]
        ]
        # ----------------------------------
        # F0 Cache
        # ----------------------------------

        fname = (
            f"{item['speaker']}_{item['utt_id']}.npy"
        )

        f0_path_l2 = os.path.join(
            self.f0_dir_l2,
            fname
        )

        f0_path_libri = os.path.join(
            self.f0_dir_libri,
            fname
        )

        if os.path.exists(f0_path_l2):

            f0_path = f0_path_l2

        elif os.path.exists(f0_path_libri):

            f0_path = f0_path_libri

        else:

            raise FileNotFoundError(
                f"Missing F0 cache: {fname}"
            )

        f0 = torch.tensor(
            np.load(f0_path),
            dtype=torch.float32
        )

        
     
        return {
            "waveform": wav,
            "phonemes": phonemes,
            "f0": f0,
            "speaker_id": speaker_id,
            "utt_id": item["utt_id"],
        }