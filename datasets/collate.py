import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):

    # ----------------------------------
    # Waveforms
    # ----------------------------------

    waveforms = [
        item["waveform"].squeeze(0)
        for item in batch
    ]

    waveform_lengths = torch.tensor(
        [len(w) for w in waveforms],
        dtype=torch.long
    )

    waveforms = pad_sequence(
        waveforms,
        batch_first=True
    )

    # ----------------------------------
    # Phonemes
    # ----------------------------------

    phonemes = [
        item["phonemes"]
        for item in batch
    ]

    phoneme_lengths = torch.tensor(
        [len(p) for p in phonemes],
        dtype=torch.long
    )

    phonemes = pad_sequence(
        phonemes,
        batch_first=True,
        padding_value=0
    )

    # ----------------------------------
    # F0
    # ----------------------------------

    f0s = [
        item["f0"]
        for item in batch
    ]

    f0_lengths = torch.tensor(
        [len(f0) for f0 in f0s],
        dtype=torch.long
    )

    f0s = pad_sequence(
        f0s,
        batch_first=True,
        padding_value=0.0
    )

    # ----------------------------------
    # Metadata
    # ----------------------------------

    speaker_ids = torch.tensor(
        [item["speaker_id"] for item in batch],
        dtype=torch.long
    )

    utt_ids = [
        item["utt_id"]
        for item in batch
    ]

    # ----------------------------------
    # Return
    # ----------------------------------

    return {
        "waveforms": waveforms,
        "waveform_lengths": waveform_lengths,
        "phonemes": phonemes,
        "phoneme_lengths": phoneme_lengths,
        "f0": f0s,
        "f0_lengths": f0_lengths,
        "speaker_ids": speaker_ids,
        "utt_ids": utt_ids,
    }