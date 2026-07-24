import sys
import torch
import torch.nn as nn
import torchaudio

sys.path.insert(
    0,
    "/host/home/dc/lv02-server/accent_conversion/content_encoder/src"
)

from torchaudio.models import Emformer


class ContentEncoder(nn.Module):

    def __init__(self, vocab_size=89):

        super().__init__()

        # ----------------------------------
        # Mel Frontend (Paper)
        # ----------------------------------

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=800,
            win_length=800,
            hop_length=320,
            n_mels=80,
        )

        self.log_offset = 1e-5

        # ----------------------------------
        # Input Projection
        # ----------------------------------

        self.input_proj = nn.Linear(
            80,
            1024,
        )

        # ----------------------------------
        # Emformer (Paper)
        # ----------------------------------

        self.emformer = Emformer(
            input_dim=1024,
            num_heads=8,
            ffn_dim=4096,
            num_layers=12,
            segment_length=4,
            left_context_length=30,
            right_context_length=8,
            max_memory_size=32,
            dropout=0.1,
            tanh_on_mem=True,
        )

        # ----------------------------------
        # CTC Head
        # ----------------------------------

        self.ctc_head = nn.Linear(
            1024,
            vocab_size,
        )

        # ----------------------------------
        # F0 Head (Paper)
        # ----------------------------------

        self.f0_head = nn.Linear(
            1024,
            1,
        )

    def _wave_to_mel(
        self,
        waveforms,
    ):

        feats = self.mel(
            waveforms
        )

        feats = torch.log(
            feats + self.log_offset
        )

        feats = feats.transpose(
            1,
            2,
        )

        return feats

    def forward(
        self,
        waveforms,
        waveform_lengths,
    ):

        # ----------------------------------
        # Mel Features
        # ----------------------------------

        x = self._wave_to_mel(
            waveforms
        )

        # ----------------------------------
        # Frame Lengths
        # ----------------------------------

        feat_lengths = (
            waveform_lengths // 320
        )

        feat_lengths = torch.clamp(
            feat_lengths,
            max=x.shape[1]
        )

        # ----------------------------------
        # Input Projection
        # ----------------------------------

        x = self.input_proj(
            x
        )

        # ----------------------------------
        # Emformer
        # ----------------------------------

        x, out_lengths = self.emformer(
            x,
            feat_lengths,
        )

        out_lengths = torch.clamp(
            out_lengths,
            max=x.shape[1]
        )

        # ----------------------------------
        # CTC Branch
        # ----------------------------------

        logits = self.ctc_head(
            x
        )

        # ----------------------------------
        # F0 Branch
        # ----------------------------------

        f0_pred = self.f0_head(
            x
        ).squeeze(-1)

        return (
            logits,
            f0_pred,
            out_lengths,
        )


if __name__ == "__main__":

    model = ContentEncoder(
        vocab_size=89
    )

    wav = torch.randn(
        2,
        160000
    )

    lengths = torch.tensor(
        [160000, 120000]
    )

    logits, f0_pred, out_lengths = model(
        wav,
        lengths,
    )

    print(
        "logits:",
        logits.shape
    )

    print(
        "f0_pred:",
        f0_pred.shape
    )

    print(
        "out_lengths:",
        out_lengths
    )