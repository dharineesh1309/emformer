 HEAD
torchaudio: an audio library for PyTorch


[![Documentation](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Ftorchaudio%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://pytorch.org/audio/main/)
[![Anaconda Badge](https://anaconda.org/pytorch/torchaudio/badges/downloads.svg)](https://anaconda.org/pytorch/torchaudio)
[![Anaconda-Server Badge](https://anaconda.org/pytorch/torchaudio/badges/platforms.svg)](https://anaconda.org/pytorch/torchaudio)

![TorchAudio Logo](docs/source/_static/img/logo.png)

> [!NOTE]
> **We have transitioned TorchAudio into a
>  maintenance phase. This process removed some user-facing
>  features. These features were deprecated from TorchAudio 2.8 and removed in 2.9.
>  Our main goals were to reduce redundancies with the rest of the
>  PyTorch ecosystem, make it easier to maintain, and create a version of
>  TorchAudio that is more tightly scoped to its strengths: processing audio
>  data for ML. Please see
>  [our community message](https://github.com/pytorch/audio/issues/3902)
>  for more details.**

The aim of torchaudio is to apply [PyTorch](https://github.com/pytorch/pytorch) to
the audio domain. By supporting PyTorch, torchaudio follows the same philosophy
of providing strong GPU acceleration, having a focus on trainable features through
the autograd system, and having consistent style (tensor names and dimension names).
Therefore, it is primarily a machine learning library and not a general signal
processing library. The benefits of PyTorch can be seen in torchaudio through
having all the computations be through PyTorch operations which makes it easy
to use and feel like a natural extension.

- [Dataloaders for common audio datasets](http://pytorch.org/audio/main/datasets.html)
- Audio and speech processing functions
  - [forced_align](https://pytorch.org/audio/main/generated/torchaudio.functional.forced_align.html)
- Common audio transforms
  - [Spectrogram, AmplitudeToDB, MelScale, MelSpectrogram, MFCC, MuLawEncoding, MuLawDecoding, Resample](http://pytorch.org/audio/main/transforms.html)
- Compliance interfaces: Run code using PyTorch that align with other libraries
  - [Kaldi: spectrogram, fbank, mfcc](https://pytorch.org/audio/main/compliance.kaldi.html)

Installation
------------

Please refer to https://pytorch.org/audio/main/installation.html for installation and build process of TorchAudio.


API Reference
-------------

API Reference is located here: http://pytorch.org/audio/main/

Contributing Guidelines
-----------------------

Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md)

Citation
--------

If you find this package useful, please cite as:

```bibtex
@article{yang2021torchaudio,
  title={TorchAudio: Building Blocks for Audio and Speech Processing},
  author={Yao-Yuan Yang and Moto Hira and Zhaoheng Ni and Anjali Chourdia and Artyom Astafurov and Caroline Chen and Ching-Feng Yeh and Christian Puhrsch and David Pollack and Dmitriy Genzel and Donny Greenberg and Edward Z. Yang and Jason Lian and Jay Mahadeokar and Jeff Hwang and Ji Chen and Peter Goldsborough and Prabhat Roy and Sean Narenthiran and Shinji Watanabe and Soumith Chintala and Vincent Quenneville-Bélair and Yangyang Shi},
  journal={arXiv preprint arXiv:2110.15018},
  year={2021}
}
```

```bibtex
@misc{hwang2023torchaudio,
      title={TorchAudio 2.1: Advancing speech recognition, self-supervised learning, and audio processing components for PyTorch},
      author={Jeff Hwang and Moto Hira and Caroline Chen and Xiaohui Zhang and Zhaoheng Ni and Guangzhi Sun and Pingchuan Ma and Ruizhe Huang and Vineel Pratap and Yuekai Zhang and Anurag Kumar and Chin-Yun Yu and Chuang Zhu and Chunxi Liu and Jacob Kahn and Mirco Ravanelli and Peng Sun and Shinji Watanabe and Yangyang Shi and Yumeng Tao and Robin Scheibler and Samuele Cornell and Sean Kim and Stavros Petridis},
      year={2023},
      eprint={2310.17864},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

Disclaimer on Datasets
----------------------

This is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the ML community!

Pre-trained Model License
-------------------------

The pre-trained models provided in this library may have their own licenses or terms and conditions derived from the dataset used for training. It is your responsibility to determine whether you have permission to use the models for your use case.

For instance, SquimSubjective model is released under the Creative Commons Attribution Non Commercial 4.0 International (CC-BY-NC 4.0) license. See [the link](https://zenodo.org/record/4660670#.ZBtWPOxuerN) for additional details.

Other pre-trained models that have different license are noted in documentation. Please checkout the [documentation page](https://pytorch.org/audio/main/).
=======
# emfomer
A custom-trained 12-layer Emformer Content Encoder for streaming speech representation learning, optimized using multi-task CTC phoneme prediction and F0 estimation to extract linguistic and prosodic embeddings.
# Emformer Content Encoder

## Overview

The Emformer is the core neural network used in the Content Encoder. It is a streaming Transformer architecture designed for speech processing that efficiently models both local and long-range temporal dependencies while supporting low-latency inference.

Unlike conventional Transformer encoders that require the complete utterance before processing, the Emformer processes speech in small segments while maintaining contextual memory from previous segments. This makes it suitable for real-time speech applications.

In this project, the Emformer is trained as a **multi-task content encoder** rather than a speech recognizer. Its objective is to learn frame-level speech representations that preserve linguistic content and prosodic information, which are later used by the accent conversion pipeline.

---

# Architecture

The input to the network is a raw speech waveform sampled at **16 kHz**.

The waveform is first converted into log Mel spectrogram features using:

| Parameter | Value |
|-----------|------:|
| Sample Rate | 16000 Hz |
| FFT Size | 800 |
| Window Length | 800 |
| Hop Length | 320 (20 ms) |
| Mel Bins | 80 |

Each 20 ms frame is represented as an 80-dimensional log Mel feature vector.

These features are projected into the Emformer hidden space:

```
80 → 1024
```

using a linear projection layer.

The projected features are then processed by a **12-layer Emformer encoder**.

---

# Emformer Configuration

The encoder follows the configuration described in the paper.

| Parameter | Value |
|-----------|------:|
| Hidden Dimension | 1024 |
| Number of Layers | 12 |
| Attention Heads | 8 |
| Feed Forward Dimension | 4096 |
| Segment Length | 4 |
| Left Context | 30 |
| Right Context | 8 |
| Memory Size | 32 |
| Dropout | 0.1 |

The encoder produces a contextual embedding for every acoustic frame.

Each output frame is represented by a **1024-dimensional embedding**.

---

# Multi-task Training

The Emformer itself does not directly predict speech.

Instead, two prediction heads are attached during training.

```
                    Emformer
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
   CTC Head                     F0 Head
        │                             │
Phoneme Prediction          Pitch Prediction
```

---

## 1. CTC Head

A linear layer maps the Emformer embeddings into the phoneme vocabulary.

```
1024 → Vocabulary Size
```

The predictions are trained using **Connectionist Temporal Classification (CTC)** loss.

The objective is to predict the correct phoneme sequence without requiring explicit frame-level alignment.

This encourages the Emformer to encode **linguistic content** rather than simply memorizing acoustic patterns.

---

## 2. F0 Head

A second linear layer predicts the **Fundamental Frequency (F0)** for every frame.

```
1024 → 1
```

Ground-truth F0 values are precomputed using the YAAPT pitch extraction algorithm and stored as cached NumPy files to avoid recomputation during training.

The predicted F0 values are supervised using an L1 regression loss.

This auxiliary task forces the encoder to preserve pitch and prosodic information.

---

# Training Objective

The network is optimized using a weighted combination of the two objectives:

```
L = 0.8 × LCTC + 0.2 × LF0
```

where

- **LCTC** is the Connectionist Temporal Classification loss.
- **LF0** is the L1 loss between predicted and target F0.

The CTC objective teaches the encoder to preserve **what was spoken**, while the F0 objective teaches it to preserve **how it was spoken**.

---

# Custom Training Procedure

Unlike pretrained Emformer ASR models, this encoder is trained **from scratch**.

The training procedure consists of the following stages:

1. Convert speech into log Mel spectrograms.
2. Project the Mel features into a 1024-dimensional feature space.
3. Process the features using a 12-layer Emformer encoder.
4. Predict phoneme sequences through the CTC head.
5. Predict frame-level F0 values through the F0 head.
6. Compute the combined multi-task loss.
7. Backpropagate gradients through the entire network.
8. Update all Emformer parameters using AdamW optimization.

No pretrained weights are used.

The Emformer learns directly from the speech corpus.

---

# Training Dataset

Training is performed using a combined dataset consisting of:

- L2-ARCTIC
- LibriSpeech

The dataset provides:

- Speech waveform
- Phoneme sequence
- Frame-level F0 targets

Only the waveform is used as input to the model.

The phoneme labels and F0 values are used solely for computing the training loss.

---

# Inference

During inference, only the speech waveform is required.

```
Waveform
     │
     ▼
Mel Spectrogram
     │
     ▼
Input Projection
     │
     ▼
12-layer Emformer
     │
     ▼
Frame-level Embeddings
```

The CTC and F0 heads are not required by downstream components.

The output embeddings are passed to the Bottleneck Extractor, which transforms accented speech representations into native-like representations for the subsequent speech synthesis stage.

---

# Role in the Accent Conversion Pipeline

The Emformer serves as the feature extraction stage of the accent conversion system.

Rather than recognizing text, it converts raw speech into contextual embeddings that retain:

- Linguistic content (phonetic information)
- Temporal context
- Prosodic characteristics (through F0 supervision)

These embeddings form the intermediate representation used by the remaining modules of the accent conversion pipline 
0c3ac5be233b17868a16151588a026d27b135645
