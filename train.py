import json
import torch

from datasets.l2arctic_dataset import L2ArcticDataset
from models.content_encoder import ContentEncoder

# ==========================================
# CONFIG
# ==========================================

CHECKPOINT = "checkpoints/best.pt"

MANIFEST = "manifests/combined_manifest.json"
VOCAB = "vocab/phoneme_vocab.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================
# VOCAB
# ==========================================

with open(VOCAB, "r") as f:
    vocab = json.load(f)

id2phone = {
    v: k
    for k, v in vocab.items()
}

VOCAB_SIZE = len(vocab)

# ==========================================
# MODEL
# ==========================================

model = ContentEncoder(
    vocab_size=VOCAB_SIZE
).to(DEVICE)

ckpt = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    ckpt["model_state_dict"]
)

model.eval()

print("Loaded checkpoint.")

# ==========================================
# DATASET
# ==========================================

dataset = L2ArcticDataset(
    MANIFEST,
    VOCAB
)

sample = dataset[4]

waveform = sample["waveform"].to(
    DEVICE
)

phonemes = sample["phonemes"]

utt_id = sample["utt_id"]

# IMPORTANT:
# waveform is already [1, Samples]
# DO NOT unsqueeze again

print(
    "Waveform shape:",
    waveform.shape
)

waveform_lengths = torch.tensor(
    [waveform.shape[-1]],
    device=DEVICE
)

# ==========================================
# INFERENCE
# ==========================================

with torch.no_grad():

    logits, f0_pred, out_lengths = model(
        waveform,
        waveform_lengths
    )

print(
    "\nLogits shape:",
    logits.shape
)

print(
    "F0 shape:",
    f0_pred.shape
)

print(
    "Output lengths:",
    out_lengths
)
# ==========================================
# RAW PROBABILITY INSPECTION
# ==========================================

probs = torch.softmax(
    logits,
    dim=-1
)

topk_probs, topk_ids = probs[0].topk(
    k=5,
    dim=-1
)

print("\n====================================")
print("FIRST 20 FRAMES")
print("====================================")

for t in range(min(20, probs.shape[1])):

    print(f"\nFrame {t}")

    for k in range(5):

        idx = topk_ids[t, k].item()

        phone = id2phone.get(
            idx,
            "UNK"
        )

        p = topk_probs[t, k].item()

        print(
            f"{idx:3d} "
            f"{phone:10s} "
            f"{p:.4f}"
        )

# ==========================================
# GREEDY CTC DECODE
# ==========================================

from collections import Counter

pred_ids = logits.argmax(
    dim=-1
)[0]
print(
    "\nBlank ratio:",
    (
        pred_ids == 0
    ).float().mean().item()
)

# ----------------------------------
# DEBUG PREDICTION DISTRIBUTION
# ----------------------------------

counts = Counter(
    pred_ids.cpu().tolist()
)

print("\n====================================")
print("TOP PREDICTED IDS")
print("====================================")

for idx, count in counts.most_common(20):

    token = id2phone.get(
        idx,
        "UNK"
    )

    print(
        f"{idx:3d}  "
        f"{token:10s}  "
        f"{count}"
    )

# ----------------------------------
# CTC COLLAPSE
# ----------------------------------

decoded = []

prev = -1

for p in pred_ids.cpu().tolist():

    if p == prev:
        continue

    prev = p

    # Skip blank
    if p == 0:
        continue

    decoded.append(p)

# ==========================================
# TARGET PHONEMES
# ==========================================

target = []

for x in phonemes:

    idx = x.item()

    if idx in id2phone:
        target.append(
            id2phone[idx]
        )

# ==========================================
# PREDICTED PHONEMES
# ==========================================

predicted = []

for idx in decoded:

    if idx in id2phone:
        predicted.append(
            id2phone[idx]
        )

# ==========================================
# RESULTS
# ==========================================

print("\n====================================")
print("UTT:", utt_id)
print("====================================")

print("\nTARGET:")
print(
    " ".join(target[:100])
)

print("\nPREDICTED:")
print(
    " ".join(predicted[:100])
)

print(
    "\nTarget length:",
    len(target)
)

print(
    "Predicted length:",
    len(predicted)
)
print("epoch:", ckpt.get("epoch"))
print("loss:", ckpt.get("loss"))
print("ctc_loss:", ckpt.get("ctc_loss"))
print("f0_loss:", ckpt.get("f0_loss"))
print("blank_ratio:", ckpt.get("blank_ratio", "N/A"))
#for resuming training use below code 
# import os
# import json
# import torch
# import torch.nn as nn
# import traceback
# import re

# from torch.utils.data import DataLoader

# from datasets.l2arctic_dataset import L2ArcticDataset
# from datasets.collate import collate_fn
# from models.content_encoder import ContentEncoder

# # ==========================================
# # CONFIG
# # ==========================================
# MANIFEST = "manifests/combined_manifest.json"
# VOCAB = "vocab/phoneme_vocab.json"
# BATCH_SIZE = 2
# EPOCHS = 30
# LR = 5e-5
# MAX_PHONEMES = 225   # ← set after checking distribution

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# os.makedirs("checkpoints", exist_ok=True)

# # ==========================================
# # VOCAB
# # ==========================================
# with open(VOCAB, "r") as f:
#     vocab = json.load(f)

# VOCAB_SIZE = len(vocab)
# print(f"Vocabulary size: {VOCAB_SIZE}")

# # ==========================================
# # TEXTGRID PARSER
# # ==========================================
# SILENCE_TOKENS = {"sil", "sp", ""}

# def parse_textgrid_phonemes(textgrid_path, vocab):
#     """
#     Parse the phones tier from a Praat TextGrid file.
#     - Takes first token if comma-separated (e.g. "T, S, s" -> "T")
#     - Skips silence tokens: sil, sp, empty string
#     - Skips tokens not in vocab
#     Returns list of vocab indices.
#     """
#     with open(textgrid_path, "r") as f:
#         content = f.read()

#     # find the phones tier
#     phones_match = re.search(
#         r'name = "phones".*?intervals: size = \d+(.*?)(?=item \[\d+\]|$)',
#         content,
#         re.DOTALL
#     )

#     if phones_match is None:
#         return None

#     tier_text = phones_match.group(1)

#     # extract all text = "..." values
#     texts = re.findall(r'text = "(.*?)"', tier_text)

#     phonemes = []
#     for text in texts:
#         # take first token if comma-separated
#         token = text.split(",")[0].strip()

#         # skip silence
#         if token in SILENCE_TOKENS:
#             continue

#         # skip unknown tokens
#         if token not in vocab:
#             continue

#         phonemes.append(vocab[token])

#     return phonemes if len(phonemes) > 0 else None


# # ==========================================
# # BUILD UNIFIED MANIFEST
# # ==========================================
# with open(MANIFEST, "r") as f:
#     manifest = json.load(f)

# print(f"Manifest size before processing: {len(manifest)}")

# unified = []
# skipped_no_phonemes = 0
# skipped_textgrid_fail = 0
# skipped_too_long = 0

# for entry in manifest:
#     if "phonemes" in entry:
#         phonemes = entry["phonemes"]

#     elif "textgrid" in entry:
#         phonemes = parse_textgrid_phonemes(
#             entry["textgrid"],
#             vocab
#         )
#         if phonemes is None:
#             skipped_textgrid_fail += 1
#             continue

#     else:
#         skipped_no_phonemes += 1
#         continue

#     if len(phonemes) > MAX_PHONEMES:
#         skipped_too_long += 1
#         continue

#     unified.append({
#         "speaker":  entry["speaker"],
#         "utt_id":   entry["utt_id"],
#         "wav":      entry["wav"],
#         "phonemes": phonemes,
#     })

# print(f"Unified manifest size:      {len(unified)}")
# print(f"Skipped (no phoneme key):   {skipped_no_phonemes}")
# print(f"Skipped (textgrid failed):  {skipped_textgrid_fail}")
# print(f"Skipped (too long):         {skipped_too_long}")

# UNIFIED_MANIFEST = "manifests/combined_manifest_unified.json"
# with open(UNIFIED_MANIFEST, "w") as f:
#     json.dump(unified, f)

# print(f"Saved unified manifest to {UNIFIED_MANIFEST}")

# # ==========================================
# # DATASET
# # ==========================================
# dataset = L2ArcticDataset(
#     UNIFIED_MANIFEST,
#     VOCAB
# )

# print(f"Dataset size: {len(dataset)}")

# loader = DataLoader(
#     dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     collate_fn=collate_fn,
#     num_workers=4,
#     pin_memory=True,
# )

# # ==========================================
# # MODEL
# # ==========================================
# model = ContentEncoder(
#     vocab_size=VOCAB_SIZE
# ).to(DEVICE)

# total_params = sum(
#     p.numel()
#     for p in model.parameters()
# )

# print(f"Parameters: {total_params:,}")

# # ==========================================
# # LOSS
# # ==========================================
# ctc_loss = nn.CTCLoss(
#     blank=0,
#     zero_infinity=True,
# )

# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=LR,
# )

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode="min",
#     factor=0.5,
#     patience=2
# )

# # ==========================================
# # RESUME TRAINING
# # ==========================================
# RESUME = True
# CHECKPOINT = "checkpoints/content_encoder_epoch_30.pt"
# BEST_CHECKPOINT = "checkpoints/best.pt"

# best_loss = float("inf")
# start_epoch = 0

# if RESUME and os.path.exists(CHECKPOINT):

#     print(f"Loading checkpoint: {CHECKPOINT}")

#     ckpt = torch.load(
#         CHECKPOINT,
#         map_location=DEVICE
#     )

#     model.load_state_dict(
#         ckpt["model_state_dict"]
#     )

#     optimizer.load_state_dict(
#         ckpt["optimizer_state_dict"]
#     )

#     if "scheduler_state_dict" in ckpt:
#         scheduler.load_state_dict(
#             ckpt["scheduler_state_dict"]
#         )
#         print("Scheduler state restored")
#     else:
#         print("No scheduler state found, starting fresh schedule")

#     start_epoch = ckpt["epoch"] + 1

#     if os.path.exists(BEST_CHECKPOINT):
#         best_ckpt = torch.load(
#             BEST_CHECKPOINT,
#             map_location=DEVICE
#         )
#         best_loss = best_ckpt["loss"]
#         print(f"Loaded true best loss from {BEST_CHECKPOINT}")
#     else:
#         best_loss = ckpt["loss"]
#         print("No best.pt found, using resumed checkpoint loss as best")

#     print(f"Resuming from epoch {start_epoch}")
#     print(f"Best loss so far: {best_loss:.4f}")

# else:
#     print("No checkpoint found, starting from scratch")

# # ==========================================
# # TRAIN
# # ==========================================
# for epoch in range(start_epoch, start_epoch + EPOCHS):
#     model.train()
#     running_loss = 0.0
#     running_ctc_loss = 0.0
#     running_f0_loss = 0.0

#     for step, batch in enumerate(loader):
#         waveforms = batch["waveforms"].to(
#             DEVICE,
#             non_blocking=True
#         )

#         waveform_lengths = batch["waveform_lengths"].to(
#             DEVICE,
#             non_blocking=True
#         )

#         phonemes = batch["phonemes"].to(
#             DEVICE,
#             non_blocking=True
#         )

#         phoneme_lengths = batch["phoneme_lengths"].to(
#             DEVICE,
#             non_blocking=True
#         )

#         f0_targets = batch["f0"].to(
#             DEVICE,
#             non_blocking=True
#         )

#         optimizer.zero_grad(set_to_none=True)

#         try:
#             logits, f0_pred, out_lengths = model(
#                 waveforms,
#                 waveform_lengths
#             )

#             if step == 0:
#                 print("logits:", logits.shape)
#                 print("f0_pred:", f0_pred.shape)
#                 print("f0_targets:", f0_targets.shape)
#                 print("out_lengths:", out_lengths)

#             # ----------------------------------
#             # CTC LOSS
#             # ----------------------------------
#             log_probs = logits.log_softmax(dim=-1)

#             ctc_loss_value = ctc_loss(
#                 log_probs.transpose(0, 1),
#                 phonemes,
#                 out_lengths,
#                 phoneme_lengths
#             )

#             with torch.no_grad():
#                 pred_ids = logits.argmax(dim=-1)
#                 blank_ratio = (
#                     (pred_ids == 0)
#                     .float()
#                     .mean()
#                     .item()
#                 )

#             if step % 100 == 0:
#                 print(f"BlankRatio={blank_ratio:.4f}")

#             # ----------------------------------
#             # ALIGN F0
#             # ----------------------------------
#             T = min(
#                 f0_pred.size(1),
#                 f0_targets.size(1)
#             )
#             f0_pred = f0_pred[:, :T]
#             f0_targets = f0_targets[:, :T]

#             # ----------------------------------
#             # F0 LOSS
#             # ----------------------------------
#             mask = f0_targets > 0
#             if mask.sum() > 0:
#                 f0_loss_value = torch.nn.functional.l1_loss(
#                     f0_pred[mask],
#                     f0_targets[mask]
#                 )
#             else:
#                 f0_loss_value = torch.tensor(0.0, device=DEVICE)

#             # ----------------------------------
#             # PAPER LOSS
#             # L = 0.8 * CTC + 0.2 * F0
#             # ----------------------------------
#             loss = (
#                 0.8 * ctc_loss_value
#                 + 0.2 * f0_loss_value
#             )

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#             optimizer.step()

#             running_loss += loss.item()
#             running_ctc_loss += ctc_loss_value.item()
#             running_f0_loss += f0_loss_value.item()

#             if step % 50 == 0:
#                 print(
#                     f"Epoch {epoch+1} | "
#                     f"Step {step}/{len(loader)} | "
#                     f"Loss {loss.item():.4f} | "
#                     f"CTC {ctc_loss_value.item():.4f} | "
#                     f"F0 {f0_loss_value.item():.4f}"
#                 )

#             if step % 100 == 0:
#                 print(f"GPU: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

#             del logits
#             del f0_pred
#             del log_probs
#             del loss

#         except Exception:
#             print(f"\nTraining failed at step {step}")
#             if "utt_ids" in batch:
#                 print("UTT:", batch["utt_ids"][0])
#             traceback.print_exc()
#             raise

#     num_steps = step + 1
#     avg_loss = running_loss / num_steps
#     avg_ctc_loss = running_ctc_loss / num_steps
#     avg_f0_loss = running_f0_loss / num_steps

#     scheduler.step(avg_ctc_loss)

#     print(f"\nEpoch {epoch+1}")
#     print(f"Average Total Loss: {avg_loss:.4f}")
#     print(f"Average CTC Loss: {avg_ctc_loss:.4f}")
#     print(f"Average F0 Loss: {avg_f0_loss:.4f}")

#     checkpoint = {
#         "epoch": epoch,
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "scheduler_state_dict": scheduler.state_dict(),
#         "loss": avg_loss,
#         "ctc_loss": avg_ctc_loss,
#         "f0_loss": avg_f0_loss,
#     }

#     torch.save(
#         checkpoint,
#         f"checkpoints/content_encoder_epoch_{epoch+1}.pt"
#     )

#     if avg_loss < best_loss:
#         best_loss = avg_loss
#         torch.save(checkpoint, "checkpoints/best.pt")
#         print(f"New best model saved (loss={avg_loss:.4f})")

# print("Training finished.")