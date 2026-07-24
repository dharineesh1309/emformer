import json
import random
import editdistance
import torch

from datasets.l2arctic_dataset import L2ArcticDataset
from models.content_encoder import ContentEncoder

# =====================================================
# CONFIG
# =====================================================

CHECKPOINT = "checkpoints/best.pt"
MANIFEST = "manifests/combined_manifest.json"
VOCAB = "vocab/phoneme_vocab.json"

NUM_TEST = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# VOCAB
# =====================================================

with open(VOCAB) as f:
    vocab = json.load(f)

id2phone = {v: k for k, v in vocab.items()}

# =====================================================
# MODEL
# =====================================================

model = ContentEncoder(
    vocab_size=len(vocab)
).to(DEVICE)

ckpt = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("Checkpoint loaded.\n")

# =====================================================
# DATASET
# =====================================================

dataset = L2ArcticDataset(
    MANIFEST,
    VOCAB
)

NUM_TEST = min(NUM_TEST, len(dataset))
indices = random.sample(range(len(dataset)), NUM_TEST)

# =====================================================
# METRICS
# =====================================================

total_edit = 0
total_target = 0
total_accuracy = 0
total_blank_ratio = 0

best_acc = -1
worst_acc = 101

best_utt = ""
worst_utt = ""

# =====================================================
# EVALUATION
# =====================================================

for i, idx in enumerate(indices):

    sample = dataset[idx]

    waveform = sample["waveform"].to(DEVICE)
    phonemes = sample["phonemes"]
    utt_id = sample["utt_id"]

    waveform_lengths = torch.tensor(
        [waveform.shape[-1]],
        device=DEVICE
    )

    with torch.no_grad():

        logits, _, _ = model(
            waveform,
            waveform_lengths
        )

    pred_ids = logits.argmax(dim=-1)[0]

    blank_ratio = (
        (pred_ids == 0)
        .float()
        .mean()
        .item()
    )

    total_blank_ratio += blank_ratio

    # =================================================
    # GREEDY CTC DECODE
    # =================================================

    decoded = []

    prev = -1

    for p in pred_ids.cpu().tolist():

        if p == prev:
            continue

        prev = p

        if p == 0:
            continue

        decoded.append(p)

    target_ids = phonemes.cpu().tolist()

    # =================================================
    # EDIT DISTANCE
    # =================================================

    edit = editdistance.eval(
        target_ids,
        decoded
    )

    total_edit += edit
    total_target += len(target_ids)

    accuracy = max(
        0,
        1 - edit / max(1, len(target_ids))
    ) * 100

    total_accuracy += accuracy

    if accuracy > best_acc:
        best_acc = accuracy
        best_utt = utt_id

    if accuracy < worst_acc:
        worst_acc = accuracy
        worst_utt = utt_id

    # =================================================
    # PRINT
    # =================================================

    target = [
        id2phone[x]
        for x in target_ids
        if x in id2phone
    ]

    predicted = [
        id2phone[x]
        for x in decoded
        if x in id2phone
    ]

    print("=" * 90)
    print(f"[{i+1:3d}/{NUM_TEST}] {utt_id}")
    print("=" * 90)

    print(f"Target Length    : {len(target)}")
    print(f"Pred Length      : {len(predicted)}")
    print(f"Edit Distance    : {edit}")
    print(f"Accuracy         : {accuracy:.2f}%")
    print(f"Blank Ratio      : {blank_ratio:.3f}")

    print("\nTARGET:")
    print(" ".join(target))

    print("\nPREDICTED:")
    print(" ".join(predicted))

    print()

# =====================================================
# FINAL REPORT
# =====================================================

overall_per = total_edit / total_target

overall_accuracy = (
    total_accuracy / NUM_TEST
)

avg_blank = (
    total_blank_ratio / NUM_TEST
)

print("\n")
print("=" * 100)
print("FINAL EVALUATION")
print("=" * 100)

print(f"Utterances Tested      : {NUM_TEST}")
print(f"Total Target Phones    : {total_target}")
print(f"Total Edit Distance    : {total_edit}")

print(f"\nOverall PER            : {overall_per:.4f}")
print(f"Overall Accuracy       : {(1-overall_per)*100:.2f}%")

print(f"\nAverage Utterance Acc  : {overall_accuracy:.2f}%")
print(f"Average Blank Ratio    : {avg_blank:.4f}")

print(f"\nBest Utterance         : {best_utt}")
print(f"Best Accuracy          : {best_acc:.2f}%")

print(f"\nWorst Utterance        : {worst_utt}")
print(f"Worst Accuracy         : {worst_acc:.2f}%")

print("=" * 100)