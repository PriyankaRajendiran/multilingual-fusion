import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# STEP 42: 80/20 PROJECTION-ALIGNED BASELINE EVALUATION
# ==============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")

CSV_PATH = os.path.join(
    DATASET_DIR,
    "multilingual_captions_clean.csv"
)

CHECKPOINT_PATH = os.path.join(
    EMBEDDINGS_DIR,
    "stage28_80_20_alignment_model.pt"
)

IMAGE_EMB_PATH = os.path.join(
    EMBEDDINGS_DIR,
    "image_embeddings.npy"
)

ENGLISH_EMB_PATH = os.path.join(
    EMBEDDINGS_DIR,
    "english_text_embeddings.npy"
)

TAMIL_EMB_PATH = os.path.join(
    EMBEDDINGS_DIR,
    "tamil_text_embeddings.npy"
)

HINDI_EMB_PATH = os.path.join(
    EMBEDDINGS_DIR,
    "hindi_text_embeddings.npy"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "stage42_fair_baseline_results.csv"
)


# ==============================================================
# LANGUAGE PROJECTION
# ==============================================================

class LanguageProjection(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )


# ==============================================================
# MULTILINGUAL ALIGNMENT MODEL
# ==============================================================

class MultilingualAlignmentModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.language_weights = nn.Parameter(
            torch.ones(3)
        )

        self.english_projection = LanguageProjection()

        self.tamil_projection = LanguageProjection()

        self.hindi_projection = LanguageProjection()


# ==============================================================
# RETRIEVAL METRICS
# ==============================================================

def recall_at_k(similarity_matrix, k):

    correct = 0

    num_samples = similarity_matrix.shape[0]

    for i in range(num_samples):

        ranked_indices = np.argsort(
            -similarity_matrix[i]
        )

        if i in ranked_indices[:k]:
            correct += 1

    return 100.0 * correct / num_samples


# ==============================================================
# EVALUATE I2T
# ==============================================================

def evaluate_i2t(image_embeddings, text_embeddings):

    similarity = (
        image_embeddings
        @ text_embeddings.T
    )

    r1 = recall_at_k(similarity, 1)
    r5 = recall_at_k(similarity, 5)
    r10 = recall_at_k(similarity, 10)

    return r1, r5, r10


# ==============================================================
# EVALUATE T2I
# ==============================================================

def evaluate_t2i(text_embeddings, image_embeddings):

    similarity = (
        text_embeddings
        @ image_embeddings.T
    )

    r1 = recall_at_k(similarity, 1)
    r5 = recall_at_k(similarity, 5)
    r10 = recall_at_k(similarity, 10)

    return r1, r5, r10


# ==============================================================
# START
# ==============================================================

print("=" * 70)
print("STEP 42: 80/20 PROJECTION-ALIGNED BASELINE EVALUATION")
print("=" * 70)


# ==============================================================
# LOAD CHECKPOINT
# ==============================================================

print("\nLoading Stage-28 alignment checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

model = MultilingualAlignmentModel()

model.load_state_dict(
    checkpoint,
    strict=True
)

model.eval()

print(
    "Checkpoint parameters loaded:",
    len(model.state_dict())
)

print("[OK] Expected: 13 parameters")


# ==============================================================
# CHECKPOINT VALIDATION
# ==============================================================

if len(model.state_dict()) != 13:

    raise RuntimeError(
        "Checkpoint loading failed: expected 13 parameters."
    )

print("[OK] All 13 checkpoint parameters loaded correctly.")


# ==============================================================
# LOAD DATASET
# ==============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    CSV_PATH
)

print(
    "\nDataset shape:",
    df.shape
)


# ==============================================================
# LOAD EMBEDDINGS
# ==============================================================

image_embeddings = np.load(
    IMAGE_EMB_PATH
)

english_embeddings = np.load(
    ENGLISH_EMB_PATH
)

tamil_embeddings = np.load(
    TAMIL_EMB_PATH
)

hindi_embeddings = np.load(
    HINDI_EMB_PATH
)

print(
    "Image embeddings :",
    image_embeddings.shape
)

print(
    "English embeddings:",
    english_embeddings.shape
)

print(
    "Tamil embeddings  :",
    tamil_embeddings.shape
)

print(
    "Hindi embeddings  :",
    hindi_embeddings.shape
)


# ==============================================================
# VALIDATION
# ==============================================================

assert image_embeddings.shape[0] == 100
assert english_embeddings.shape[0] == 100
assert tamil_embeddings.shape[0] == 100
assert hindi_embeddings.shape[0] == 100

assert image_embeddings.shape[1] == 512
assert english_embeddings.shape[1] == 768
assert tamil_embeddings.shape[1] == 768
assert hindi_embeddings.shape[1] == 768


# ==============================================================
# 80/20 SPLIT
# ==============================================================

train_indices = np.arange(
    0,
    80
)

test_indices = np.arange(
    80,
    100
)

print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    "Training samples :",
    len(train_indices)
)

print(
    "Testing samples  :",
    len(test_indices)
)

print(
    "Training indices : 0 - 79"
)

print(
    "Testing indices  : 80 - 99"
)


# ==============================================================
# CONVERT TO TORCH
# ==============================================================

english_tensor = torch.tensor(
    english_embeddings,
    dtype=torch.float32
)

tamil_tensor = torch.tensor(
    tamil_embeddings,
    dtype=torch.float32
)

hindi_tensor = torch.tensor(
    hindi_embeddings,
    dtype=torch.float32
)

image_tensor = torch.tensor(
    image_embeddings,
    dtype=torch.float32
)


# ==============================================================
# PROJECT TEXT EMBEDDINGS
# ==============================================================

print("\n" + "=" * 70)
print("PROJECTING TEXT EMBEDDINGS")
print("=" * 70)

with torch.no_grad():

    english_projected = model.english_projection(
        english_tensor
    )

    tamil_projected = model.tamil_projection(
        tamil_tensor
    )

    hindi_projected = model.hindi_projection(
        hindi_tensor
    )


# ==============================================================
# NORMALIZE EMBEDDINGS
# ==============================================================

english_projected = F.normalize(
    english_projected,
    p=2,
    dim=1
)

tamil_projected = F.normalize(
    tamil_projected,
    p=2,
    dim=1
)

hindi_projected = F.normalize(
    hindi_projected,
    p=2,
    dim=1
)

image_tensor = F.normalize(
    image_tensor,
    p=2,
    dim=1
)


# ==============================================================
# CONVERT TO NUMPY
# ==============================================================

english_projected = (
    english_projected.numpy()
)

tamil_projected = (
    tamil_projected.numpy()
)

hindi_projected = (
    hindi_projected.numpy()
)

image_embeddings_normalized = (
    image_tensor.numpy()
)


# ==============================================================
# TEST-SET ONLY
# ==============================================================

test_images = (
    image_embeddings_normalized[
        test_indices
    ]
)

test_english = (
    english_projected[
        test_indices
    ]
)

test_tamil = (
    tamil_projected[
        test_indices
    ]
)

test_hindi = (
    hindi_projected[
        test_indices
    ]
)


# ==============================================================
# LANGUAGE WEIGHTS
# ==============================================================

with torch.no_grad():

    language_weights = torch.softmax(
        model.language_weights,
        dim=0
    ).numpy()

print("\nLearned language weights:")

print(
    "English:",
    round(float(language_weights[0]), 4)
)

print(
    "Tamil  :",
    round(float(language_weights[1]), 4)
)

print(
    "Hindi  :",
    round(float(language_weights[2]), 4)
)


# ==============================================================
# RESULT STORAGE
# ==============================================================

results = []


# ==============================================================
# ENGLISH XLM-R
# ==============================================================

print("\n" + "=" * 70)
print("English XLM-R")
print("=" * 70)

i2t_r1, i2t_r5, i2t_r10 = evaluate_i2t(
    test_images,
    test_english
)

t2i_r1, t2i_r5, t2i_r10 = evaluate_t2i(
    test_english,
    test_images
)

print(
    f"I2T R@1  : {i2t_r1:.2f}%"
)

print(
    f"I2T R@5  : {i2t_r5:.2f}%"
)

print(
    f"I2T R@10 : {i2t_r10:.2f}%"
)

print(
    f"T2I R@1  : {t2i_r1:.2f}%"
)

print(
    f"T2I R@5  : {t2i_r5:.2f}%"
)

print(
    f"T2I R@10 : {t2i_r10:.2f}%"
)

results.append([
    "English XLM-R",
    80,
    20,
    "80-99",
    i2t_r1,
    i2t_r5,
    i2t_r10,
    t2i_r1,
    t2i_r5,
    t2i_r10
])


# ==============================================================
# TAMIL XLM-R
# ==============================================================

print("\n" + "=" * 70)
print("Tamil XLM-R")
print("=" * 70)

i2t_r1, i2t_r5, i2t_r10 = evaluate_i2t(
    test_images,
    test_tamil
)

t2i_r1, t2i_r5, t2i_r10 = evaluate_t2i(
    test_tamil,
    test_images
)

print(
    f"I2T R@1  : {i2t_r1:.2f}%"
)

print(
    f"I2T R@5  : {i2t_r5:.2f}%"
)

print(
    f"I2T R@10 : {i2t_r10:.2f}%"
)

print(
    f"T2I R@1  : {t2i_r1:.2f}%"
)

print(
    f"T2I R@5  : {t2i_r5:.2f}%"
)

print(
    f"T2I R@10 : {t2i_r10:.2f}%"
)

results.append([
    "Tamil XLM-R",
    80,
    20,
    "80-99",
    i2t_r1,
    i2t_r5,
    i2t_r10,
    t2i_r1,
    t2i_r5,
    t2i_r10
])


# ==============================================================
# HINDI XLM-R
# ==============================================================

print("\n" + "=" * 70)
print("Hindi XLM-R")
print("=" * 70)

i2t_r1, i2t_r5, i2t_r10 = evaluate_i2t(
    test_images,
    test_hindi
)

t2i_r1, t2i_r5, t2i_r10 = evaluate_t2i(
    test_hindi,
    test_images
)

print(
    f"I2T R@1  : {i2t_r1:.2f}%"
)

print(
    f"I2T R@5  : {i2t_r5:.2f}%"
)

print(
    f"I2T R@10 : {i2t_r10:.2f}%"
)

print(
    f"T2I R@1  : {t2i_r1:.2f}%"
)

print(
    f"T2I R@5  : {t2i_r5:.2f}%"
)

print(
    f"T2I R@10 : {t2i_r10:.2f}%"
)

results.append([
    "Hindi XLM-R",
    80,
    20,
    "80-99",
    i2t_r1,
    i2t_r5,
    i2t_r10,
    t2i_r1,
    t2i_r5,
    t2i_r10
])


# ==============================================================
# MULTILINGUAL FUSION
# ==============================================================

print("\n" + "=" * 70)
print("Multilingual Fusion")
print("=" * 70)

# Weighted multilingual representation

multilingual_test = (
    language_weights[0] * test_english
    + language_weights[1] * test_tamil
    + language_weights[2] * test_hindi
)

multilingual_test = (
    multilingual_test
    / np.linalg.norm(
        multilingual_test,
        axis=1,
        keepdims=True
    )
)

i2t_r1, i2t_r5, i2t_r10 = evaluate_i2t(
    test_images,
    multilingual_test
)

t2i_r1, t2i_r5, t2i_r10 = evaluate_t2i(
    multilingual_test,
    test_images
)

print(
    f"I2T R@1  : {i2t_r1:.2f}%"
)

print(
    f"I2T R@5  : {i2t_r5:.2f}%"
)

print(
    f"I2T R@10 : {i2t_r10:.2f}%"
)

print(
    f"T2I R@1  : {t2i_r1:.2f}%"
)

print(
    f"T2I R@5  : {t2i_r5:.2f}%"
)

print(
    f"T2I R@10 : {t2i_r10:.2f}%"
)

results.append([
    "Multilingual Fusion",
    80,
    20,
    "80-99",
    i2t_r1,
    i2t_r5,
    i2t_r10,
    t2i_r1,
    t2i_r5,
    t2i_r10
])


# ==============================================================
# SAVE RESULTS
# ==============================================================

columns = [
    "Method",
    "Training_Samples",
    "Testing_Samples",
    "Test_Index_Range",
    "I2T_R1",
    "I2T_R5",
    "I2T_R10",
    "T2I_R1",
    "T2I_R5",
    "T2I_R10"
]

results_df = pd.DataFrame(
    results,
    columns=columns
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==============================================================
# FINAL OUTPUT
# ==============================================================

print("\n" + "=" * 70)
print("STEP 42 FINAL RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nResults saved:")

print(
    OUTPUT_PATH
)

print("\n" + "=" * 70)
print("STEP 42 COMPLETED")
print("=" * 70)