import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel


# ============================================================
# STEP 41: TEST ALL 100 IMAGES × 3 LANGUAGES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(
    BASE_DIR, "all_100_test_queries.csv"
)

IMAGE_DIR = os.path.join(
    BASE_DIR, "dataset", "images"
)

TAMIL_EMB_PATH = os.path.join(
    BASE_DIR, "embeddings", "tamil_text_embeddings.npy"
)

HINDI_EMB_PATH = os.path.join(
    BASE_DIR, "embeddings", "hindi_text_embeddings.npy"
)

IMAGE_EMB_PATH = os.path.join(
    BASE_DIR, "embeddings", "image_embeddings.npy"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "stage41_all_100_retrieval_results.csv"
)


print("=" * 70)
print("STEP 41: ALL 100 × 3 RETRIEVAL TEST")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

print()
print("Total test cases :", len(df))

if len(df) != 100:
    raise ValueError("Expected exactly 100 test cases.")


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

image_embeddings = np.load(IMAGE_EMB_PATH)

tamil_embeddings = np.load(TAMIL_EMB_PATH)
hindi_embeddings = np.load(HINDI_EMB_PATH)

print("Image embeddings :", image_embeddings.shape)
print("Tamil embeddings :", tamil_embeddings.shape)
print("Hindi embeddings :", hindi_embeddings.shape)


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

image_embeddings = torch.tensor(
    image_embeddings,
    dtype=torch.float32
)

tamil_embeddings = torch.tensor(
    tamil_embeddings,
    dtype=torch.float32
)

hindi_embeddings = torch.tensor(
    hindi_embeddings,
    dtype=torch.float32
)

image_embeddings = F.normalize(
    image_embeddings,
    dim=1
)

tamil_embeddings = F.normalize(
    tamil_embeddings,
    dim=1
)

hindi_embeddings = F.normalize(
    hindi_embeddings,
    dim=1
)


# ============================================================
# LOAD XLM-R
# ============================================================

print()
print("Loading XLM-R...")

tokenizer = AutoTokenizer.from_pretrained(
    "xlm-roberta-base"
)

xlmr = AutoModel.from_pretrained(
    "xlm-roberta-base"
)

xlmr.eval()

print("XLM-R loaded.")


# ============================================================
# XLM-R ENCODING
# ============================================================

def encode_xlmr(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = xlmr(**inputs)

        hidden = outputs.last_hidden_state

        mask = inputs["attention_mask"].unsqueeze(-1)

        masked_hidden = hidden * mask

        embedding = (
            masked_hidden.sum(dim=1)
            / mask.sum(dim=1)
        )

        embedding = F.normalize(
            embedding,
            dim=1
        )

    return embedding


# ============================================================
# LOAD CLIP
# ============================================================

print()
print("Loading CLIP...")

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

clip_model.eval()

print("CLIP loaded.")


# ============================================================
# ENGLISH CAPTION CLIP EMBEDDINGS
# ============================================================

print()
print("Encoding English captions with CLIP...")

english_captions = df["english_caption"].tolist()

english_features = []

with torch.no_grad():

    for caption in english_captions:

        inputs = clip_processor(
            text=[caption],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        feature = clip_model.get_text_features(
            **inputs
        )

        if hasattr(feature, "pooler_output"):
            feature = feature.pooler_output

        if hasattr(feature, "last_hidden_state"):
            feature = feature.last_hidden_state[:, 0]

        feature = feature.float()

        feature = F.normalize(
            feature,
            dim=1
        )

        english_features.append(
            feature.squeeze(0).cpu()
        )

english_features = torch.stack(
    english_features
)

print(
    "English CLIP embeddings:",
    english_features.shape
)


# ============================================================
# IMAGE → TEXT
# ============================================================

print()
print("=" * 70)
print("IMAGE → TEXT TEST")
print("=" * 70)

i2t_results = []

for index, row in df.iterrows():

    filename = str(row["filename"])

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    if not os.path.exists(image_path):

        print(
            f"[SKIP] Image not found: {filename}"
        )

        continue

    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = clip_processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        image_feature = clip_model.get_image_features(
            **inputs
        )

        if hasattr(image_feature, "pooler_output"):
            image_feature = image_feature.pooler_output

        if hasattr(image_feature, "last_hidden_state"):
            image_feature = image_feature.last_hidden_state[:, 0]

        image_feature = image_feature.float()

        image_feature = F.normalize(
            image_feature,
            dim=1
        )

    scores = image_feature @ english_features.T

    best_index = torch.argmax(
        scores,
        dim=1
    ).item()

    retrieved_image = str(
        df.iloc[best_index]["filename"]
    )

    correct = (
        retrieved_image.lower().strip()
        == filename.lower().strip()
    )

    i2t_results.append({
        "Test_No": int(row["Test_No"]),
        "Image": filename,
        "I2T_Retrieved": retrieved_image,
        "I2T_Score": float(
            scores[0, best_index]
        ),
        "I2T_Correct": correct
    })

    print(
        f"{int(row['Test_No']):03d} | "
        f"{filename:20s} | "
        f"{retrieved_image:20s} | "
        f"{'CORRECT' if correct else 'WRONG'}"
    )


# ============================================================
# TEXT → IMAGE
# ============================================================

print()
print("=" * 70)
print("TEXT → IMAGE TEST")
print("=" * 70)

results = []

for index, row in df.iterrows():

    test_no = int(row["Test_No"])

    filename = str(row["filename"])


    # ========================================================
    # ENGLISH
    # ========================================================

    english_query = str(
        row["english_caption"]
    )

    with torch.no_grad():

        inputs = clip_processor(
            text=[english_query],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        feature = clip_model.get_text_features(
            **inputs
        )

        if hasattr(feature, "pooler_output"):
            feature = feature.pooler_output

        if hasattr(feature, "last_hidden_state"):
            feature = feature.last_hidden_state[:, 0]

        feature = feature.float()

        english_query_embedding = F.normalize(
            feature,
            dim=1
        )

    english_scores = (
        english_query_embedding
        @ image_embeddings.T
    )

    english_best = torch.argmax(
        english_scores,
        dim=1
    ).item()

    # IMPORTANT:
    # Use dataframe order, not image_ids.npy
    english_retrieved = str(
        df.iloc[english_best]["filename"]
    )

    english_correct = (
        english_retrieved.lower().strip()
        == filename.lower().strip()
    )


    # ========================================================
    # TAMIL
    # ========================================================

    tamil_query = str(
        row["tamil_caption"]
    )

    tamil_query_embedding = encode_xlmr(
        tamil_query
    )

    tamil_scores = (
        tamil_query_embedding
        @ tamil_embeddings.T
    )

    tamil_best = torch.argmax(
        tamil_scores,
        dim=1
    ).item()

    # IMPORTANT:
    # Use dataframe order, not image_ids.npy
    tamil_retrieved = str(
        df.iloc[tamil_best]["filename"]
    )

    tamil_correct = (
        tamil_retrieved.lower().strip()
        == filename.lower().strip()
    )


    # ========================================================
    # HINDI
    # ========================================================

    hindi_query = str(
        row["hindi_caption"]
    )

    hindi_query_embedding = encode_xlmr(
        hindi_query
    )

    hindi_scores = (
        hindi_query_embedding
        @ hindi_embeddings.T
    )

    hindi_best = torch.argmax(
        hindi_scores,
        dim=1
    ).item()

    # IMPORTANT:
    # Use dataframe order, not image_ids.npy
    hindi_retrieved = str(
        df.iloc[hindi_best]["filename"]
    )

    hindi_correct = (
        hindi_retrieved.lower().strip()
        == filename.lower().strip()
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    results.append({

        "Test_No": test_no,

        "Expected_Image": filename,

        "English_Query": english_query,

        "English_Retrieved": english_retrieved,

        "English_Score": float(
            english_scores[0, english_best]
        ),

        "English_Correct": english_correct,

        "Tamil_Query": tamil_query,

        "Tamil_Retrieved": tamil_retrieved,

        "Tamil_Score": float(
            tamil_scores[0, tamil_best]
        ),

        "Tamil_Correct": tamil_correct,

        "Hindi_Query": hindi_query,

        "Hindi_Retrieved": hindi_retrieved,

        "Hindi_Score": float(
            hindi_scores[0, hindi_best]
        ),

        "Hindi_Correct": hindi_correct
    })


    print(
        f"{test_no:03d} | "
        f"EN={'OK' if english_correct else 'WRONG':5s} | "
        f"TA={'OK' if tamil_correct else 'WRONG':5s} | "
        f"HI={'OK' if hindi_correct else 'WRONG':5s}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

english_correct_count = int(
    results_df["English_Correct"].sum()
)

tamil_correct_count = int(
    results_df["Tamil_Correct"].sum()
)

hindi_correct_count = int(
    results_df["Hindi_Correct"].sum()
)

total_correct = (
    english_correct_count
    + tamil_correct_count
    + hindi_correct_count
)

total_queries = 300


print()
print("=" * 70)
print("FINAL 100-TEST SUMMARY")
print("=" * 70)

print(
    f"English : {english_correct_count}/100 "
    f"({english_correct_count:.2f}%)"
)

print(
    f"Tamil   : {tamil_correct_count}/100 "
    f"({tamil_correct_count:.2f}%)"
)

print(
    f"Hindi   : {hindi_correct_count}/100 "
    f"({hindi_correct_count:.2f}%)"
)

print(
    f"TOTAL   : {total_correct}/{total_queries} "
    f"({total_correct / total_queries * 100:.2f}%)"
)

print()
print("Results saved:")
print(OUTPUT_PATH)

print()
print("=" * 70)
print("STEP 41 COMPLETED")
print("=" * 70)