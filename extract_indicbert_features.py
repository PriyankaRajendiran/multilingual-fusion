"""
extract_indicbert_features.py
=============================
Extract IndicBERT embeddings for English, Tamil, and Hindi captions.

Input:
    dataset/multilingual_captions.csv

Output:
    embeddings/english_indicbert_embeddings.npy
    embeddings/tamil_indicbert_embeddings.npy
    embeddings/hindi_indicbert_embeddings.npy
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

from config import (
    MULTILINGUAL_CSV,
    EMBEDDINGS_DIR,
    INDICBERT_MODEL_NAME,
    INDICBERT_DIM,
    DEVICE,
    BATCH_SIZE,
)

print("=" * 70)
print("INDICBERT MULTILINGUAL FEATURE EXTRACTION")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

if not os.path.exists(MULTILINGUAL_CSV):
    raise FileNotFoundError(
        f"Dataset file not found:\n{MULTILINGUAL_CSV}"
    )

df = pd.read_csv(MULTILINGUAL_CSV)

required_columns = [
    "image_id",
    "english_caption",
    "tamil_caption",
    "hindi_caption",
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Required column '{col}' not found in {MULTILINGUAL_CSV}"
        )

print(f"Dataset rows: {len(df)}")
print(f"Model: {INDICBERT_MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"Expected embedding dimension: {INDICBERT_DIM}")

# ------------------------------------------------------------
# 2. Load IndicBERT
# ------------------------------------------------------------

print("\nLoading IndicBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    INDICBERT_MODEL_NAME
)

model = AutoModel.from_pretrained(
    INDICBERT_MODEL_NAME
)

model.to(DEVICE)
model.eval()

print("IndicBERT loaded successfully.")

# ------------------------------------------------------------
# 3. Embedding function
# ------------------------------------------------------------

def extract_embeddings(texts):
    all_embeddings = []

    for start in range(0, len(texts), BATCH_SIZE):

        batch_texts = texts[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

            hidden_states = outputs.last_hidden_state

            attention_mask = inputs["attention_mask"].unsqueeze(-1)

            masked_hidden = hidden_states * attention_mask

            summed = masked_hidden.sum(dim=1)

            counts = attention_mask.sum(dim=1).clamp(min=1)

            embeddings = summed / counts

            embeddings = torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1
            )

        all_embeddings.append(
            embeddings.cpu().numpy()
        )

        print(
            f"Processed {min(start + BATCH_SIZE, len(texts))}"
            f"/{len(texts)}"
        )

    return np.vstack(all_embeddings)


# ------------------------------------------------------------
# 4. Process each language
# ------------------------------------------------------------

languages = {
    "english": "english_caption",
    "tamil": "tamil_caption",
    "hindi": "hindi_caption",
}

for language, column in languages.items():

    print("\n" + "-" * 70)
    print(f"Processing: {language.upper()}")
    print("-" * 70)

    texts = (
        df[column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    embeddings = extract_embeddings(texts)

    print(
        f"{language.capitalize()} shape: "
        f"{embeddings.shape}"
    )

    output_path = os.path.join(
        EMBEDDINGS_DIR,
        f"{language}_indicbert_embeddings.npy"
    )

    np.save(
        output_path,
        embeddings
    )

    print(f"Saved: {output_path}")


# ------------------------------------------------------------
# 5. Save image IDs for consistency
# ------------------------------------------------------------

image_ids = df["image_id"].astype(str).values

image_ids_path = os.path.join(
    EMBEDDINGS_DIR,
    "indicbert_text_image_ids.npy"
)

np.save(
    image_ids_path,
    image_ids
)

# ------------------------------------------------------------
# 6. Final verification
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INDICBERT EXTRACTION COMPLETED")
print("=" * 70)

for language in languages:

    path = os.path.join(
        EMBEDDINGS_DIR,
        f"{language}_indicbert_embeddings.npy"
    )

    arr = np.load(path)

    print(
        f"{language.capitalize():10s}: "
        f"{arr.shape}"
    )

print("\nAll IndicBERT embeddings saved successfully.")