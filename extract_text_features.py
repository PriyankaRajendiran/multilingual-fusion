import os
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel

# ============================================================
# PATHS
# ============================================================

CSV_FILE = "dataset/multilingual_captions_clean.csv"
OUTPUT_DIR = "embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "xlm-roberta-base"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LANGUAGES = [
    "english",
    "tamil",
    "hindi"
]

TEXT_COLUMNS = {
    "english": "english_caption",
    "tamil": "tamil_caption",
    "hindi": "hindi_caption"
}

# ============================================================
# START
# ============================================================

print("=" * 70)
print("MULTILINGUAL TEXT FEATURE EXTRACTION")
print("=" * 70)

print("\nDevice:", DEVICE)
print("Model:", MODEL_NAME)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(CSV_FILE)

print("\nDataset rows:", len(df))
print("Languages:", LANGUAGES)

# ============================================================
# LOAD XLM-R
# ============================================================

print("\nLoading XLM-R tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Loading XLM-R model...")

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model = model.to(DEVICE)
model.eval()

print("XLM-R loaded successfully.")

# ============================================================
# EXTRACT TEXT EMBEDDINGS
# ============================================================

all_embeddings = {}

for language in LANGUAGES:

    print("\n" + "-" * 60)
    print("Language:", language)
    print("-" * 60)

    column = TEXT_COLUMNS[language]

    texts = df[column].astype(str).tolist()

    embeddings = []

    for i, text in enumerate(texts):

        print(f"Text: {i + 1}/{len(texts)}")

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = model(**inputs)

            # Mean pooling over valid tokens
            token_embeddings = outputs.last_hidden_state

            attention_mask = inputs["attention_mask"]

            mask = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()

            summed = torch.sum(
                token_embeddings * mask,
                dim=1
            )

            counts = torch.clamp(
                mask.sum(dim=1),
                min=1e-9
            )

            sentence_embedding = summed / counts

            # Normalize
            sentence_embedding = sentence_embedding / (
                sentence_embedding.norm(
                    dim=-1,
                    keepdim=True
                )
            )

        embeddings.append(
            sentence_embedding.cpu().numpy()[0]
        )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    all_embeddings[language] = embeddings

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{language}_text_embeddings.npy"
    )

    np.save(
        output_file,
        embeddings
    )

    print("\nSaved:", output_file)
    print("Shape:", embeddings.shape)

# ============================================================
# SAVE IMAGE IDs
# ============================================================

image_ids = df["image_id"].astype(str).values

np.save(
    os.path.join(
        OUTPUT_DIR,
        "text_image_ids.npy"
    ),
    image_ids
)

# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("TEXT FEATURE EXTRACTION COMPLETED")
print("=" * 70)

for language in LANGUAGES:
    print(
        language,
        ":",
        all_embeddings[language].shape
    )

print("\nSaved files:")
print("embeddings/english_text_embeddings.npy")
print("embeddings/tamil_text_embeddings.npy")
print("embeddings/hindi_text_embeddings.npy")
print("embeddings/text_image_ids.npy")

print("=" * 70)