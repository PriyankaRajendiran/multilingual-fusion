import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(
    PROJECT_ROOT, "dataset", "multilingual_captions.csv"
)

IMAGE_DIR = os.path.join(
    PROJECT_ROOT, "dataset", "images"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "embeddings"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("CLIP IMAGE FEATURE EXTRACTION")
print("=" * 70)

print("Device:", DEVICE)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Dataset rows:", len(df))
print("Images found:", len(os.listdir(IMAGE_DIR)))


# ============================================================
# LOAD CLIP
# ============================================================

MODEL_NAME = "openai/clip-vit-base-patch32"

print("\nLoading CLIP model...")
print(MODEL_NAME)

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

print("CLIP loaded successfully.")


# ============================================================
# EXTRACT IMAGE EMBEDDINGS
# ============================================================

image_embeddings = []
valid_ids = []

print("\nExtracting image embeddings...")

for i, row in df.iterrows():

    image_id = str(row["image_id"])
    filename = str(row["filename"])

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    try:

        image = Image.open(image_path).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }
        with torch.no_grad():
            vision_outputs = model.vision_model(
                pixel_values=inputs["pixel_values"]
            )

            pooled_output = vision_outputs.pooler_output

            embedding = model.visual_projection(
                pooled_output
            )

        # Normalize CLIP embedding
        embedding = embedding / embedding.norm(
            dim=-1,
            keepdim=True
        )

        image_embeddings.append(
            embedding.cpu().numpy()[0]
        )

        valid_ids.append(image_id)

        print(
            f"Image: {i + 1}/{len(df)}"
        )

    except Exception as e:

        print(
            f"[ERROR] {filename}: {e}"
        )


# ============================================================
# SAVE
# ============================================================

image_embeddings = np.array(
    image_embeddings,
    dtype=np.float32
)

output_file = os.path.join(
    OUTPUT_DIR,
    "image_embeddings.npy"
)

np.save(
    output_file,
    image_embeddings
)


# ============================================================
# SAVE IMAGE IDS
# ============================================================

ids_file = os.path.join(
    OUTPUT_DIR,
    "image_ids.npy"
)

np.save(
    ids_file,
    np.array(valid_ids)
)


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 70)
print("IMAGE FEATURE EXTRACTION COMPLETED")
print("=" * 70)

print("Embedding shape:", image_embeddings.shape)
print("Saved:", output_file)
print("Saved:", ids_file)