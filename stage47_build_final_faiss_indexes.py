import os
import numpy as np
import faiss


# ============================================================
# STEP 47: BUILD FINAL FAISS INDEXES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")

os.makedirs(INDEX_DIR, exist_ok=True)

print("=" * 70)
print("STEP 47: BUILD FINAL FAISS INDEXES")
print("=" * 70)


# ============================================================
# FILES
# ============================================================

IMAGE_FILE = os.path.join(
    EMBED_DIR,
    "image_embeddings.npy"
)

ENGLISH_FILE = os.path.join(
    EMBED_DIR,
    "stage46_english_aligned_text_embeddings.npy"
)

TAMIL_FILE = os.path.join(
    EMBED_DIR,
    "stage46_tamil_aligned_text_embeddings.npy"
)

HINDI_FILE = os.path.join(
    EMBED_DIR,
    "stage46_hindi_aligned_text_embeddings.npy"
)


# ============================================================
# OUTPUT INDEXES
# ============================================================

IMAGE_INDEX_FILE = os.path.join(
    INDEX_DIR,
    "final_image_index.faiss"
)

ENGLISH_INDEX_FILE = os.path.join(
    INDEX_DIR,
    "final_english_text_index.faiss"
)

TAMIL_INDEX_FILE = os.path.join(
    INDEX_DIR,
    "final_tamil_text_index.faiss"
)

HINDI_INDEX_FILE = os.path.join(
    INDEX_DIR,
    "final_hindi_text_index.faiss"
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("\nLoading embeddings...")

image_embeddings = np.load(
    IMAGE_FILE
).astype(np.float32)

english_embeddings = np.load(
    ENGLISH_FILE
).astype(np.float32)

tamil_embeddings = np.load(
    TAMIL_FILE
).astype(np.float32)

hindi_embeddings = np.load(
    HINDI_FILE
).astype(np.float32)


print("Image   :", image_embeddings.shape)
print("English :", english_embeddings.shape)
print("Tamil   :", tamil_embeddings.shape)
print("Hindi   :", hindi_embeddings.shape)


# ============================================================
# VALIDATION
# ============================================================

assert image_embeddings.shape == (100, 512)
assert english_embeddings.shape == (100, 512)
assert tamil_embeddings.shape == (100, 512)
assert hindi_embeddings.shape == (100, 512)


# ============================================================
# L2 NORMALIZATION
# ============================================================

print("\nNormalizing embeddings...")

image_embeddings = np.ascontiguousarray(
    image_embeddings
)

english_embeddings = np.ascontiguousarray(
    english_embeddings
)

tamil_embeddings = np.ascontiguousarray(
    tamil_embeddings
)

hindi_embeddings = np.ascontiguousarray(
    hindi_embeddings
)

faiss.normalize_L2(image_embeddings)
faiss.normalize_L2(english_embeddings)
faiss.normalize_L2(tamil_embeddings)
faiss.normalize_L2(hindi_embeddings)


# ============================================================
# BUILD IMAGE INDEX
# ============================================================

print("\nBuilding image FAISS index...")

image_index = faiss.IndexFlatIP(512)

image_index.add(
    image_embeddings
)

faiss.write_index(
    image_index,
    IMAGE_INDEX_FILE
)

print(
    "Image vectors:",
    image_index.ntotal
)


# ============================================================
# BUILD ENGLISH TEXT INDEX
# ============================================================

print("\nBuilding English text FAISS index...")

english_index = faiss.IndexFlatIP(512)

english_index.add(
    english_embeddings
)

faiss.write_index(
    english_index,
    ENGLISH_INDEX_FILE
)

print(
    "English vectors:",
    english_index.ntotal
)


# ============================================================
# BUILD TAMIL TEXT INDEX
# ============================================================

print("\nBuilding Tamil text FAISS index...")

tamil_index = faiss.IndexFlatIP(512)

tamil_index.add(
    tamil_embeddings
)

faiss.write_index(
    tamil_index,
    TAMIL_INDEX_FILE
)

print(
    "Tamil vectors:",
    tamil_index.ntotal
)


# ============================================================
# BUILD HINDI TEXT INDEX
# ============================================================

print("\nBuilding Hindi text FAISS index...")

hindi_index = faiss.IndexFlatIP(512)

hindi_index.add(
    hindi_embeddings
)

faiss.write_index(
    hindi_index,
    HINDI_INDEX_FILE
)

print(
    "Hindi vectors:",
    hindi_index.ntotal
)


# ============================================================
# VERIFY
# ============================================================

print("\n" + "=" * 70)
print("FINAL FAISS INDEX VERIFICATION")
print("=" * 70)

for name, path in [
    ("Image", IMAGE_INDEX_FILE),
    ("English", ENGLISH_INDEX_FILE),
    ("Tamil", TAMIL_INDEX_FILE),
    ("Hindi", HINDI_INDEX_FILE),
]:

    index = faiss.read_index(path)

    print(
        f"{name:8s} index : "
        f"{index.ntotal} vectors | "
        f"dimension = {index.d}"
    )


print("\n" + "=" * 70)
print("STEP 47 COMPLETED")
print("=" * 70)