import os
import numpy as np
import faiss


# ============================================================
# STEP 43: BUILD FAISS INDEXES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")

os.makedirs(INDEX_DIR, exist_ok=True)


# ============================================================
# FILES
# ============================================================

IMAGE_FILE = os.path.join(
    EMBED_DIR,
    "image_embeddings.npy"
)

ENGLISH_FILE = os.path.join(
    EMBED_DIR,
    "english_xlmr_indicbertv2_embeddings.npy"
)

TAMIL_FILE = os.path.join(
    EMBED_DIR,
    "tamil_xlmr_indicbertv2_embeddings.npy"
)

HINDI_FILE = os.path.join(
    EMBED_DIR,
    "hindi_xlmr_indicbertv2_embeddings.npy"
)


# ============================================================
# INDEX FILES
# ============================================================

IMAGE_INDEX_FILE = os.path.join(
    INDEX_DIR,
    "image_index.faiss"
)

ENGLISH_INDEX_FILE = os.path.join(
    INDEX_DIR,
    "english_text_index.faiss"
)

TAMIL_INDEX_FILE = os.path.join(
    INDEX_DIR,
    "tamil_text_index.faiss"
)

HINDI_INDEX_FILE = os.path.join(
    INDEX_DIR,
    "hindi_text_index.faiss"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("STEP 43: BUILDING FAISS INDEXES")
print("=" * 70)

image_embeddings = np.load(IMAGE_FILE).astype(np.float32)
english_embeddings = np.load(ENGLISH_FILE).astype(np.float32)
tamil_embeddings = np.load(TAMIL_FILE).astype(np.float32)
hindi_embeddings = np.load(HINDI_FILE).astype(np.float32)


# ============================================================
# SHAPE CHECK
# ============================================================

print("\nEmbedding shapes:")

print("Image   :", image_embeddings.shape)
print("English :", english_embeddings.shape)
print("Tamil   :", tamil_embeddings.shape)
print("Hindi   :", hindi_embeddings.shape)

assert image_embeddings.shape == (100, 512)
assert english_embeddings.shape == (100, 512)
assert tamil_embeddings.shape == (100, 512)
assert hindi_embeddings.shape == (100, 512)


# ============================================================
# NORMALIZE
# ============================================================

def normalize_embeddings(x):

    norms = np.linalg.norm(
        x,
        axis=1,
        keepdims=True
    )

    return x / (norms + 1e-8)


image_embeddings = normalize_embeddings(
    image_embeddings
)

english_embeddings = normalize_embeddings(
    english_embeddings
)

tamil_embeddings = normalize_embeddings(
    tamil_embeddings
)

hindi_embeddings = normalize_embeddings(
    hindi_embeddings
)


# ============================================================
# BUILD IMAGE INDEX
# ============================================================

print("\nBuilding Image FAISS index...")

image_index = faiss.IndexFlatIP(512)

image_index.add(
    image_embeddings
)

faiss.write_index(
    image_index,
    IMAGE_INDEX_FILE
)

print(
    "Image index vectors:",
    image_index.ntotal
)


# ============================================================
# BUILD ENGLISH TEXT INDEX
# ============================================================

print("\nBuilding English Text FAISS index...")

english_index = faiss.IndexFlatIP(512)

english_index.add(
    english_embeddings
)

faiss.write_index(
    english_index,
    ENGLISH_INDEX_FILE
)

print(
    "English index vectors:",
    english_index.ntotal
)


# ============================================================
# BUILD TAMIL TEXT INDEX
# ============================================================

print("\nBuilding Tamil Text FAISS index...")

tamil_index = faiss.IndexFlatIP(512)

tamil_index.add(
    tamil_embeddings
)

faiss.write_index(
    tamil_index,
    TAMIL_INDEX_FILE
)

print(
    "Tamil index vectors:",
    tamil_index.ntotal
)


# ============================================================
# BUILD HINDI TEXT INDEX
# ============================================================

print("\nBuilding Hindi Text FAISS index...")

hindi_index = faiss.IndexFlatIP(512)

hindi_index.add(
    hindi_embeddings
)

faiss.write_index(
    hindi_index,
    HINDI_INDEX_FILE
)

print(
    "Hindi index vectors:",
    hindi_index.ntotal
)


# ============================================================
# VERIFY
# ============================================================

print("\n" + "=" * 70)
print("FAISS INDEX VERIFICATION")
print("=" * 70)

print(
    "Image index   :",
    faiss.read_index(IMAGE_INDEX_FILE).ntotal
)

print(
    "English index :",
    faiss.read_index(ENGLISH_INDEX_FILE).ntotal
)

print(
    "Tamil index   :",
    faiss.read_index(TAMIL_INDEX_FILE).ntotal
)

print(
    "Hindi index   :",
    faiss.read_index(HINDI_INDEX_FILE).ntotal
)

print("\nSTEP 43 COMPLETED")