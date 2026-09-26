import os
import numpy as np
import faiss


# ============================================================
# STEP 48: FINAL FAISS RETRIEVAL EVALUATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")

print("=" * 70)
print("STEP 48: FINAL FAISS RETRIEVAL EVALUATION")
print("=" * 70)


# ============================================================
# FILES
# ============================================================

IMAGE_EMBED_FILE = os.path.join(
    EMBED_DIR,
    "image_embeddings.npy"
)

ENGLISH_EMBED_FILE = os.path.join(
    EMBED_DIR,
    "stage46_english_aligned_text_embeddings.npy"
)

TAMIL_EMBED_FILE = os.path.join(
    EMBED_DIR,
    "stage46_tamil_aligned_text_embeddings.npy"
)

HINDI_EMBED_FILE = os.path.join(
    EMBED_DIR,
    "stage46_hindi_aligned_text_embeddings.npy"
)

IMAGE_IDS_FILE = os.path.join(
    EMBED_DIR,
    "image_ids.npy"
)

TEXT_IMAGE_IDS_FILE = os.path.join(
    EMBED_DIR,
    "text_image_ids.npy"
)


# ============================================================
# FINAL FAISS INDEXES
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
# LOAD INDEXES
# ============================================================

print("\nLoading final FAISS indexes...")

image_index = faiss.read_index(
    IMAGE_INDEX_FILE
)

english_index = faiss.read_index(
    ENGLISH_INDEX_FILE
)

tamil_index = faiss.read_index(
    TAMIL_INDEX_FILE
)

hindi_index = faiss.read_index(
    HINDI_INDEX_FILE
)

print("Image index   :", image_index.ntotal)
print("English index :", english_index.ntotal)
print("Tamil index   :", tamil_index.ntotal)
print("Hindi index   :", hindi_index.ntotal)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("\nLoading embeddings...")

image_embeddings = np.load(
    IMAGE_EMBED_FILE
).astype(np.float32)

english_embeddings = np.load(
    ENGLISH_EMBED_FILE
).astype(np.float32)

tamil_embeddings = np.load(
    TAMIL_EMBED_FILE
).astype(np.float32)

hindi_embeddings = np.load(
    HINDI_EMBED_FILE
).astype(np.float32)


# ============================================================
# LOAD IDS
# ============================================================

image_ids = np.load(
    IMAGE_IDS_FILE,
    allow_pickle=True
)

text_image_ids = np.load(
    TEXT_IMAGE_IDS_FILE,
    allow_pickle=True
)


print("\nID verification:")
print("Image IDs      :", image_ids.shape)
print("Text-image IDs :", text_image_ids.shape)


# ============================================================
# NORMALIZATION
# ============================================================

faiss.normalize_L2(
    image_embeddings
)

faiss.normalize_L2(
    english_embeddings
)

faiss.normalize_L2(
    tamil_embeddings
)

faiss.normalize_L2(
    hindi_embeddings
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_i2t(
    image_vectors,
    text_index,
    text_ids,
    image_ids,
    language
):

    print("\n" + "-" * 70)
    print("IMAGE-TO-TEXT:", language)
    print("-" * 70)

    scores, indices = text_index.search(
        image_vectors,
        10
    )

    r1 = 0
    r5 = 0
    r10 = 0

    for i in range(len(image_vectors)):

        correct_id = str(
            image_ids[i]
        )

        retrieved_ids = [
            str(text_ids[j])
            for j in indices[i]
        ]

        if correct_id in retrieved_ids[:1]:
            r1 += 1

        if correct_id in retrieved_ids[:5]:
            r5 += 1

        if correct_id in retrieved_ids[:10]:
            r10 += 1

    n = len(image_vectors)

    r1 = 100.0 * r1 / n
    r5 = 100.0 * r5 / n
    r10 = 100.0 * r10 / n

    print(f"I2T R@1  : {r1:.2f}%")
    print(f"I2T R@5  : {r5:.2f}%")
    print(f"I2T R@10 : {r10:.2f}%")

    print("\nExample query:")
    print("Image ID:", image_ids[0])

    print("\nTop-10 retrieved text image IDs:")

    for rank, idx in enumerate(
        indices[0],
        start=1
    ):

        print(
            f"Rank {rank:2d} "
            f"{text_ids[idx]} "
            f"sim {scores[0][rank-1]:.6f}"
        )

    return r1, r5, r10


def calculate_t2i(
    text_vectors,
    image_index,
    image_ids,
    text_ids,
    language
):

    print("\n" + "-" * 70)
    print("TEXT-TO-IMAGE:", language)
    print("-" * 70)

    scores, indices = image_index.search(
        text_vectors,
        10
    )

    r1 = 0
    r5 = 0
    r10 = 0

    for i in range(len(text_vectors)):

        correct_id = str(
            text_ids[i]
        )

        retrieved_ids = [
            str(image_ids[j])
            for j in indices[i]
        ]

        if correct_id in retrieved_ids[:1]:
            r1 += 1

        if correct_id in retrieved_ids[:5]:
            r5 += 1

        if correct_id in retrieved_ids[:10]:
            r10 += 1

    n = len(text_vectors)

    r1 = 100.0 * r1 / n
    r5 = 100.0 * r5 / n
    r10 = 100.0 * r10 / n

    print(f"T2I R@1  : {r1:.2f}%")
    print(f"T2I R@5  : {r5:.2f}%")
    print(f"T2I R@10 : {r10:.2f}%")

    print("\nExample query:")
    print("Correct Image ID:", text_ids[0])

    print("\nTop-10 retrieved image IDs:")

    for rank, idx in enumerate(
        indices[0],
        start=1
    ):

        print(
            f"Rank {rank:2d} "
            f"{image_ids[idx]} "
            f"sim {scores[0][rank-1]:.6f}"
        )

    return r1, r5, r10


# ============================================================
# RUN FINAL EVALUATION
# ============================================================

results = {}


# ------------------------------------------------------------
# ENGLISH
# ------------------------------------------------------------

results["English_I2T"] = calculate_i2t(
    image_embeddings,
    english_index,
    text_image_ids,
    image_ids,
    "English"
)

results["English_T2I"] = calculate_t2i(
    english_embeddings,
    image_index,
    image_ids,
    text_image_ids,
    "English"
)


# ------------------------------------------------------------
# TAMIL
# ------------------------------------------------------------

results["Tamil_I2T"] = calculate_i2t(
    image_embeddings,
    tamil_index,
    text_image_ids,
    image_ids,
    "Tamil"
)

results["Tamil_T2I"] = calculate_t2i(
    tamil_embeddings,
    image_index,
    image_ids,
    text_image_ids,
    "Tamil"
)


# ------------------------------------------------------------
# HINDI
# ------------------------------------------------------------

results["Hindi_I2T"] = calculate_i2t(
    image_embeddings,
    hindi_index,
    text_image_ids,
    image_ids,
    "Hindi"
)

results["Hindi_T2I"] = calculate_t2i(
    hindi_embeddings,
    image_index,
    image_ids,
    text_image_ids,
    "Hindi"
)


# ============================================================
# FINAL RESULTS TABLE
# ============================================================

print("\n")
print("=" * 70)
print("FINAL FAISS RETRIEVAL RESULTS")
print("=" * 70)

print(
    f"{'Language':<10}"
    f"{'Direction':<8}"
    f"{'R@1':>10}"
    f"{'R@5':>10}"
    f"{'R@10':>10}"
)

print("-" * 70)

for key, values in results.items():

    language, direction = key.split("_")

    print(
        f"{language:<10}"
        f"{direction:<8}"
        f"{values[0]:>9.2f}%"
        f"{values[1]:>9.2f}%"
        f"{values[2]:>9.2f}%"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

RESULT_FILE = os.path.join(
    EMBED_DIR,
    "stage48_final_faiss_retrieval_results.npy"
)

np.save(
    RESULT_FILE,
    results,
    allow_pickle=True
)


print("\nResults saved to:")
print(RESULT_FILE)

print("\n" + "=" * 70)
print("STEP 48 COMPLETED")
print("=" * 70)