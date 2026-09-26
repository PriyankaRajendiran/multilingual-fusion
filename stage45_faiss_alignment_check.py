import os
import numpy as np
import faiss


# ============================================================
# STEP 45: FAISS IMAGE-TEXT ALIGNMENT CHECK
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")

IMAGE_FILE = os.path.join(
    EMBED_DIR, "image_embeddings.npy"
)

ENGLISH_FILE = os.path.join(
    EMBED_DIR, "english_xlmr_indicbertv2_embeddings.npy"
)

TAMIL_FILE = os.path.join(
    EMBED_DIR, "tamil_xlmr_indicbertv2_embeddings.npy"
)

HINDI_FILE = os.path.join(
    EMBED_DIR, "hindi_xlmr_indicbertv2_embeddings.npy"
)

IMAGE_INDEX_FILE = os.path.join(
    INDEX_DIR, "image_index.faiss"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("STEP 45: FAISS IMAGE-TEXT ALIGNMENT CHECK")
print("=" * 70)

image = np.load(IMAGE_FILE).astype(np.float32)
english = np.load(ENGLISH_FILE).astype(np.float32)
tamil = np.load(TAMIL_FILE).astype(np.float32)
hindi = np.load(HINDI_FILE).astype(np.float32)


# ============================================================
# NORMALIZE
# ============================================================

def normalize(x):
    return x / (
        np.linalg.norm(
            x,
            axis=1,
            keepdims=True
        ) + 1e-8
    )


image = normalize(image)
english = normalize(english)
tamil = normalize(tamil)
hindi = normalize(hindi)


# ============================================================
# DIRECT SIMILARITY MATRIX
# ============================================================

def check_alignment(name, text):

    similarity = np.matmul(
        image,
        text.T
    )

    diagonal = np.diag(similarity)

    # Exclude paired diagonal for negative statistics
    mask = ~np.eye(
        similarity.shape[0],
        dtype=bool
    )

    negatives = similarity[mask]

    mean_positive = diagonal.mean()
    mean_negative = negatives.mean()

    median_positive = np.median(diagonal)
    median_negative = np.median(negatives)

    margin = (
        mean_positive -
        mean_negative
    )

    # How often is the correct paired item top-1?
    top1 = 0

    for i in range(len(image)):

        ranking = np.argsort(
            -similarity[i]
        )

        if ranking[0] == i:
            top1 += 1

    top1_accuracy = (
        100.0 * top1 / len(image)
    )

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print(
        "Mean paired similarity   : {:.6f}".format(
            mean_positive
        )
    )

    print(
        "Mean non-paired similarity: {:.6f}".format(
            mean_negative
        )
    )

    print(
        "Median paired similarity : {:.6f}".format(
            median_positive
        )
    )

    print(
        "Median non-paired similarity: {:.6f}".format(
            median_negative
        )
    )

    print(
        "Positive-negative margin  : {:.6f}".format(
            margin
        )
    )

    print(
        "Paired Top-1 alignment    : {:.2f}%".format(
            top1_accuracy
        )
    )

    if margin > 0:
        print(
            "STATUS: Paired embeddings have higher "
            "average similarity."
        )
    else:
        print(
            "STATUS: Paired embeddings are NOT "
            "higher on average."
        )


# ============================================================
# CHECK ALL LANGUAGES
# ============================================================

check_alignment(
    "ENGLISH",
    english
)

check_alignment(
    "TAMIL",
    tamil
)

check_alignment(
    "HINDI",
    hindi
)


# ============================================================
# FAISS INDEX CHECK
# ============================================================

print("\n" + "=" * 70)
print("FAISS INDEX CHECK")
print("=" * 70)

index = faiss.read_index(
    IMAGE_INDEX_FILE
)

print(
    "Image FAISS vectors:",
    index.ntotal
)

print(
    "Image dimension:",
    index.d
)

print(
    "Index type:",
    type(index).__name__
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STEP 45 COMPLETED")
print("=" * 70)