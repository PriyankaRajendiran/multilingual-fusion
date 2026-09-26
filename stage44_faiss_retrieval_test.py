import os
import numpy as np
import faiss


# ============================================================
# STEP 44: ACTUAL FAISS RETRIEVAL TEST
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")


# ============================================================
# FILES
# ============================================================

IMAGE_EMBEDDINGS_FILE = os.path.join(
    EMBED_DIR,
    "image_embeddings.npy"
)

IMAGE_IDS_FILE = os.path.join(
    EMBED_DIR,
    "image_ids.npy"
)

TEXT_IMAGE_IDS_FILE = os.path.join(
    EMBED_DIR,
    "text_image_ids.npy"
)

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
# LOAD FAISS INDEXES
# ============================================================

print("=" * 70)
print("STEP 44: ACTUAL FAISS RETRIEVAL TEST")
print("=" * 70)

print("\nLoading FAISS indexes...")

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
# LOAD EMBEDDINGS AND IDS
# ============================================================

image_embeddings = np.load(
    IMAGE_EMBEDDINGS_FILE
).astype(np.float32)

image_ids = np.load(
    IMAGE_IDS_FILE
)

text_image_ids = np.load(
    TEXT_IMAGE_IDS_FILE,
    allow_pickle=True
)


print("\nEmbedding shape:")
print("Images:", image_embeddings.shape)

print("\nID shapes:")
print("Image IDs:", image_ids.shape)
print("Text image IDs:", text_image_ids.shape)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(x):

    norms = np.linalg.norm(
        x,
        axis=1,
        keepdims=True
    )

    return x / (norms + 1e-8)


image_embeddings = normalize(
    image_embeddings
)


# ============================================================
# HELPER
# ============================================================

def get_id(x):

    if isinstance(x, np.ndarray):
        x = x.item()

    return str(x)


# ============================================================
# I2T RETRIEVAL
# ============================================================

def test_i2t(
    image_index,
    image_embeddings,
    text_name,
    text_index,
    text_image_ids
):

    print("\n" + "=" * 70)
    print("IMAGE → TEXT")
    print("Language:", text_name)
    print("=" * 70)

    top_k = min(10, text_index.ntotal)

    scores, indices = text_index.search(
        image_embeddings,
        top_k
    )

    correct_r1 = 0
    correct_r5 = 0
    correct_r10 = 0

    for query_idx in range(
        len(image_embeddings)
    ):

        query_image_id = get_id(
            image_ids[query_idx]
        )

        retrieved_ids = [
            get_id(text_image_ids[j])
            for j in indices[query_idx]
        ]

        if query_image_id == retrieved_ids[0]:
            correct_r1 += 1

        if query_image_id in retrieved_ids[:5]:
            correct_r5 += 1

        if query_image_id in retrieved_ids[:10]:
            correct_r10 += 1

    n = len(image_embeddings)

    print(
        "I2T R@1  : {:.2f}%".format(
            100 * correct_r1 / n
        )
    )

    print(
        "I2T R@5  : {:.2f}%".format(
            100 * correct_r5 / n
        )
    )

    print(
        "I2T R@10 : {:.2f}%".format(
            100 * correct_r10 / n
        )
    )

    # --------------------------------------------------------
    # DISPLAY FIRST QUERY
    # --------------------------------------------------------

    print("\nExample query:")
    print("Image ID:", query_image_id)

    print("\nTop-10 retrieved text image IDs:")

    for rank in range(top_k):

        retrieved_id = get_id(
            text_image_ids[
                indices[0][rank]
            ]
        )

        print(
            "Rank {:2d} | Image ID {:15s} | "
            "Similarity {:.6f}".format(
                rank + 1,
                retrieved_id,
                scores[0][rank]
            )
        )


# ============================================================
# T2I RETRIEVAL
# ============================================================

def test_t2i(
    image_index,
    text_name,
    text_index,
    text_image_ids
):

    print("\n" + "=" * 70)
    print("TEXT → IMAGE")
    print("Language:", text_name)
    print("=" * 70)

    top_k = min(10, image_index.ntotal)

    # --------------------------------------------------------
    # Retrieve the text vectors directly from FAISS
    # --------------------------------------------------------

    text_vectors = text_index.reconstruct_n(
        0,
        text_index.ntotal
    )

    text_vectors = np.asarray(
        text_vectors,
        dtype=np.float32
    )

    scores, indices = image_index.search(
        text_vectors,
        top_k
    )

    correct_r1 = 0
    correct_r5 = 0
    correct_r10 = 0

    for query_idx in range(
        len(text_vectors)
    ):

        query_image_id = get_id(
            text_image_ids[query_idx]
        )

        retrieved_ids = [
            get_id(image_ids[j])
            for j in indices[query_idx]
        ]

        if query_image_id == retrieved_ids[0]:
            correct_r1 += 1

        if query_image_id in retrieved_ids[:5]:
            correct_r5 += 1

        if query_image_id in retrieved_ids[:10]:
            correct_r10 += 1

    n = len(text_vectors)

    print(
        "T2I R@1  : {:.2f}%".format(
            100 * correct_r1 / n
        )
    )

    print(
        "T2I R@5  : {:.2f}%".format(
            100 * correct_r5 / n
        )
    )

    print(
        "T2I R@10 : {:.2f}%".format(
            100 * correct_r10 / n
        )
    )

    # --------------------------------------------------------
    # DISPLAY FIRST QUERY
    # --------------------------------------------------------

    query_image_id = get_id(
        text_image_ids[0]
    )

    print("\nExample query:")
    print("Correct Image ID:", query_image_id)

    print("\nTop-10 retrieved image IDs:")

    for rank in range(top_k):

        retrieved_id = get_id(
            image_ids[
                indices[0][rank]
            ]
        )

        print(
            "Rank {:2d} | Image ID {:15s} | "
            "Similarity {:.6f}".format(
                rank + 1,
                retrieved_id,
                scores[0][rank]
            )
        )


# ============================================================
# RUN I2T
# ============================================================

test_i2t(
    image_index,
    image_embeddings,
    "English",
    english_index,
    text_image_ids
)

test_i2t(
    image_index,
    image_embeddings,
    "Tamil",
    tamil_index,
    text_image_ids
)

test_i2t(
    image_index,
    image_embeddings,
    "Hindi",
    hindi_index,
    text_image_ids
)


# ============================================================
# RUN T2I
# ============================================================

test_t2i(
    image_index,
    "English",
    english_index,
    text_image_ids
)

test_t2i(
    image_index,
    "Tamil",
    tamil_index,
    text_image_ids
)

test_t2i(
    image_index,
    "Hindi",
    hindi_index,
    text_image_ids
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("STEP 44 COMPLETED")
print("=" * 70)

print("\nFAISS retrieval search was executed using:")
print("faiss.IndexFlatIP")
print("index.search()")