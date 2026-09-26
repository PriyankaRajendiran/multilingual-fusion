import os
import numpy as np

# ============================================================
# STEP 27
# FINAL RETRIEVAL EVALUATION
# ============================================================

EMBEDDINGS_DIR = "embeddings"

print("=" * 70)
print("FINAL CROSS-MODAL RETRIEVAL EVALUATION")
print("=" * 70)


# ============================================================
# LOAD IMAGE EMBEDDINGS
# ============================================================

image_embeddings = np.load(
    os.path.join(
        EMBEDDINGS_DIR,
        "image_embeddings.npy"
    )
)

image_ids = np.load(
    os.path.join(
        EMBEDDINGS_DIR,
        "image_ids.npy"
    ),
    allow_pickle=True
)

print("\nImage embeddings:", image_embeddings.shape)


# ============================================================
# NORMALIZE IMAGE EMBEDDINGS
# ============================================================

image_embeddings = (
    image_embeddings /
    np.linalg.norm(
        image_embeddings,
        axis=1,
        keepdims=True
    )
)


# ============================================================
# METHODS
# ============================================================

methods = {

    "Multilingual Fusion":
        "multilingual_text_embeddings.npy",

    "Tamil XLM-R + IndicBERTv2":
        "tamil_xlmr_indicbertv2_embeddings.npy",

    "Hindi XLM-R + IndicBERTv2":
        "hindi_xlmr_indicbertv2_embeddings.npy",

    "Multilingual Aligned":
        "multilingual_aligned_text_embeddings.npy",

    "BLIP-2 XLM-R + IndicBERTv2":
        "blip2_xlmr_indicbertv2_fused_embeddings.npy"
}


# ============================================================
# RECALL FUNCTION
# ============================================================

def calculate_recall(similarity, direction, k):

    correct = 0
    total = similarity.shape[0]

    for i in range(total):

        if direction == "image_to_text":

            rankings = np.argsort(
                -similarity[i]
            )

        else:

            rankings = np.argsort(
                -similarity[:, i]
            )

        if i in rankings[:k]:
            correct += 1

    return (correct / total) * 100


# ============================================================
# EVALUATE METHODS
# ============================================================

results = []


for method_name, filename in methods.items():

    print("\n" + "-" * 70)
    print(method_name)
    print("-" * 70)

    path = os.path.join(
        EMBEDDINGS_DIR,
        filename
    )

    text_embeddings = np.load(path)

    print("Text embeddings:", text_embeddings.shape)

    if text_embeddings.shape[0] != image_embeddings.shape[0]:
        raise ValueError(
            f"Sample count mismatch for {method_name}"
        )

    if text_embeddings.shape[1] != image_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch for {method_name}"
        )

    # Normalize
    text_embeddings = (
        text_embeddings /
        np.linalg.norm(
            text_embeddings,
            axis=1,
            keepdims=True
        )
    )

    # Similarity
    similarity = np.matmul(
        image_embeddings,
        text_embeddings.T
    )

    # I2T
    i2t_r1 = calculate_recall(
        similarity,
        "image_to_text",
        1
    )

    i2t_r5 = calculate_recall(
        similarity,
        "image_to_text",
        5
    )

    i2t_r10 = calculate_recall(
        similarity,
        "image_to_text",
        10
    )

    # T2I
    t2i_r1 = calculate_recall(
        similarity,
        "text_to_image",
        1
    )

    t2i_r5 = calculate_recall(
        similarity,
        "text_to_image",
        5
    )

    t2i_r10 = calculate_recall(
        similarity,
        "text_to_image",
        10
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
        method_name,
        i2t_r1,
        i2t_r5,
        i2t_r10,
        t2i_r1,
        t2i_r5,
        t2i_r10
    ])


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print("\n")
print("=" * 100)
print("FINAL RETRIEVAL RESULTS")
print("=" * 100)

print(
    f"{'Method':35}"
    f"{'I2T R@1':>10}"
    f"{'I2T R@5':>10}"
    f"{'I2T R@10':>10}"
    f"{'T2I R@1':>10}"
    f"{'T2I R@5':>10}"
    f"{'T2I R@10':>10}"
)

print("-" * 100)

for row in results:

    print(
        f"{row[0]:35}"
        f"{row[1]:10.2f}"
        f"{row[2]:10.2f}"
        f"{row[3]:10.2f}"
        f"{row[4]:10.2f}"
        f"{row[5]:10.2f}"
        f"{row[6]:10.2f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "final_retrieval_results.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FINAL CROSS-MODAL RETRIEVAL RESULTS\n"
    )

    f.write("=" * 100 + "\n\n")

    f.write(
        f"{'Method':35}"
        f"{'I2T R@1':>10}"
        f"{'I2T R@5':>10}"
        f"{'I2T R@10':>10}"
        f"{'T2I R@1':>10}"
        f"{'T2I R@5':>10}"
        f"{'T2I R@10':>10}\n"
    )

    f.write("-" * 100 + "\n")

    for row in results:

        f.write(
            f"{row[0]:35}"
            f"{row[1]:10.2f}"
            f"{row[2]:10.2f}"
            f"{row[3]:10.2f}"
            f"{row[4]:10.2f}"
            f"{row[5]:10.2f}"
            f"{row[6]:10.2f}\n"
        )

print("\nSaved:")
print("final_retrieval_results.txt")

print("\n" + "=" * 70)
print("STEP 27 COMPLETED")
print("=" * 70)