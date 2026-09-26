import os
import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# FINAL FAISS RETRIEVAL DEMO
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

IMAGE_DIR = os.path.join(
    DATASET_DIR,
    "images"
)

CSV_PATH = os.path.join(
    DATASET_DIR,
    "multilingual_captions_clean.csv"
)

EMBED_DIR = os.path.join(
    BASE_DIR,
    "embeddings"
)

INDEX_DIR = os.path.join(
    BASE_DIR,
    "indexes"
)


# ============================================================
# EMBEDDINGS
# ============================================================

IMAGE_EMBEDDINGS = os.path.join(
    EMBED_DIR,
    "image_embeddings.npy"
)

ENGLISH_EMBEDDINGS = os.path.join(
    EMBED_DIR,
    "stage46_english_aligned_text_embeddings.npy"
)

TAMIL_EMBEDDINGS = os.path.join(
    EMBED_DIR,
    "stage46_tamil_aligned_text_embeddings.npy"
)

HINDI_EMBEDDINGS = os.path.join(
    EMBED_DIR,
    "stage46_hindi_aligned_text_embeddings.npy"
)

IMAGE_IDS = os.path.join(
    EMBED_DIR,
    "image_ids.npy"
)

TEXT_IMAGE_IDS = os.path.join(
    EMBED_DIR,
    "text_image_ids.npy"
)


# ============================================================
# FINAL FAISS INDEXES
# ============================================================

IMAGE_INDEX = os.path.join(
    INDEX_DIR,
    "final_image_index.faiss"
)

ENGLISH_INDEX = os.path.join(
    INDEX_DIR,
    "final_english_text_index.faiss"
)

TAMIL_INDEX = os.path.join(
    INDEX_DIR,
    "final_tamil_text_index.faiss"
)

HINDI_INDEX = os.path.join(
    INDEX_DIR,
    "final_hindi_text_index.faiss"
)


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("FINAL FAISS MULTILINGUAL RETRIEVAL DEMO")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

if not os.path.exists(CSV_PATH):

    raise FileNotFoundError(
        f"Dataset CSV not found:\n{CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

print(
    "Dataset rows:",
    len(df)
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print()
print("Loading final embeddings...")

image_embeddings = np.load(
    IMAGE_EMBEDDINGS
).astype(np.float32)

english_embeddings = np.load(
    ENGLISH_EMBEDDINGS
).astype(np.float32)

tamil_embeddings = np.load(
    TAMIL_EMBEDDINGS
).astype(np.float32)

hindi_embeddings = np.load(
    HINDI_EMBEDDINGS
).astype(np.float32)

image_ids = np.load(
    IMAGE_IDS,
    allow_pickle=True
)

text_image_ids = np.load(
    TEXT_IMAGE_IDS,
    allow_pickle=True
)


# ============================================================
# NORMALIZE
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
# LOAD FINAL FAISS INDEXES
# ============================================================

print()
print("Loading final FAISS indexes...")

image_index = faiss.read_index(
    IMAGE_INDEX
)

english_index = faiss.read_index(
    ENGLISH_INDEX
)

tamil_index = faiss.read_index(
    TAMIL_INDEX
)

hindi_index = faiss.read_index(
    HINDI_INDEX
)


print(
    "Image index   :",
    image_index.ntotal
)

print(
    "English index :",
    english_index.ntotal
)

print(
    "Tamil index   :",
    tamil_index.ntotal
)

print(
    "Hindi index   :",
    hindi_index.ntotal
)


# ============================================================
# DISPLAY IMAGE
# ============================================================

def display_image(
    image_filename,
    title
):

    image_path = os.path.join(
        IMAGE_DIR,
        str(image_filename)
    )

    if not os.path.exists(image_path):

        print(
            "\nImage not found:",
            image_path
        )

        return

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        plt.figure(
            figsize=(8, 6)
        )

        plt.imshow(image)

        plt.axis("off")

        plt.title(title)

        plt.show()

    except Exception as e:

        print(
            "Unable to display image:",
            e
        )


# ============================================================
# FIND DATASET ROW FROM IMAGE ID
# ============================================================

def get_row_from_image_id(
    image_id
):

    matches = np.where(
        image_ids.astype(str)
        == str(image_id)
    )[0]

    if len(matches) == 0:

        return None

    return int(matches[0])


# ============================================================
# IMAGE → TEXT
# ============================================================

def image_to_text():

    print()
    print("=" * 70)
    print("IMAGE → TEXT USING FINAL FAISS")
    print("=" * 70)

    print()
    print(
        "Enter dataset image number."
    )

    print(
        "Example: 0, 1, 2, ..."
    )

    try:

        query_index = int(
            input(
                "Enter image number: "
            ).strip()
        )

    except ValueError:

        print(
            "Please enter a valid number."
        )

        return

    if (
        query_index < 0
        or query_index >= len(image_embeddings)
    ):

        print(
            "Invalid image number."
        )

        return

    # --------------------------------------------------------
    # Query = CLIP image embedding
    # --------------------------------------------------------

    query = image_embeddings[
        query_index:query_index + 1
    ].copy()

    faiss.normalize_L2(query)

    print()
    print(
        "Searching English FAISS index..."
    )

    en_scores, en_indices = (
        english_index.search(
            query,
            5
        )
    )

    print()
    print(
        "Searching Tamil FAISS index..."
    )

    ta_scores, ta_indices = (
        tamil_index.search(
            query,
            5
        )
    )

    print()
    print(
        "Searching Hindi FAISS index..."
    )

    hi_scores, hi_indices = (
        hindi_index.search(
            query,
            5
        )
    )


    # --------------------------------------------------------
    # INPUT IMAGE
    # --------------------------------------------------------

    image_id = image_ids[
        query_index
    ]

    row_index = get_row_from_image_id(
        image_id
    )

    if row_index is not None:

        filename = df.iloc[
            row_index
        ]["filename"]

        print()
        print(
            "INPUT IMAGE:",
            filename
        )

        display_image(
            filename,
            "Input Image"
        )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    def print_results(
        language,
        scores,
        indices
    ):

        print()
        print("-" * 70)
        print(
            f"TOP 5 {language.upper()} TEXT RESULTS"
        )
        print("-" * 70)

        for rank in range(5):

            idx = int(
                indices[0][rank]
            )

            score = float(
                scores[0][rank]
            )

            result_image_id = (
                text_image_ids[idx]
            )

            result_row = (
                get_row_from_image_id(
                    result_image_id
                )
            )

            print()
            print(
                f"Rank {rank + 1}"
            )

            print(
                f"Image ID  : "
                f"{result_image_id}"
            )

            print(
                f"Similarity: "
                f"{score:.6f}"
            )

            if result_row is not None:

                row = df.iloc[
                    result_row
                ]

                print(
                    "English   :",
                    row["english_caption"]
                )

                print(
                    "Tamil     :",
                    row["tamil_caption"]
                )

                print(
                    "Hindi     :",
                    row["hindi_caption"]
                )

    print_results(
        "English",
        en_scores,
        en_indices
    )

    print_results(
        "Tamil",
        ta_scores,
        ta_indices
    )

    print_results(
        "Hindi",
        hi_scores,
        hi_indices
    )


# ============================================================
# TEXT → IMAGE
# ============================================================

def text_to_image():

    print()
    print("=" * 70)
    print("TEXT → IMAGE USING FINAL FAISS")
    print("=" * 70)

    print()
    print(
        "Select language:"
    )

    print(
        "1. English"
    )

    print(
        "2. Tamil"
    )

    print(
        "3. Hindi"
    )

    language_choice = input(
        "Enter choice: "
    ).strip()


    if language_choice == "1":

        language = "English"

        embeddings = english_embeddings

    elif language_choice == "2":

        language = "Tamil"

        embeddings = tamil_embeddings

    elif language_choice == "3":

        language = "Hindi"

        embeddings = hindi_embeddings

    else:

        print(
            "Invalid language."
        )

        return


    print()
    print(
        f"Select a {language} dataset query."
    )

    print(
        "Enter query number: 0 to",
        len(embeddings) - 1
    )

    try:

        query_index = int(
            input(
                "Enter query number: "
            ).strip()
        )

    except ValueError:

        print(
            "Please enter a valid number."
        )

        return


    if (
        query_index < 0
        or query_index >= len(embeddings)
    ):

        print(
            "Invalid query number."
        )

        return


    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query = embeddings[
        query_index:query_index + 1
    ].copy()

    faiss.normalize_L2(query)


    # --------------------------------------------------------
    # Search image FAISS
    # --------------------------------------------------------

    scores, indices = (
        image_index.search(
            query,
            5
        )
    )


    # --------------------------------------------------------
    # SHOW QUERY
    # --------------------------------------------------------

    query_image_id = text_image_ids[
        query_index
    ]

    query_row = get_row_from_image_id(
        query_image_id
    )

    print()
    print(
        "QUERY IMAGE ID:",
        query_image_id
    )

    if query_row is not None:

        print(
            f"{language} query:"
        )

        if language == "English":

            print(
                df.iloc[
                    query_row
                ]["english_caption"]
            )

        elif language == "Tamil":

            print(
                df.iloc[
                    query_row
                ]["tamil_caption"]
            )

        else:

            print(
                df.iloc[
                    query_row
                ]["hindi_caption"]
            )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"TOP 5 {language.upper()} TEXT → IMAGE RESULTS"
    )
    print("=" * 70)


    for rank in range(5):

        idx = int(
            indices[0][rank]
        )

        score = float(
            scores[0][rank]
        )

        result_image_id = image_ids[
            idx
        ]

        result_row = get_row_from_image_id(
            result_image_id
        )

        print()
        print(
            f"Rank {rank + 1}"
        )

        print(
            f"Image ID  : "
            f"{result_image_id}"
        )

        print(
            f"Similarity: "
            f"{score:.6f}"
        )

        if result_row is not None:

            print(
                "Filename  :",
                df.iloc[
                    result_row
                ]["filename"]
            )


    # --------------------------------------------------------
    # DISPLAY TOP RESULT
    # --------------------------------------------------------

    best_index = int(
        indices[0][0]
    )

    best_image_id = image_ids[
        best_index
    ]

    best_row = get_row_from_image_id(
        best_image_id
    )

    if best_row is not None:

        best_filename = df.iloc[
            best_row
        ]["filename"]

        display_image(
            best_filename,
            (
                f"{language} Text → Image\n"
                f"Top-1 Result: {best_filename}"
            )
        )


# ============================================================
# MAIN MENU
# ============================================================

def main():

    while True:

        print()
        print("=" * 70)
        print("FINAL MULTILINGUAL FAISS RETRIEVAL DEMO")
        print("=" * 70)

        print(
            "1. Image → Text"
        )

        print(
            "2. English Text → Image"
        )

        print(
            "3. Tamil Text → Image"
        )

        print(
            "4. Hindi Text → Image"
        )

        print(
            "5. Exit"
        )

        print("=" * 70)


        choice = input(
            "Enter your choice (1-5): "
        ).strip()


        if choice == "1":

            image_to_text()


        elif choice == "2":

            text_to_image()


        elif choice == "3":

            text_to_image()


        elif choice == "4":

            text_to_image()


        elif choice == "5":

            print()
            print(
                "Exiting demo..."
            )

            break


        else:

            print()
            print(
                "Invalid choice."
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()