"""
config.py
=========
Central configuration for the Multilingual Semantic Fusion Framework
(Image-to-Text / Text-to-Image retrieval using CLIP + BLIP-2 + XLM-R + IndicBERT).

Every other script imports its settings from here. Nothing in this file
downloads anything or touches the network — it only defines paths and
switches.
"""

import os

# ----------------------------------------------------------------------
# 0. PROJECT PATHS
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
CAPTIONS_CSV = os.path.join(DATASET_DIR, "captions.csv")                  # image_id, english_caption
MULTILINGUAL_CSV = os.path.join(DATASET_DIR, "multilingual_captions.csv")  # image_id, english_caption, tamil_caption, hindi_caption

EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "embeddings")
IMAGE_EMB_PATH = os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy")
TEXT_EMB_PATH = os.path.join(EMBEDDINGS_DIR, "text_embeddings.npy")
METADATA_CSV = os.path.join(EMBEDDINGS_DIR, "metadata.csv")

INDEX_DIR = os.path.join(PROJECT_ROOT, "indexes")
IMAGE_INDEX_PATH = os.path.join(INDEX_DIR, "image_index.faiss")
CAPTION_INDEX_PATH = os.path.join(INDEX_DIR, "caption_index.faiss")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

for _d in (DATASET_DIR, IMAGES_DIR, EMBEDDINGS_DIR, INDEX_DIR, OUTPUTS_DIR):
    os.makedirs(_d, exist_ok=True)

# ----------------------------------------------------------------------
# 1. EXPERIMENT SCALE  (Section 14 of the spec)
# ----------------------------------------------------------------------
# Change ONLY this number to scale the experiment up later.
# 10 / 50 / 100 / 500 / 1000 ... all work with the same code.
NUM_IMAGES = 100

# Official Multi30K "task1" split to draw the 100 images from.
# ("train" has 29,000 lines; "val" has 1,014; "test_2016" has 1,000.)
MULTI30K_SPLIT = "train"

# The English captions + image-filename list are plain text and small
# (a few hundred KB), so they ARE downloaded automatically from the
# official multi30k/dataset GitHub repo. The *images themselves* are
# Flickr30K photographs that Multi30K's licence does NOT allow to be
# redistributed — you must obtain them yourself (see README.md) and
# drop the ones you have into dataset/images/. The code below works
# correctly with however many of the 100 you actually have on disk.
MULTI30K_REPO_RAW = "https://raw.githubusercontent.com/multi30k/dataset/master"
MULTI30K_CAPTIONS_URL = f"{MULTI30K_REPO_RAW}/data/task1/raw/{MULTI30K_SPLIT}.en.gz"
MULTI30K_IMAGELIST_URL = f"{MULTI30K_REPO_RAW}/data/task1/image_splits/{MULTI30K_SPLIT}.txt"

# ----------------------------------------------------------------------
# 2. LANGUAGES
# ----------------------------------------------------------------------
LANGUAGES = ["english", "tamil", "hindi"]
LANG_CODES = {"english": "en", "tamil": "ta", "hindi": "hi"}

# ----------------------------------------------------------------------
# 3. LOW-RESOURCE MODE  (Section 15)
# ----------------------------------------------------------------------
LOW_RESOURCE_MODE = True

BATCH_SIZE = 8 if LOW_RESOURCE_MODE else 32

# ----------------------------------------------------------------------
# 4. DEVICE
# ----------------------------------------------------------------------
# torch is only imported here, lazily, so that scripts which don't touch
# any model (prepare_dataset.py, translate_captions.py, test_dataset.py)
# can run on a machine/environment where torch isn't installed yet.
def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception as e:
        print(f"[WARN] torch is not usable yet ({type(e).__name__}) -> "
              "defaulting DEVICE to 'cpu'. Install/repair requirements.txt "
              "before running feature-extraction scripts.")
        return "cpu"


DEVICE = get_device()

# ----------------------------------------------------------------------
# 5. MODELS
# ----------------------------------------------------------------------
# CLIP (Section 7) — ~600 MB download on first run.
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# BLIP-2 (Section 6) — Salesforce/blip2-opt-2.7b is ~15 GB and slow on
# CPU. It is OPTIONAL and OFF by default (MODE 1). See generate_captions.py.
BLIP2_MODEL_NAME = "Salesforce/blip2-opt-2.7b"
USE_BLIP2 = False          # False = MODE 1 (use existing captions.csv captions)
                            # True  = MODE 2 (actually run BLIP-2 to generate captions)

# XLM-R (Section 8) — ~1.1 GB download.
XLMR_MODEL_NAME = "xlm-roberta-base"

# IndicBERT (Section 9). ai4bharat/indic-bert is the standard public
# checkpoint (ALBERT-based, covers en + 11 Indian languages including
# Tamil and Hindi). ~130 MB — the smallest of the four models.
INDICBERT_MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"

# ----------------------------------------------------------------------
# 6. EMBEDDING / FUSION DIMENSIONS  (Sections 10-12)
# ----------------------------------------------------------------------
CLIP_DIM = 512          # openai/clip-vit-base-patch32 projection dim
XLMR_DIM = 768           # xlm-roberta-base hidden size
INDICBERT_DIM = 768      # ai4bharat/indic-bert hidden size
TEXT_FUSION_DIM = 512    # output dim of the XLM-R + IndicBERT fusion layer
EMBED_DIM = 512          # final common embedding space (Section 12)

# ----------------------------------------------------------------------
# 7. RETRIEVAL
# ----------------------------------------------------------------------
TOP_K_VALUES = [1, 5, 10]

# ----------------------------------------------------------------------
# 8. RANDOM SEED
# ----------------------------------------------------------------------
SEED = 42
