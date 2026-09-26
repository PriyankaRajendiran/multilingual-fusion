"""
test_dataset.py
================
Section 28's "minimal test that confirms the 100-image dataset works".

Run AFTER prepare_dataset.py (and, optionally, translate_captions.py).
Does not need torch, CLIP, BLIP-2, XLM-R, or IndicBERT — it only checks
that the fused dataset files are structurally sound, so you can verify
Stage 1 (dataset) before installing/downloading any model.

Run:
    python test_dataset.py
"""

import os
import csv
import sys

import config


def fail(msg):
    print(f"[FAIL] {msg}")
    return False


def main():
    ok = True

    print("=" * 70)
    print("DATASET SANITY TEST")
    print("=" * 70)

    # 1. captions.csv exists and has NUM_IMAGES rows with the right columns
    if not os.path.exists(config.CAPTIONS_CSV):
        ok = fail(f"{config.CAPTIONS_CSV} missing. Run prepare_dataset.py first.")
    else:
        with open(config.CAPTIONS_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        expected_cols = {"image_id", "filename", "english_caption", "image_present"}
        if not expected_cols.issubset(rows[0].keys()):
            ok = fail(f"captions.csv missing expected columns {expected_cols}")
        if len(rows) != config.NUM_IMAGES:
            ok = fail(f"captions.csv has {len(rows)} rows, expected "
                       f"NUM_IMAGES={config.NUM_IMAGES}")
        else:
            print(f"[OK]   captions.csv has {len(rows)} rows.")

        ids = [r["image_id"] for r in rows]
        if len(ids) != len(set(ids)):
            ok = fail("Duplicate image_id values found in captions.csv.")
        else:
            print("[OK]   All image_id values are unique.")

        empty = [r for r in rows if not r["english_caption"].strip()]
        if empty:
            ok = fail(f"{len(empty)} rows have an empty english_caption.")
        else:
            print("[OK]   No empty English captions.")

        n_present = sum(1 for r in rows if r["image_present"] in ("True", "true", "1"))
        print(f"[INFO] Local image files found: {n_present} / {len(rows)} "
              f"(place the rest in {config.IMAGES_DIR} before running "
              f"extract_image_features.py).")

    # 2. multilingual_captions.csv, if present, is aligned to the same image_ids
    if os.path.exists(config.MULTILINGUAL_CSV):
        with open(config.MULTILINGUAL_CSV, newline="", encoding="utf-8") as f:
            mrows = list(csv.DictReader(f))
        m_ids = {r["image_id"] for r in mrows}
        base_ids = {r["image_id"] for r in rows} if os.path.exists(config.CAPTIONS_CSV) else set()
        if base_ids and m_ids != base_ids:
            ok = fail("multilingual_captions.csv image_ids do not exactly match "
                       "captions.csv image_ids -> the language fusion is misaligned.")
        else:
            print(f"[OK]   multilingual_captions.csv aligns to all "
                  f"{len(m_ids)} image_ids.")

        n_ta = sum(1 for r in mrows if r["tamil_caption"].strip())
        n_hi = sum(1 for r in mrows if r["hindi_caption"].strip())
        print(f"[INFO] Tamil captions present : {n_ta} / {len(mrows)}")
        print(f"[INFO] Hindi captions present : {n_hi} / {len(mrows)}")
    else:
        print(f"[INFO] {config.MULTILINGUAL_CSV} not found yet -> "
              f"run translate_captions.py to fuse in Tamil/Hindi.")

    print()
    if ok:
        print("RESULT: dataset structure OK.")
    else:
        print("RESULT: dataset has problems (see [FAIL] lines above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
