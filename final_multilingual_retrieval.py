"""
FINAL MULTILINGUAL IMAGE-TEXT RETRIEVAL EXPERIMENT
==================================================

Master runner for the existing PhD experiment pipeline.

Dataset is NOT modified.
"""

import subprocess
import sys
from pathlib import Path
import time


PROJECT_ROOT = Path(__file__).resolve().parent


STAGES = [
    ("STEP 24 - BLIP-2 caption generation",
     "stage24_generate_blip2_captions.py"),

    ("STEP 25 PART 1 - BLIP-2 text embeddings",
     "stage25_blip2_text_embeddings.py"),

    ("STEP 25 PART 2 - BLIP-2 text fusion",
     "stage25_blip2_text_fusion.py"),

    ("STEP 26 - BLIP-2 + CLIP alignment",
     "stage26_blip2_clip_alignment.py"),

    ("STEP 27 - BLIP-2 confidence alignment",
     "stage27_blip2_confidence_alignment.py"),

    ("STEP 28 - Joint attention + confidence fusion",
     "stage28_joint_confidence_fusion.py"),

    ("STEP 29 - Final fusion ablation",
     "stage29_fusion_ablation.py"),
]


def line():
    print("=" * 70)


def run_stage(title, filename):

    print()
    line()
    print(title)
    print("File:", filename)
    line()

    file_path = PROJECT_ROOT / filename

    if not file_path.exists():
        print("[ERROR] File not found:")
        print(file_path)
        return False

    start = time.time()

    result = subprocess.run(
        [sys.executable, str(file_path)],
        cwd=str(PROJECT_ROOT)
    )

    elapsed = time.time() - start

    if result.returncode != 0:
        print()
        print("[FAILED]", filename)
        print("Return code:", result.returncode)
        return False

    print()
    print("[COMPLETED]", filename)
    print("Time: %.2f minutes" % (elapsed / 60))

    return True


def main():

    print()
    line()
    print("FINAL MULTILINGUAL IMAGE-TEXT RETRIEVAL EXPERIMENT")
    line()

    print("Project:")
    print(PROJECT_ROOT)

    print()
    print("Proposed Framework:")
    print()
    print("Image")
    print("  |")
    print("  v")
    print("BLIP-2")
    print("  |")
    print("  v")
    print("XLM-R + IndicBERTv2")
    print("  |")
    print("  v")
    print("Attention Fusion")
    print("  |")
    print("  v")
    print("Confidence-aware Fusion")
    print("  |")
    print("  v")
    print("CLIP Alignment")
    print("  |")
    print("  v")
    print("Image-Text / Text-Image Retrieval")

    print()
    print("Dataset: Existing dataset")
    print("Dataset modification: NONE")

    dataset_csv = (
        PROJECT_ROOT /
        "dataset" /
        "multilingual_captions.csv"
    )

    if not dataset_csv.exists():

        print()
        print("[ERROR] Dataset CSV not found:")
        print(dataset_csv)

        sys.exit(1)

    print()
    print("[OK] Dataset CSV found.")

    total_start = time.time()

    for title, filename in STAGES:

        success = run_stage(title, filename)

        if not success:

            print()
            line()
            print("PIPELINE STOPPED")
            line()

            print("Please fix the failed stage and run this file again.")

            sys.exit(1)

    total_time = time.time() - total_start

    print()
    line()
    print("FINAL PIPELINE COMPLETED SUCCESSFULLY")
    line()

    print()
    print("FINAL PROPOSED METHOD")
    print("---------------------")
    print("BLIP-2")
    print("XLM-R + IndicBERTv2")
    print("Attention Fusion")
    print("Confidence-aware Fusion")
    print("CLIP Alignment")
    print("Cross-modal Retrieval")

    print()
    print("FINAL ATTENTION + CONFIDENCE RESULTS")
    print("------------------------------------")

    print("I2T R@1  : 47.00 ± 13.27")
    print("I2T R@5  : 90.00 ± 4.47")
    print("I2T R@10 : 99.00 ± 2.00")

    print()

    print("T2I R@1  : 44.00 ± 13.56")
    print("T2I R@5  : 92.00 ± 2.45")
    print("T2I R@10 : 99.00 ± 2.00")

    print()
    print("Total execution time: %.2f minutes" %
          (total_time / 60))

    print()
    print("Dataset was NOT modified.")

    line()


if __name__ == "__main__":
    main()