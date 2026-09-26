import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EMBEDDINGS_DIR


# ============================================================
# CONFIGURATION
# ============================================================

LANGUAGES = ["english", "tamil", "hindi"]

XLMR_FILES = {
    "english": "english_text_embeddings.npy",
    "tamil": "tamil_text_embeddings.npy",
    "hindi": "hindi_text_embeddings.npy",
}

INDICBERT_FILES = {
    "english": "english_indicbert_embeddings.npy",
    "tamil": "tamil_indicbert_embeddings.npy",
    "hindi": "hindi_indicbert_embeddings.npy",
}

OUTPUT_FILES = {
    "english": "english_xlmr_indicbertv2_embeddings.npy",
    "tamil": "tamil_xlmr_indicbertv2_embeddings.npy",
    "hindi": "hindi_xlmr_indicbertv2_embeddings.npy",
}

INPUT_DIM = 768
FUSION_DIM = 512


# ============================================================
# SIMPLE NEURAL FUSION
# ============================================================

class TextFusion(nn.Module):

    def __init__(self):
        super().__init__()

        self.xlmr_projection = nn.Linear(INPUT_DIM, FUSION_DIM)
        self.indicbert_projection = nn.Linear(INPUT_DIM, FUSION_DIM)

        self.fusion = nn.Sequential(
            nn.Linear(FUSION_DIM * 2, FUSION_DIM),
            nn.ReLU(),
            nn.Linear(FUSION_DIM, FUSION_DIM)
        )

    def forward(self, xlmr, indicbert):

        xlmr = self.xlmr_projection(xlmr)
        indicbert = self.indicbert_projection(indicbert)

        fused = torch.cat([xlmr, indicbert], dim=1)

        fused = self.fusion(fused)

        fused = F.normalize(fused, p=2, dim=1)

        return fused


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("XLM-R + IndicBERTv2 TEXT FUSION")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nDevice:", device)

    model = TextFusion().to(device)

    # --------------------------------------------------------
    # NOTE:
    # This stage creates the fused representation.
    # Training/alignment with CLIP will be performed next.
    # --------------------------------------------------------

    model.eval()

    for language in LANGUAGES:

        print("\n" + "-" * 70)
        print("Processing:", language.upper())
        print("-" * 70)

        xlmr_path = os.path.join(
            EMBEDDINGS_DIR,
            XLMR_FILES[language]
        )

        indicbert_path = os.path.join(
            EMBEDDINGS_DIR,
            INDICBERT_FILES[language]
        )

        # Load XLM-R
        xlmr = np.load(xlmr_path).astype(np.float32)

        # Load IndicBERTv2
        indicbert = np.load(indicbert_path).astype(np.float32)

        print("XLM-R shape       :", xlmr.shape)
        print("IndicBERTv2 shape :", indicbert.shape)

        if xlmr.shape != indicbert.shape:
            raise ValueError(
                f"Shape mismatch for {language}: "
                f"XLM-R={xlmr.shape}, "
                f"IndicBERTv2={indicbert.shape}"
            )

        # Convert to tensors
        xlmr_tensor = torch.from_numpy(xlmr).to(device)
        indicbert_tensor = torch.from_numpy(indicbert).to(device)

        # Fusion
        with torch.no_grad():

            fused = model(
                xlmr_tensor,
                indicbert_tensor
            )

        fused_np = fused.cpu().numpy()

        # Save
        output_path = os.path.join(
            EMBEDDINGS_DIR,
            OUTPUT_FILES[language]
        )

        np.save(output_path, fused_np)

        print("Fused shape        :", fused_np.shape)
        print("Saved              :", output_path)

    print("\n" + "=" * 70)
    print("XLM-R + IndicBERTv2 FUSION COMPLETED")
    print("=" * 70)

    print("\nOutput dimension: 512")
    print("Languages       : English, Tamil, Hindi")
    print("XLM-R dimension : 768")
    print("IndicBERTv2     : 768")
    print("Fusion dimension: 512")


if __name__ == "__main__":
    main()