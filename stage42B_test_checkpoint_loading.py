import os
import torch
import torch.nn as nn


# ==============================================================
# STEP 42B: TEST CHECKPOINT LOADING - FIXED
# ==============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "embeddings",
    "stage28_80_20_alignment_model.pt"
)


# ==============================================================
# LANGUAGE PROJECTION
# ==============================================================

class LanguageProjection(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )


# ==============================================================
# MODEL
# ==============================================================

class MultilingualAlignmentModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.language_weights = nn.Parameter(
            torch.ones(3)
        )

        self.english_projection = LanguageProjection()

        self.tamil_projection = LanguageProjection()

        self.hindi_projection = LanguageProjection()


# ==============================================================
# LOAD CHECKPOINT
# ==============================================================

print("=" * 70)
print("STEP 42B: TEST CHECKPOINT LOADING - FIXED")
print("=" * 70)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

model = MultilingualAlignmentModel()

model.load_state_dict(
    checkpoint,
    strict=True
)

print("\n[OK] ALL 13 CHECKPOINT PARAMETERS LOADED SUCCESSFULLY.")

print("\nLanguage weights:")

print(
    model.language_weights.detach().numpy()
)

print("\nProjection shapes:")

print(
    "English:",
    tuple(
        model.english_projection[0].weight.shape
    ),
    "->",
    tuple(
        model.english_projection[2].weight.shape
    )
)

print(
    "Tamil  :",
    tuple(
        model.tamil_projection[0].weight.shape
    ),
    "->",
    tuple(
        model.tamil_projection[2].weight.shape
    )
)

print(
    "Hindi  :",
    tuple(
        model.hindi_projection[0].weight.shape
    ),
    "->",
    tuple(
        model.hindi_projection[2].weight.shape
    )
)

print("\n" + "=" * 70)
print("STEP 42B COMPLETED")
print("=" * 70)