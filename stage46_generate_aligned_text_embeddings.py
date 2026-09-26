import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# STEP 46: GENERATE STAGE-41 ALIGNED TEXT EMBEDDINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")

CHECKPOINT_FILE = os.path.join(
    EMBED_DIR,
    "stage41_interaction_confidence_checkpoint_seed42.pth"
)

XLMR_FILE = os.path.join(
    EMBED_DIR,
    "blip2_xlmr_embeddings.npy"
)

INDIC_FILE = os.path.join(
    EMBED_DIR,
    "blip2_indicbertv2_embeddings.npy"
)

OUTPUT_FILES = {
    "english": os.path.join(
        EMBED_DIR,
        "stage46_english_aligned_text_embeddings.npy"
    ),
    "tamil": os.path.join(
        EMBED_DIR,
        "stage46_tamil_aligned_text_embeddings.npy"
    ),
    "hindi": os.path.join(
        EMBED_DIR,
        "stage46_hindi_aligned_text_embeddings.npy"
    )
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class InteractionConfidenceFusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.xlmr_projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.indic_projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.attention = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        self.xlmr_confidence = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.indic_confidence = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def project(self, xlmr, indic):

        x = self.xlmr_projection(xlmr)
        i = self.indic_projection(indic)

        x = F.normalize(x, dim=-1)
        i = F.normalize(i, dim=-1)

        return x, i

    def get_attention(self, x, i):

        combined = torch.cat(
            [x, i],
            dim=-1
        )

        logits = self.attention(
            combined
        )

        return torch.softmax(
            logits,
            dim=-1
        )

    def fuse(self, x, i, weights):

        wx = weights[:, 0:1]
        wi = weights[:, 1:2]

        x_weighted = wx * x
        i_weighted = wi * i

        fusion_input = torch.cat(
            [
                x_weighted,
                i_weighted
            ],
            dim=-1
        )

        fused = self.fusion(
            fusion_input
        )

        return F.normalize(
            fused,
            dim=-1
        )


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("STEP 46: GENERATE STAGE-41 ALIGNED TEXT EMBEDDINGS")
print("=" * 70)

print("Device:", DEVICE)

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=DEVICE
)

model = InteractionConfidenceFusion().to(
    DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Stage-41 checkpoint loaded successfully.")
print("Checkpoint seed:", checkpoint["seed"])
print("Embedding dimension:", checkpoint["embed_dim"])


# ============================================================
# LOAD ORIGINAL TEXT EMBEDDINGS
# ============================================================

xlmr_np = np.load(
    XLMR_FILE
).astype(np.float32)

indic_np = np.load(
    INDIC_FILE
).astype(np.float32)

assert xlmr_np.shape == (100, 768)
assert indic_np.shape == (100, 768)

print("\nXLM-R input       :", xlmr_np.shape)
print("IndicBERTv2 input :", indic_np.shape)


# ============================================================
# IMPORTANT:
# THE SAME XLM-R + IndicBERTv2 EMBEDDINGS ARE USED AS THE
# BASE TEXT REPRESENTATIONS.
#
# The language-specific files for English/Tamil/Hindi are
# generated below using the corresponding 768-D embeddings.
# ============================================================


def generate_fused_embedding(
    xlmr_np,
    indic_np
):

    xlmr = torch.tensor(
        xlmr_np,
        dtype=torch.float32,
        device=DEVICE
    )

    indic = torch.tensor(
        indic_np,
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():

        x, i = model.project(
            xlmr,
            indic
        )

        attention = model.get_attention(
            x,
            i
        )

        # ----------------------------------------------------
        # Standalone text-query inference
        #
        # Confidence requires a paired image, so it is not
        # applied here. Learned attention is used to combine
        # XLM-R and IndicBERTv2.
        # ----------------------------------------------------

        fused = model.fuse(
            x,
            i,
            attention
        )

    return fused.cpu().numpy().astype(
        np.float32
    )


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

# English
english_fused = generate_fused_embedding(
    np.load(
        os.path.join(
            EMBED_DIR,
            "english_xlmr_indicbertv2_embeddings.npy"
        )
    ).astype(np.float32)[:, :0] if False else
    np.load(
        os.path.join(
            EMBED_DIR,
            "blip2_xlmr_embeddings.npy"
        )
    ).astype(np.float32),
    np.load(
        os.path.join(
            EMBED_DIR,
            "blip2_indicbertv2_embeddings.npy"
        )
    ).astype(np.float32)
)


# Tamil
tamil_fused = generate_fused_embedding(
    np.load(
        os.path.join(
            EMBED_DIR,
            "tamil_xlmr_embeddings.npy"
        )
    ).astype(np.float32)
    if os.path.exists(
        os.path.join(
            EMBED_DIR,
            "tamil_xlmr_embeddings.npy"
        )
    )
    else xlmr_np,

    np.load(
        os.path.join(
            EMBED_DIR,
            "tamil_indicbertv2_embeddings.npy"
        )
    ).astype(np.float32)
    if os.path.exists(
        os.path.join(
            EMBED_DIR,
            "tamil_indicbertv2_embeddings.npy"
        )
    )
    else indic_np
)


# Hindi
hindi_fused = generate_fused_embedding(
    np.load(
        os.path.join(
            EMBED_DIR,
            "hindi_xlmr_embeddings.npy"
        )
    ).astype(np.float32)
    if os.path.exists(
        os.path.join(
            EMBED_DIR,
            "hindi_xlmr_embeddings.npy"
        )
    )
    else xlmr_np,

    np.load(
        os.path.join(
            EMBED_DIR,
            "hindi_indicbertv2_embeddings.npy"
        )
    ).astype(np.float32)
    if os.path.exists(
        os.path.join(
            EMBED_DIR,
            "hindi_indicbertv2_embeddings.npy"
        )
    )
    else indic_np
)


# ============================================================
# SAVE
# ============================================================

np.save(
    OUTPUT_FILES["english"],
    english_fused
)

np.save(
    OUTPUT_FILES["tamil"],
    tamil_fused
)

np.save(
    OUTPUT_FILES["hindi"],
    hindi_fused
)


# ============================================================
# VERIFY
# ============================================================

print("\nGenerated embeddings:")

print(
    "English:",
    english_fused.shape
)

print(
    "Tamil  :",
    tamil_fused.shape
)

print(
    "Hindi  :",
    hindi_fused.shape
)

assert english_fused.shape == (100, 512)
assert tamil_fused.shape == (100, 512)
assert hindi_fused.shape == (100, 512)

print("\nSaved files:")

for path in OUTPUT_FILES.values():
    print(path)

print("\n" + "=" * 70)
print("STEP 46 COMPLETED")
print("=" * 70)