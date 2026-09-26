import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# STEP 41
# IMAGE-LANGUAGE INTERACTION CONFIDENCE FUSION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")

IMAGE_FILE = os.path.join(
    EMBED_DIR, "image_embeddings.npy"
)

XLMR_FILE = os.path.join(
    EMBED_DIR, "blip2_xlmr_embeddings.npy"
)

INDIC_FILE = os.path.join(
    EMBED_DIR, "blip2_indicbertv2_embeddings.npy"
)

OUTPUT_FILE = os.path.join(
    EMBED_DIR,
    "stage41_interaction_confidence_results.npy"
)

CHECKPOINT_FILE = os.path.join(
    EMBED_DIR,
    "stage41_interaction_confidence_checkpoint_seed42.pth"
)

SEEDS = [42, 10, 20, 30, 40]

TRAIN_RATIO = 0.80
EPOCHS = 100
LEARNING_RATE = 0.001
TEMPERATURE = 0.07

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MODEL
# ============================================================

class InteractionConfidenceFusion(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # XLM-R projection
        # ----------------------------------------------------

        self.xlmr_projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        # ----------------------------------------------------
        # IndicBERT projection
        # ----------------------------------------------------

        self.indic_projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        # ----------------------------------------------------
        # Language attention
        # ----------------------------------------------------

        self.attention = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        # ----------------------------------------------------
        # Interaction confidence
        #
        # Input:
        # language
        # image
        # language * image
        # |language - image|
        #
        # 512 + 512 + 512 + 512 = 2048
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Final fusion
        # ----------------------------------------------------

        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    # ========================================================
    # PROJECTION
    # ========================================================

    def project(
        self,
        xlmr,
        indic
    ):

        x = self.xlmr_projection(xlmr)

        i = self.indic_projection(indic)

        x = F.normalize(
            x,
            dim=-1
        )

        i = F.normalize(
            i,
            dim=-1
        )

        return x, i

    # ========================================================
    # ATTENTION
    # ========================================================

    def get_attention(
        self,
        x,
        i
    ):

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

    # ========================================================
    # INTERACTION FEATURES
    # ========================================================

    def interaction(
        self,
        language,
        image
    ):

        product = (
            language * image
        )

        difference = torch.abs(
            language - image
        )

        features = torch.cat(
            [
                language,
                image,
                product,
                difference
            ],
            dim=-1
        )

        return features

    # ========================================================
    # CONFIDENCE
    # ========================================================

    def get_confidence(
        self,
        x,
        i,
        image
    ):

        x_features = self.interaction(
            x,
            image
        )

        i_features = self.interaction(
            i,
            image
        )

        cx = self.xlmr_confidence(
            x_features
        )

        ci = self.indic_confidence(
            i_features
        )

        return cx, ci

    # ========================================================
    # FUSION
    # ========================================================

    def fuse(
        self,
        x,
        i,
        weights
    ):

        wx = weights[:, 0:1]

        wi = weights[:, 1:2]

        x_weighted = (
            wx * x
        )

        i_weighted = (
            wi * i
        )

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
# CONTRASTIVE LOSS
# ============================================================

def contrastive_loss(
    image_emb,
    text_emb,
    temperature=0.07
):

    logits = torch.matmul(
        image_emb,
        text_emb.T
    ) / temperature

    labels = torch.arange(
        image_emb.size(0),
        device=image_emb.device
    )

    loss_i2t = F.cross_entropy(
        logits,
        labels
    )

    loss_t2i = F.cross_entropy(
        logits.T,
        labels
    )

    return (
        loss_i2t + loss_t2i
    ) / 2


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def retrieval_metrics(
    image_emb,
    text_emb
):

    similarity = torch.matmul(
        image_emb,
        text_emb.T
    )

    n = similarity.size(0)

    i2t = {
        1: 0,
        5: 0,
        10: 0
    }

    t2i = {
        1: 0,
        5: 0,
        10: 0
    }

    # --------------------------------------------------------
    # IMAGE -> TEXT
    # --------------------------------------------------------

    for i in range(n):

        ranking = torch.argsort(
            similarity[i],
            descending=True
        )

        position = (
            ranking == i
        ).nonzero(
            as_tuple=False
        ).item()

        for k in [1, 5, 10]:

            if position < k:
                i2t[k] += 1

    # --------------------------------------------------------
    # TEXT -> IMAGE
    # --------------------------------------------------------

    for i in range(n):

        ranking = torch.argsort(
            similarity[:, i],
            descending=True
        )

        position = (
            ranking == i
        ).nonzero(
            as_tuple=False
        ).item()

        for k in [1, 5, 10]:

            if position < k:
                t2i[k] += 1

    return {
        "I2T_R1": 100 * i2t[1] / n,
        "I2T_R5": 100 * i2t[5] / n,
        "I2T_R10": 100 * i2t[10] / n,
        "T2I_R1": 100 * t2i[1] / n,
        "T2I_R5": 100 * t2i[5] / n,
        "T2I_R10": 100 * t2i[10] / n
    }


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("=" * 70)
print("STEP 41")
print("IMAGE-LANGUAGE INTERACTION CONFIDENCE FUSION")
print("=" * 70)

print("Device:", DEVICE)

image_embeddings = np.load(
    IMAGE_FILE
).astype(np.float32)

xlmr_embeddings = np.load(
    XLMR_FILE
).astype(np.float32)

indic_embeddings = np.load(
    INDIC_FILE
).astype(np.float32)


# ============================================================
# SHAPE CHECK
# ============================================================

assert image_embeddings.shape == (100, 512)

assert xlmr_embeddings.shape == (100, 768)

assert indic_embeddings.shape == (100, 768)

print(
    "Image embeddings:",
    image_embeddings.shape
)

print(
    "XLM-R embeddings:",
    xlmr_embeddings.shape
)

print(
    "IndicBERT embeddings:",
    indic_embeddings.shape
)


# ============================================================
# NORMALIZE INPUT EMBEDDINGS
# ============================================================

image_embeddings = (
    image_embeddings /
    (
        np.linalg.norm(
            image_embeddings,
            axis=1,
            keepdims=True
        ) + 1e-8
    )
)

xlmr_embeddings = (
    xlmr_embeddings /
    (
        np.linalg.norm(
            xlmr_embeddings,
            axis=1,
            keepdims=True
        ) + 1e-8
    )
)

indic_embeddings = (
    indic_embeddings /
    (
        np.linalg.norm(
            indic_embeddings,
            axis=1,
            keepdims=True
        ) + 1e-8
    )
)


# ============================================================
# RESULTS
# ============================================================

all_results = []


# ============================================================
# FIVE SEEDS
# ============================================================

for seed in SEEDS:

    print("\n" + "=" * 70)
    print("SEED:", seed)
    print("=" * 70)

    set_seed(seed)

    model = InteractionConfidenceFusion().to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    image_tensor = torch.tensor(
        image_embeddings,
        dtype=torch.float32,
        device=DEVICE
    )

    xlmr_tensor = torch.tensor(
        xlmr_embeddings,
        dtype=torch.float32,
        device=DEVICE
    )

    indic_tensor = torch.tensor(
        indic_embeddings,
        dtype=torch.float32,
        device=DEVICE
    )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    indices = np.arange(100)

    np.random.shuffle(indices)

    train_size = int(
        TRAIN_RATIO * 100
    )

    train_indices = torch.tensor(
        indices[:train_size],
        dtype=torch.long,
        device=DEVICE
    )

    test_indices = torch.tensor(
        indices[train_size:],
        dtype=torch.long,
        device=DEVICE
    )

    print(
        "Training samples:",
        len(train_indices)
    )

    print(
        "Testing samples :",
        len(test_indices)
    )

    # ========================================================
    # TRAINING
    # ========================================================

    model.train()

    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        x, i = model.project(
            xlmr_tensor[train_indices],
            indic_tensor[train_indices]
        )

        target_images = image_tensor[
            train_indices
        ]

        # ----------------------------------------------------
        # IMAGE-LANGUAGE SIMILARITY
        # ----------------------------------------------------

        sim_x = torch.sum(
            x * target_images,
            dim=-1
        )

        sim_i = torch.sum(
            i * target_images,
            dim=-1
        )

        # ----------------------------------------------------
        # SIMILARITY TARGET
        # ----------------------------------------------------

        target_x = (
            sim_x.detach() + 1.0
        ) / 2.0

        target_i = (
            sim_i.detach() + 1.0
        ) / 2.0

        target_x = torch.clamp(
            target_x,
            0.0,
            1.0
        )

        target_i = torch.clamp(
            target_i,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # INTERACTION CONFIDENCE
        # ----------------------------------------------------

        cx, ci = model.get_confidence(
            x,
            i,
            target_images
        )

        cx = cx.squeeze(1)

        ci = ci.squeeze(1)

        confidence_loss = (
            F.mse_loss(
                cx,
                target_x
            )
            +
            F.mse_loss(
                ci,
                target_i
            )
        )

        # ----------------------------------------------------
        # LANGUAGE ATTENTION
        # ----------------------------------------------------

        attention = model.get_attention(
            x,
            i
        )

        # ----------------------------------------------------
        # ATTENTION × CONFIDENCE
        # ----------------------------------------------------

        wx = (
            attention[:, 0]
            * cx
        )

        wi = (
            attention[:, 1]
            * ci
        )

        weight_sum = (
            wx + wi + 1e-8
        )

        wx = wx / weight_sum

        wi = wi / weight_sum

        weights = torch.stack(
            [
                wx,
                wi
            ],
            dim=1
        )

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        fused = model.fuse(
            x,
            i,
            weights
        )

        # ----------------------------------------------------
        # RETRIEVAL LOSS
        # ----------------------------------------------------

        retrieval_loss = contrastive_loss(
            target_images,
            fused,
            TEMPERATURE
        )

        # ----------------------------------------------------
        # TOTAL LOSS
        # ----------------------------------------------------

        loss = (
            retrieval_loss
            +
            0.5 * confidence_loss
        )

        loss.backward()

        optimizer.step()

        if (
            epoch == 0
            or (epoch + 1) % 20 == 0
            or epoch == EPOCHS - 1
        ):

            print(
                "Epoch {:3d} | "
                "Retrieval {:.6f} | "
                "Confidence {:.6f} | "
                "Total {:.6f}".format(
                    epoch + 1,
                    retrieval_loss.item(),
                    confidence_loss.item(),
                    loss.item()
                )
            )

    # ========================================================
    # TEST
    # ========================================================

    model.eval()

    with torch.no_grad():

        x_test, i_test = model.project(
            xlmr_tensor[test_indices],
            indic_tensor[test_indices]
        )

        image_test = image_tensor[
            test_indices
        ]

        # ----------------------------------------------------
        # INTERACTION CONFIDENCE
        # ----------------------------------------------------

        cx, ci = model.get_confidence(
            x_test,
            i_test,
            image_test
        )

        cx = cx.squeeze(1)

        ci = ci.squeeze(1)

        # ----------------------------------------------------
        # ATTENTION
        # ----------------------------------------------------

        attention = model.get_attention(
            x_test,
            i_test
        )

        # ----------------------------------------------------
        # EFFECTIVE WEIGHTS
        # ----------------------------------------------------

        wx = (
            attention[:, 0]
            * cx
        )

        wi = (
            attention[:, 1]
            * ci
        )

        weight_sum = (
            wx + wi + 1e-8
        )

        wx = wx / weight_sum

        wi = wi / weight_sum

        weights = torch.stack(
            [
                wx,
                wi
            ],
            dim=1
        )

        # ----------------------------------------------------
        # FUSED REPRESENTATION
        # ----------------------------------------------------

        fused_test = model.fuse(
            x_test,
            i_test,
            weights
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        metrics = retrieval_metrics(
            image_test,
            fused_test
        )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        mean_cx = cx.mean().item()
        std_cx = cx.std().item()

        mean_ci = ci.mean().item()
        std_ci = ci.std().item()

        mean_wx = wx.mean().item()
        mean_wi = wi.mean().item()

        mean_ax = attention[:, 0].mean().item()
        mean_ai = attention[:, 1].mean().item()

    # ========================================================
    # PRINT TEST RESULTS
    # ========================================================

    print("\nTEST RESULTS")

    print(
        "I2T R@1  : {:.2f}%".format(
            metrics["I2T_R1"]
        )
    )

    print(
        "I2T R@5  : {:.2f}%".format(
            metrics["I2T_R5"]
        )
    )

    print(
        "I2T R@10 : {:.2f}%".format(
            metrics["I2T_R10"]
        )
    )

    print(
        "T2I R@1  : {:.2f}%".format(
            metrics["T2I_R1"]
        )
    )

    print(
        "T2I R@5  : {:.2f}%".format(
            metrics["T2I_R5"]
        )
    )

    print(
        "T2I R@10 : {:.2f}%".format(
            metrics["T2I_R10"]
        )
    )

    print("\nINTERACTION CONFIDENCE")

    print(
        "XLM-R confidence     : {:.4f} ± {:.4f}".format(
            mean_cx,
            std_cx
        )
    )

    print(
        "IndicBERT confidence : {:.4f} ± {:.4f}".format(
            mean_ci,
            std_ci
        )
    )

    print("\nATTENTION")

    print(
        "XLM-R attention     : {:.4f}".format(
            mean_ax
        )
    )

    print(
        "IndicBERT attention : {:.4f}".format(
            mean_ai
        )
    )

    print("\nEFFECTIVE WEIGHTS")

    print(
        "XLM-R effective     : {:.4f}".format(
            mean_wx
        )
    )

    print(
        "IndicBERT effective : {:.4f}".format(
            mean_wi
        )
    )

    # ========================================================
    # STORE RESULTS
    # ========================================================

    result = {

        "seed": seed,

        "I2T_R1":
            metrics["I2T_R1"],

        "I2T_R5":
            metrics["I2T_R5"],

        "I2T_R10":
            metrics["I2T_R10"],

        "T2I_R1":
            metrics["T2I_R1"],

        "T2I_R5":
            metrics["T2I_R5"],

        "T2I_R10":
            metrics["T2I_R10"],

        "XLMR_CONFIDENCE":
            mean_cx,

        "XLMR_CONFIDENCE_STD":
            std_cx,

        "INDICBERT_CONFIDENCE":
            mean_ci,

        "INDICBERT_CONFIDENCE_STD":
            std_ci,

        "XLMR_ATTENTION":
            mean_ax,

        "INDICBERT_ATTENTION":
            mean_ai,

        "XLMR_EFFECTIVE_WEIGHT":
            mean_wx,

        "INDICBERT_EFFECTIVE_WEIGHT":
            mean_wi
    }

    all_results.append(result)

    # ========================================================
    # SAVE SEED 42 CHECKPOINT
    # ========================================================

    if seed == 42:

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "seed":
                    seed,

                "xlmr_dim":
                    768,

                "indicbert_dim":
                    768,

                "embed_dim":
                    512,

                "temperature":
                    TEMPERATURE
            },
            CHECKPOINT_FILE
        )

        print(
            "\nSeed 42 checkpoint saved:"
        )

        print(
            CHECKPOINT_FILE
        )


# ============================================================
# SUMMARY FUNCTION
# ============================================================

def mean_std(key):

    values = np.array(
        [
            r[key]
            for r in all_results
        ],
        dtype=np.float32
    )

    return (
        values.mean(),
        values.std()
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 41 SUMMARY")
print("=" * 70)

summary_keys = [

    "I2T_R1",
    "I2T_R5",
    "I2T_R10",

    "T2I_R1",
    "T2I_R5",
    "T2I_R10",

    "XLMR_CONFIDENCE",
    "INDICBERT_CONFIDENCE",

    "XLMR_ATTENTION",
    "INDICBERT_ATTENTION",

    "XLMR_EFFECTIVE_WEIGHT",
    "INDICBERT_EFFECTIVE_WEIGHT"
]

for key in summary_keys:

    mean_value, std_value = mean_std(
        key
    )

    print(
        "{:<28}: {:.4f} ± {:.4f}".format(
            key,
            mean_value,
            std_value
        )
    )


# ============================================================
# SAVE
# ============================================================

np.save(
    OUTPUT_FILE,
    np.array(
        all_results,
        dtype=object
    ),
    allow_pickle=True
)

print("\nResults saved:")
print(OUTPUT_FILE)

print("\nDataset was NOT modified.")

print("\nSTEP 41 COMPLETED")