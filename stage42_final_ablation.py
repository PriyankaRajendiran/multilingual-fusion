import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# STEP 42
# FINAL FAIR ABLATION COMPARISON
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
    "stage42_final_ablation_results.npy"
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

class AblationModel(nn.Module):

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

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        # Interaction confidence
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

        # Same fusion layer for fusion-based methods
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

        return torch.softmax(
            self.attention(combined),
            dim=-1
        )

    def get_confidence(self, x, i, image):

        x_interaction = torch.cat(
            [
                x,
                image,
                x * image,
                torch.abs(x - image)
            ],
            dim=-1
        )

        i_interaction = torch.cat(
            [
                i,
                image,
                i * image,
                torch.abs(i - image)
            ],
            dim=-1
        )

        cx = self.xlmr_confidence(
            x_interaction
        ).squeeze(1)

        ci = self.indic_confidence(
            i_interaction
        ).squeeze(1)

        return cx, ci

    def fuse(self, x, i, weights):

        wx = weights[:, 0:1]
        wi = weights[:, 1:2]

        combined = torch.cat(
            [
                wx * x,
                wi * i
            ],
            dim=-1
        )

        fused = self.fusion(combined)

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

def retrieval_metrics(image_emb, text_emb):

    similarity = torch.matmul(
        image_emb,
        text_emb.T
    )

    n = similarity.size(0)

    i2t = {1: 0, 5: 0, 10: 0}
    t2i = {1: 0, 5: 0, 10: 0}

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
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 42")
print("FINAL FAIR ABLATION COMPARISON")
print("=" * 70)

print("Device:", DEVICE)

image_np = np.load(
    IMAGE_FILE
).astype(np.float32)

xlmr_np = np.load(
    XLMR_FILE
).astype(np.float32)

indic_np = np.load(
    INDIC_FILE
).astype(np.float32
)

assert image_np.shape == (100, 512)
assert xlmr_np.shape == (100, 768)
assert indic_np.shape == (100, 768)


# ============================================================
# NORMALIZATION
# ============================================================

image_np = image_np / (
    np.linalg.norm(
        image_np,
        axis=1,
        keepdims=True
    ) + 1e-8
)

xlmr_np = xlmr_np / (
    np.linalg.norm(
        xlmr_np,
        axis=1,
        keepdims=True
    ) + 1e-8
)

indic_np = indic_np / (
    np.linalg.norm(
        indic_np,
        axis=1,
        keepdims=True
    ) + 1e-8
)


# ============================================================
# METHODS
# ============================================================

METHODS = [
    "XLM-R Only",
    "IndicBERT Only",
    "Average Fusion",
    "Attention Fusion",
    "Interaction Confidence Fusion"
]


all_results = []


# ============================================================
# FIVE SEEDS
# ============================================================

for seed in SEEDS:

    print("\n" + "=" * 70)
    print("SEED:", seed)
    print("=" * 70)

    set_seed(seed)

    # --------------------------------------------------------
    # SAME SPLIT FOR ALL METHODS
    # --------------------------------------------------------

    indices = np.arange(100)

    np.random.shuffle(indices)

    train_size = int(
        TRAIN_RATIO * 100
    )

    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    print(
        "Train:",
        len(train_idx),
        "| Test:",
        len(test_idx)
    )

    # --------------------------------------------------------
    # TENSORS
    # --------------------------------------------------------

    image = torch.tensor(
        image_np,
        dtype=torch.float32,
        device=DEVICE
    )

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

    train_indices = torch.tensor(
        train_idx,
        dtype=torch.long,
        device=DEVICE
    )

    test_indices = torch.tensor(
        test_idx,
        dtype=torch.long,
        device=DEVICE
    )

    # ========================================================
    # TRAIN FIVE METHODS
    # ========================================================

    for method in METHODS:

        print("\n----------------------------------------")
        print(method)
        print("----------------------------------------")

        set_seed(seed)

        model = AblationModel().to(
            DEVICE
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )

        model.train()

        for epoch in range(EPOCHS):

            optimizer.zero_grad()

            x, i = model.project(
                xlmr[train_indices],
                indic[train_indices]
            )

            target_image = image[
                train_indices
            ]

            # ------------------------------------------------
            # METHOD 1: XLM-R ONLY
            # ------------------------------------------------

            if method == "XLM-R Only":

                fused = x

            # ------------------------------------------------
            # METHOD 2: INDICBERT ONLY
            # ------------------------------------------------

            elif method == "IndicBERT Only":

                fused = i

            # ------------------------------------------------
            # METHOD 3: AVERAGE FUSION
            # ------------------------------------------------

            elif method == "Average Fusion":

                weights = torch.tensor(
                    [
                        [0.5, 0.5]
                    ],
                    dtype=torch.float32,
                    device=DEVICE
                ).repeat(
                    x.size(0),
                    1
                )

                fused = model.fuse(
                    x,
                    i,
                    weights
                )

            # ------------------------------------------------
            # METHOD 4: ATTENTION
            # ------------------------------------------------

            elif method == "Attention Fusion":

                attention = model.get_attention(
                    x,
                    i
                )

                fused = model.fuse(
                    x,
                    i,
                    attention
                )

            # ------------------------------------------------
            # METHOD 5: INTERACTION CONFIDENCE
            # ------------------------------------------------

            else:

                attention = model.get_attention(
                    x,
                    i
                )

                cx, ci = model.get_confidence(
                    x,
                    i,
                    target_image
                )

                wx = (
                    attention[:, 0]
                    * cx
                )

                wi = (
                    attention[:, 1]
                    * ci
                )

                total = (
                    wx + wi + 1e-8
                )

                wx = wx / total
                wi = wi / total

                weights = torch.stack(
                    [wx, wi],
                    dim=1
                )

                fused = model.fuse(
                    x,
                    i,
                    weights
                )

                # --------------------------------------------
                # Similarity-guided confidence target
                # --------------------------------------------

                sim_x = torch.sum(
                    x * target_image,
                    dim=-1
                )

                sim_i = torch.sum(
                    i * target_image,
                    dim=-1
                )

                target_x = (
                    sim_x.detach() + 1.0
                ) / 2.0

                target_i = (
                    sim_i.detach() + 1.0
                ) / 2.0

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

            # ------------------------------------------------
            # RETRIEVAL LOSS
            # ------------------------------------------------

            retrieval_loss = contrastive_loss(
                target_image,
                fused,
                TEMPERATURE
            )

            if method == "Interaction Confidence Fusion":

                loss = (
                    retrieval_loss
                    +
                    0.5 * confidence_loss
                )

            else:

                loss = retrieval_loss

            loss.backward()

            optimizer.step()

        # ====================================================
        # TEST
        # ====================================================

        model.eval()

        with torch.no_grad():

            x_test, i_test = model.project(
                xlmr[test_indices],
                indic[test_indices]
            )

            image_test = image[
                test_indices
            ]

            if method == "XLM-R Only":

                fused_test = x_test

            elif method == "IndicBERT Only":

                fused_test = i_test

            elif method == "Average Fusion":

                weights = torch.tensor(
                    [
                        [0.5, 0.5]
                    ],
                    dtype=torch.float32,
                    device=DEVICE
                ).repeat(
                    x_test.size(0),
                    1
                )

                fused_test = model.fuse(
                    x_test,
                    i_test,
                    weights
                )

            elif method == "Attention Fusion":

                attention = model.get_attention(
                    x_test,
                    i_test
                )

                fused_test = model.fuse(
                    x_test,
                    i_test,
                    attention
                )

            else:

                attention = model.get_attention(
                    x_test,
                    i_test
                )

                cx, ci = model.get_confidence(
                    x_test,
                    i_test,
                    image_test
                )

                wx = (
                    attention[:, 0]
                    * cx
                )

                wi = (
                    attention[:, 1]
                    * ci
                )

                total = (
                    wx + wi + 1e-8
                )

                wx = wx / total
                wi = wi / total

                weights = torch.stack(
                    [wx, wi],
                    dim=1
                )

                fused_test = model.fuse(
                    x_test,
                    i_test,
                    weights
                )

                mean_cx = cx.mean().item()
                mean_ci = ci.mean().item()

                mean_wx = wx.mean().item()
                mean_wi = wi.mean().item()

            metrics = retrieval_metrics(
                image_test,
                fused_test
            )

        print(
            "I2T R@1 : {:.2f}%".format(
                metrics["I2T_R1"]
            )
        )

        print(
            "I2T R@5 : {:.2f}%".format(
                metrics["I2T_R5"]
            )
        )

        print(
            "I2T R@10: {:.2f}%".format(
                metrics["I2T_R10"]
            )
        )

        print(
            "T2I R@1 : {:.2f}%".format(
                metrics["T2I_R1"]
            )
        )

        print(
            "T2I R@5 : {:.2f}%".format(
                metrics["T2I_R5"]
            )
        )

        print(
            "T2I R@10: {:.2f}%".format(
                metrics["T2I_R10"]
            )
        )

        if method == "Interaction Confidence Fusion":

            print(
                "XLM-R confidence     : {:.4f}".format(
                    mean_cx
                )
            )

            print(
                "IndicBERT confidence : {:.4f}".format(
                    mean_ci
                )
            )

            print(
                "XLM-R effective      : {:.4f}".format(
                    mean_wx
                )
            )

            print(
                "IndicBERT effective  : {:.4f}".format(
                    mean_wi
                )
            )

        result = {
            "seed": seed,
            "method": method,
            "I2T_R1": metrics["I2T_R1"],
            "I2T_R5": metrics["I2T_R5"],
            "I2T_R10": metrics["I2T_R10"],
            "T2I_R1": metrics["T2I_R1"],
            "T2I_R5": metrics["T2I_R5"],
            "T2I_R10": metrics["T2I_R10"]
        }

        all_results.append(
            result
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 42 FINAL FAIR ABLATION SUMMARY")
print("=" * 70)

for method in METHODS:

    method_results = [
        r for r in all_results
        if r["method"] == method
    ]

    print("\n" + method)

    for key in [
        "I2T_R1",
        "I2T_R5",
        "I2T_R10",
        "T2I_R1",
        "T2I_R5",
        "T2I_R10"
    ]:

        values = np.array(
            [
                r[key]
                for r in method_results
            ],
            dtype=np.float32
        )

        print(
            "{:<10}: {:.2f} ± {:.2f}".format(
                key,
                values.mean(),
                values.std()
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

print("\nSTEP 42 COMPLETED")