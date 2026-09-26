import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "embeddings",
    "stage28_80_20_alignment_model.pt"
)

print("=" * 70)
print("STEP 42A: CHECK STAGE-28 CHECKPOINT")
print("=" * 70)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

print("\nCheckpoint type:")
print(type(checkpoint))

print("\nTop-level keys:")

if isinstance(checkpoint, dict):
    for key in checkpoint.keys():
        print("  ", key)

print("\n" + "=" * 70)
print("PARAMETER KEYS")
print("=" * 70)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


for key, value in state_dict.items():

    if torch.is_tensor(value):

        print(
            f"{key:50s} shape={tuple(value.shape)}"
        )

print("\n" + "=" * 70)
print("TOTAL PARAMETERS")
print("=" * 70)

count = 0

for key, value in state_dict.items():

    if torch.is_tensor(value):
        count += 1

print("Tensor parameters:", count)

print("\nSTEP 42A COMPLETED")
print("=" * 70)