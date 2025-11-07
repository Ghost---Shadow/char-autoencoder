import torch
import numpy as np
from pathlib import Path

from model import Seq2SeqAutoencoder

# Configuration
NUM_UNITS = 128
TIME_STEPS = 10
INPUT_SIZE = 27
UNK_TOKEN = 26

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
model = Seq2SeqAutoencoder(
    state_size=NUM_UNITS,
    time_steps=TIME_STEPS,
    input_size=INPUT_SIZE,
    unk_token=UNK_TOKEN,
).to(device)

# Load checkpoint
checkpoint_path = Path("./checkpoints/model_final.pt")
if not checkpoint_path.exists():
    # Try to find the latest checkpoint
    checkpoint_dir = Path("./checkpoints")
    checkpoints = sorted(checkpoint_dir.glob("model_epoch_*.pt"))
    if checkpoints:
        checkpoint_path = checkpoints[-1]
    else:
        raise FileNotFoundError(
            "No checkpoint found. Please train the model first using train.py"
        )

print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully!")
print("=" * 80)

# Test strings for interpolation
test_cases = [
    ["apple", "ball", "goat"],
    ["identity", "identity", "identity"],
    ["baseball", "ball", "base"],
    ["cabbage", "cab", "cabin"],
    ["abc", "def", "ghi"],
    ["toooot", "toooooooot", "tot"],
    ["ball ball", "cat cat", "dog dog"],
]

# Generate interpolations
with torch.no_grad():
    for i, strings in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {strings}")
        print("-" * 80)

        try:
            result_strings = model.generate_text_summary(strings, device=device)

            # Print the interpolation grid
            print("Interpolation grid (rows: apple→ball, cols: apple→goat):")
            print()
            for row_idx, row in enumerate(result_strings):
                print(f"Row {row_idx}: ", end="")
                print(" | ".join(f"{s:>12s}" for s in row))

            print()

        except Exception as e:
            print(f"Error processing test case: {e}")
            continue

print("=" * 80)
print("Demo complete!")
