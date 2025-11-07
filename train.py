import torch
import torch.optim as optim
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

from model import Seq2SeqAutoencoder

# Hyperparameters
NUM_UNITS = 128
TIME_STEPS = 10
INPUT_SIZE = 27
UNK_TOKEN = 26

EPOCHS = 10000
BATCH_SIZE = 100
CHECKPOINT_EVERY = 1000  # Save checkpoint every 1000 epochs (reduced from 250)
LEARNING_RATE = 0.001

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
data_path = Path("./data/preprocessed.npy")
if not data_path.exists():
    print(f"Data not found at {data_path}")
    print("Running preprocessing to download and prepare data...")
    from preprocess import preprocess
    preprocess()

data = np.load(data_path)
DATA_SIZE = len(data)
print(f"Loaded {DATA_SIZE} words")

# Remove last dimension and convert to tensor once
data = data.squeeze(-1)  # Shape: (DATA_SIZE, 10)


def get_batch(size):
    """Get random batch of data."""
    idx = np.random.randint(0, DATA_SIZE, size=size)
    batch = data[idx]
    return torch.from_numpy(batch).long().to(device)


# Initialize model
model = Seq2SeqAutoencoder(
    state_size=NUM_UNITS,
    time_steps=TIME_STEPS,
    input_size=INPUT_SIZE,
    unk_token=UNK_TOKEN,
).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Create checkpoint directory
checkpoint_dir = Path("./artifacts")
checkpoint_dir.mkdir(exist_ok=True)

print("Starting training...")
print("-" * 60)

# Training loop with tqdm progress bar
with tqdm(range(1, EPOCHS + 1), desc="Training", unit="epoch") as pbar:
    for epoch in pbar:
        model.train()

        # Get batch
        batch = get_batch(BATCH_SIZE)

        # Forward pass
        predictions, y = model(batch)
        loss = model.compute_loss(predictions, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar with current loss
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Evaluation and logging
        if epoch % CHECKPOINT_EVERY == 0:
            model.eval()
            with torch.no_grad():
                # Get evaluation batch
                eval_batch = get_batch(BATCH_SIZE)
                predictions, y = model(eval_batch)

                # Compute metrics
                loss_val = model.compute_loss(predictions, y)
                accuracy = model.compute_accuracy(predictions, y)

                # Generate text summary
                result_strings = model.generate_text_summary(
                    ["apple", "ball", "goat"], device=device
                )

                # Print results
                tqdm.write(f"\nEpoch {epoch:5d} | Accuracy: {accuracy:.4f} | Loss: {loss_val:.4f}")
                tqdm.write("Interpolation grid:")
                for row in result_strings:
                    tqdm.write(" | ".join(f"{s:>10s}" for s in row))
                tqdm.write("-" * 60)

                # Update progress bar with evaluation metrics
                pbar.set_postfix({"loss": f"{loss_val.item():.4f}", "acc": f"{accuracy.item():.4f}"})

            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"model_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_val.item(),
                    "accuracy": accuracy.item(),
                },
                checkpoint_path,
            )

# Save final model
final_model_path = checkpoint_dir / "model_final.pt"
torch.save(
    {
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    final_model_path,
)

print("\nTraining complete!")
print(f"Final model saved to {final_model_path}")
print("\a")  # Bell sound
