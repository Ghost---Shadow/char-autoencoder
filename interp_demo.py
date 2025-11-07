import torch
import numpy as np
from pathlib import Path
import argparse

from model import Seq2SeqAutoencoder
from embedding_models import get_embedding_model


# Dummy context manager for non-torch models
class DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Demonstrate latent space interpolation with different models"
)
parser.add_argument(
    "--model",
    choices=["autoencoder", "glove", "word2vec", "all"],
    default="autoencoder",
    help="Which model to use for interpolation (default: autoencoder)",
)
args = parser.parse_args()

print("=" * 80)
print("Latent Space Interpolation Demo")
print("=" * 80)
print()
print("Interpolation Method:")
print(
    "  The model uses parallelogram interpolation to create a 2D manifold in latent space."
)
print("  Given 3 words A, B, C:")
print("    - Corner [0,0]: A (first word)")
print("    - Corner [max,0]: B (second word)")
print("    - Corner [0,max]: C (third word)")
print("    - Corner [max,max]: B + C - A (computed fourth corner)")
print("  Formula: point[i,j] = A + (i/max)*(B-A) + (j/max)*(C-A)")
print("=" * 80)
print()

# Configuration
NUM_UNITS = 128
TIME_STEPS = 10
INPUT_SIZE = 27
UNK_TOKEN = 26

# Store models
models = {}
model_names = []

# Determine which models to load
if args.model == "all":
    model_names = ["autoencoder", "glove", "word2vec"]
else:
    model_names = [args.model]

# Load autoencoder if needed
if "autoencoder" in model_names:
    print("Loading Character Autoencoder...")
    print("-" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    autoencoder = Seq2SeqAutoencoder(
        state_size=NUM_UNITS,
        time_steps=TIME_STEPS,
        input_size=INPUT_SIZE,
        unk_token=UNK_TOKEN,
    ).to(device)

    # Load checkpoint
    checkpoint_path = Path("./artifacts/model_final.pt")
    if not checkpoint_path.exists():
        # Try to find the latest checkpoint
        checkpoint_dir = Path("./artifacts")
        checkpoints = sorted(checkpoint_dir.glob("model_epoch_*.pt"))
        if checkpoints:
            checkpoint_path = checkpoints[-1]
        else:
            print("⚠ No checkpoint found. Please train the model first using train.py")
            print("Skipping autoencoder...")
            model_names.remove("autoencoder")
            autoencoder = None

    if autoencoder is not None:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        autoencoder.load_state_dict(checkpoint["model_state_dict"])
        autoencoder.eval()
        print("✓ Autoencoder loaded successfully!")
        models["autoencoder"] = (autoencoder, device)
    print()

# Load GloVe if needed
if "glove" in model_names:
    print("Loading GloVe Embeddings...")
    print("-" * 80)
    try:
        glove = get_embedding_model("glove")
        print("✓ GloVe loaded successfully!")
        models["glove"] = (glove, None)
    except FileNotFoundError as e:
        print(f"⚠ {e}")
        print("Skipping GloVe...")
        model_names.remove("glove")
    print()

# Load Word2Vec if needed
if "word2vec" in model_names:
    print("Loading Word2Vec Embeddings...")
    print("-" * 80)
    try:
        word2vec = get_embedding_model("word2vec")
        print("✓ Word2Vec loaded successfully!")
        models["word2vec"] = (word2vec, None)
    except FileNotFoundError as e:
        print(f"⚠ {e}")
        print("Skipping Word2Vec...")
        model_names.remove("word2vec")
    print()

if not models:
    print("No models available. Please train the autoencoder or download embeddings.")
    print("  Train autoencoder: python train.py")
    print("  Download embeddings: python download_embeddings.py")
    exit(1)

print("=" * 80)

# Test strings for interpolation
# These examples demonstrate parallelogram interpolation where B+C-A should produce
# intuitive results by transferring transformations between words.
test_cases = [
    # Classic word2vec analogy: man→woman, king→? (expects "queen" - gender transfer)
    ["man", "woman", "king"],
    # Plural transformation: cat→cats, dog→? (expects "dogs")
    ["cat", "cats", "dog"],
    # Gerund transformation: run→running, walk→? (expects "walking")
    ["run", "running", "walk"],
    # Comparative adjectives: happy→happier, sad→? (expects "sadder")
    ["happy", "happier", "sad"],
    # Comparative size: small→smaller, big→? (expects "bigger")
    ["small", "smaller", "big"],
    # Comparative speed: fast→faster, slow→? (expects "slower")
    ["fast", "faster", "slow"],
    # Past tense: walk→walked, jump→? (expects "jumped")
    ["walk", "walked", "jump"],
]

# Generate interpolations for each model
for model_name in model_names:
    model_obj, device = models[model_name]

    print()
    print("=" * 80)
    print(f"Model: {model_name.upper()}")
    print("=" * 80)

    with torch.no_grad() if model_name == "autoencoder" else DummyContext():
        for i, strings in enumerate(test_cases, 1):
            print(f"\nTest case {i}: {strings}")
            print("-" * 80)

            try:
                if model_name == "autoencoder":
                    result_strings = model_obj.generate_text_summary(
                        strings, device=device
                    )
                    # Compute parallelogram error (wobbliness)
                    error_stats = model_obj.compute_parallelogram_error(
                        strings, device=device
                    )
                else:
                    # GloVe or Word2Vec
                    result_strings = model_obj.interpolate_grid(
                        strings[0], strings[1], strings[2], grid_size=10
                    )
                    # Compute parallelogram error (wobbliness)
                    error_stats = model_obj.compute_parallelogram_error(
                        strings[0], strings[1], strings[2], grid_size=10
                    )

                # Print the interpolation grid
                print(f"Interpolation grid (2D manifold):")
                print(
                    f"  Corners: [{strings[0]}] at [0,0], [{strings[1]}] at [9,0], [{strings[2]}] at [0,9]"
                )
                print(
                    f"  Fourth corner [9,9] = {strings[1]} + {strings[2]} - {strings[0]} (computed)"
                )
                print()

                # Print parallelogram error statistics (Wobbly Line Hypothesis)
                print(f"📊 Parallelogram Error (Wobbliness):")
                print(f"  Mean L2 distance:  {error_stats['mean_error']:.4f}")
                print(f"  Max L2 distance:   {error_stats['max_error']:.4f}")
                print(f"  Min L2 distance:   {error_stats['min_error']:.4f}")
                print(f"  Std L2 distance:   {error_stats['std_error']:.4f}")
                print()
                print(
                    "  (This measures how much the decoded words deviate from the ideal parallelogram)"
                )
                print()

                for row_idx, row in enumerate(result_strings):
                    print(f"Row {row_idx}: ", end="")
                    print(" | ".join(f"{s:>12s}" for s in row))

                print()

            except Exception as e:
                print(f"Error processing test case: {e}")
                import traceback

                traceback.print_exc()
                continue

print()
print("=" * 80)
print("Demo complete!")
print("=" * 80)
