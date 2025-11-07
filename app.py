"""
Flask Web Server for Character Autoencoder Interpolation Demo
Allows interactive word interpolation through a web interface.
Supports: Autoencoder, GloVe, Word2Vec

Interpolation Method (Autoencoder):
    Uses parallelogram interpolation to create a 2D manifold in latent space.
    Given three words A, B, C:
        - Corner [0,0]: A (first word)
        - Corner [max,0]: B (second word)
        - Corner [0,max]: C (third word)
        - Corner [max,max]: B + C - A (computed fourth corner)
    Formula: point[i,j] = A + (i/max)*(B-A) + (j/max)*(C-A)

    The fourth corner is automatically computed from the other three corners,
    not directly encoded from an input word. This creates a bilinear interpolation
    forming a parallelogram in the latent space.
"""

from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from pathlib import Path

from model import Seq2SeqAutoencoder
from embedding_models import get_embedding_model

app = Flask(__name__)

# Configuration
NUM_UNITS = 128
TIME_STEPS = 10
INPUT_SIZE = 27
UNK_TOKEN = 26

# Global model variable
model = None
device = None


def load_model():
    """Load the trained model."""
    global model, device

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
    checkpoint_path = Path("./artifacts/model_final.pt")
    if not checkpoint_path.exists():
        # Try to find the latest checkpoint
        checkpoint_dir = Path("./artifacts")
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


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/interpolate", methods=["POST"])
def interpolate():
    """Handle interpolation requests."""
    try:
        data = request.get_json()
        word1 = data.get("word1", "").strip().lower()
        word2 = data.get("word2", "").strip().lower()
        word3 = data.get("word3", "").strip().lower()
        model_type = data.get("model_type", "autoencoder").strip().lower()

        # Validate inputs
        if not all([word1, word2, word3]):
            return (
                jsonify({"success": False, "error": "All three words are required."}),
                400,
            )

        # Check for invalid characters (only a-z allowed)
        for word in [word1, word2, word3]:
            if not word.isalpha():
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": f"Invalid characters in '{word}'. Only letters (a-z) are allowed.",
                        }
                    ),
                    400,
                )

        # Perform interpolation based on model type
        if model_type == "autoencoder":
            # Length check only for autoencoder
            for word in [word1, word2, word3]:
                if len(word) > 10:
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": f"Word '{word}' is too long for autoencoder. Maximum length is 10 characters.",
                            }
                        ),
                        400,
                    )

            with torch.no_grad():
                result_strings = model.generate_text_summary(
                    [word1, word2, word3], device=device
                )
            grid = result_strings.tolist()

        elif model_type in ["glove", "word2vec"]:
            try:
                embedding_model = get_embedding_model(model_type)
                result_strings = embedding_model.interpolate_grid(
                    word1, word2, word3, grid_size=10
                )
                grid = result_strings.tolist()
            except ValueError as e:
                return jsonify({"success": False, "error": str(e)}), 400

        else:
            return (
                jsonify(
                    {"success": False, "error": f"Unknown model type: {model_type}"}
                ),
                400,
            )

        return jsonify(
            {
                "success": True,
                "grid": grid,
                "words": [word1, word2, word3],
                "model_type": model_type,
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return (
            jsonify(
                {"success": False, "error": f"Error during interpolation: {str(e)}"}
            ),
            500,
        )


@app.route("/examples")
def examples():
    """
    Get example word triplets that demonstrate parallelogram interpolation.

    These examples follow patterns where B+C-A produces intuitive results:
    - [A, B, C] where B applies a transformation to A, and the fourth corner
      should apply the same transformation to C.

    Examples:
        ["man", "woman", "king"] → fourth corner ≈ "queen" (classic word2vec analogy)
        ["cat", "cats", "dog"] → fourth corner ≈ "dogs" (plural transfer)
        ["run", "running", "walk"] → fourth corner ≈ "walking" (gerund transfer)
    """
    example_triplets = [
        # Classic word2vec analogy: man→woman, king→queen (gender transfer)
        ["man", "woman", "king"],
        # Plural transformation: cat→cats, dog→dogs
        ["cat", "cats", "dog"],
        # Gerund transformation: run→running, walk→walking
        ["run", "running", "walk"],
        # Comparative adjectives: happy→happier, sad→sadder
        ["happy", "happier", "sad"],
        # Comparative size: small→smaller, big→bigger
        ["small", "smaller", "big"],
        # Comparative speed: fast→faster, slow→slower
        ["fast", "faster", "slow"],
        # Temperature comparatives: hot→hotter, cold→colder
        ["hot", "hotter", "cold"],
        # Verb gerunds: play→playing, jump→jumping
        ["play", "playing", "jump"],
        # Adverb formation: quick→quickly, slow→slowly
        ["quick", "quickly", "slow"],
        # Past tense: walk→walked, jump→jumped
        ["walk", "walked", "jump"],
    ]
    return jsonify({"examples": example_triplets})


if __name__ == "__main__":
    print("Loading model...")
    load_model()
    print("\nStarting Flask server...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    app.run(debug=True, host="0.0.0.0", port=5000)
