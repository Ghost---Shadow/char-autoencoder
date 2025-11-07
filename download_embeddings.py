"""
Download pre-trained embedding models (GloVe and Word2Vec).
This script pre-downloads and caches the models for faster usage.
"""

import argparse
from pathlib import Path
import pickle
import numpy as np
import gensim.downloader as api


def download_glove(model_name: str = "glove-wiki-gigaword-50"):
    """
    Download and cache GloVe embeddings.

    Args:
        model_name: GloVe model name from gensim-data
            Options: glove-wiki-gigaword-50 (~66MB),
                     glove-wiki-gigaword-100 (~128MB),
                     glove-wiki-gigaword-200 (~252MB),
                     glove-wiki-gigaword-300 (~376MB)
    """
    cache_path = Path(f"./data/{model_name}.pkl")

    if cache_path.exists():
        print(f"✓ GloVe model already cached at {cache_path}")
        return

    print(f"Downloading GloVe model: {model_name}...")
    print("This may take a few minutes depending on your connection...")

    # Download from gensim
    model = api.load(model_name)

    # Convert to our format
    print("Converting to internal format...")
    vocab = list(model.key_to_index.keys())
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    embeddings = np.array([model[word] for word in vocab])

    # Cache for faster loading next time
    cache_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"Saving cache to {cache_path}...")
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "embeddings": embeddings,
                "vocab": vocab,
                "word_to_idx": word_to_idx,
            },
            f,
        )

    print(f"✓ Successfully cached GloVe model!")
    print(f"  - {len(vocab)} words")
    print(f"  - {embeddings.shape[1]}-dimensional embeddings")
    print(f"  - Saved to {cache_path}")


def download_word2vec(model_name: str = "word2vec-google-news-300"):
    """
    Download and cache Word2Vec embeddings.

    Args:
        model_name: Word2Vec model name from gensim-data
            Options: word2vec-google-news-300 (~1.6GB)
    """
    cache_path = Path(f"./data/{model_name}.pkl")

    if cache_path.exists():
        print(f"✓ Word2Vec model already cached at {cache_path}")
        return

    print(f"Downloading Word2Vec model: {model_name}...")
    print("WARNING: This is a large file (~1.6GB) and will take significant time!")
    print("Please be patient...")

    # Download from gensim
    model = api.load(model_name)

    # Convert to our format (only use most common words to save memory)
    print("Converting to internal format (using top 100k words)...")
    vocab = list(model.key_to_index.keys())[:100000]  # Limit to 100k words
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    embeddings = np.array([model[word] for word in vocab])

    # Cache for faster loading next time
    cache_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"Saving cache to {cache_path}...")
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "embeddings": embeddings,
                "vocab": vocab,
                "word_to_idx": word_to_idx,
            },
            f,
        )

    print(f"✓ Successfully cached Word2Vec model!")
    print(f"  - {len(vocab)} words")
    print(f"  - {embeddings.shape[1]}-dimensional embeddings")
    print(f"  - Saved to {cache_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and cache pre-trained embedding models"
    )
    parser.add_argument(
        "--model",
        choices=["glove", "word2vec", "all"],
        default="all",
        help="Which model to download (default: all)",
    )
    parser.add_argument(
        "--glove-size",
        choices=["50", "100", "200", "300"],
        default="50",
        help="GloVe embedding dimension (default: 50)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Embedding Model Downloader")
    print("=" * 80)
    print()

    if args.model in ["glove", "all"]:
        print("Downloading GloVe embeddings...")
        print("-" * 80)
        glove_model = f"glove-wiki-gigaword-{args.glove_size}"
        download_glove(glove_model)
        print()

    if args.model in ["word2vec", "all"]:
        print("Downloading Word2Vec embeddings...")
        print("-" * 80)
        download_word2vec("word2vec-google-news-300")
        print()

    print("=" * 80)
    print("Download complete!")
    print("=" * 80)
    print()
    print("You can now use these models in:")
    print("  - Web interface: python app.py")
    print("  - Command line: python interp_demo.py")


if __name__ == "__main__":
    main()
