"""
Pre-trained embedding models (GloVe, Word2Vec) for comparison with autoencoder.
Uses KNN to find nearest neighbors for interpolation visualization.
"""

import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import gensim.downloader as api
from typing import List, Optional
import pickle


class EmbeddingModel:
    """Base class for embedding models."""

    def __init__(self):
        self.embeddings = None
        self.vocab = None
        self.word_to_idx = None
        self.knn = None

    def interpolate_grid(self, word1: str, word2: str, word3: str, grid_size: int = 10):
        """
        Create interpolation grid using KNN lookup.

        Args:
            word1, word2, word3: Three words to interpolate between
            grid_size: Size of the grid (grid_size x grid_size)

        Returns:
            Grid of words (numpy array)
        """
        # Get embeddings for input words
        try:
            emb1 = self.get_embedding(word1)
            emb2 = self.get_embedding(word2)
            emb3 = self.get_embedding(word3)
        except KeyError as e:
            raise ValueError(f"Word not in vocabulary: {e}")

        # Create interpolation grid
        grid = []
        for i in range(grid_size):
            row = []
            t1 = i / (grid_size - 1)  # 0 to 1
            for j in range(grid_size):
                t2 = j / (grid_size - 1)  # 0 to 1

                # Bilinear interpolation between three points
                # word1 at (0,0), word2 at (grid_size-1, 0), word3 at (0, grid_size-1)
                # Interpolate from word1 towards word2 and word3
                interp_emb = emb1 + t1 * (emb2 - emb1) + t2 * (emb3 - emb1)

                # Find nearest neighbor
                nearest_word = self.find_nearest(interp_emb)
                row.append(nearest_word)
            grid.append(row)

        return np.array(grid)

    def get_embedding(self, word: str):
        """Get embedding for a word."""
        raise NotImplementedError

    def find_nearest(self, embedding: np.ndarray, k: int = 1) -> str:
        """Find k nearest neighbors to an embedding."""
        if self.knn is None:
            raise RuntimeError("Model not loaded")

        distances, indices = self.knn.kneighbors([embedding], n_neighbors=k)
        return self.vocab[indices[0][0]]


class GloVeModel(EmbeddingModel):
    """GloVe pre-trained embeddings."""

    def __init__(self, model_name: str = "glove-wiki-gigaword-50"):
        """
        Initialize GloVe model.

        Args:
            model_name: GloVe model name from gensim-data
                Options: glove-wiki-gigaword-50, glove-wiki-gigaword-100,
                         glove-wiki-gigaword-200, glove-wiki-gigaword-300
        """
        super().__init__()
        self.model_name = model_name
        self.cache_path = Path(f"./data/{model_name}.pkl")

    def is_downloaded(self) -> bool:
        """Check if model is already downloaded/cached."""
        return self.cache_path.exists()

    def load(self):
        """Load GloVe embeddings."""
        if not self.is_downloaded():
            raise FileNotFoundError(
                f"GloVe model not found at {self.cache_path}.\n"
                f"Please download it first by running:\n"
                f"  python download_embeddings.py --model glove\n"
                f"Or to download all models:\n"
                f"  python download_embeddings.py"
            )

        print(f"Loading GloVe model: {self.model_name}...")
        print("Loading from cache...")
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)
            self.embeddings = data["embeddings"]
            self.vocab = data["vocab"]
            self.word_to_idx = data["word_to_idx"]

        # Build KNN index
        print("Building KNN index...")
        self.knn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="auto")
        self.knn.fit(self.embeddings)

        print(
            f"Loaded {len(self.vocab)} words with {self.embeddings.shape[1]}-dimensional embeddings"
        )

    def get_embedding(self, word: str):
        """Get embedding for a word."""
        word = word.lower()
        if word not in self.word_to_idx:
            raise KeyError(word)
        idx = self.word_to_idx[word]
        return self.embeddings[idx]


class Word2VecModel(EmbeddingModel):
    """Word2Vec pre-trained embeddings."""

    def __init__(self, model_name: str = "word2vec-google-news-300"):
        """
        Initialize Word2Vec model.

        Args:
            model_name: Word2Vec model name from gensim-data
                Options: word2vec-google-news-300 (default and most common)
        """
        super().__init__()
        self.model_name = model_name
        self.cache_path = Path(f"./data/{model_name}.pkl")

    def is_downloaded(self) -> bool:
        """Check if model is already downloaded/cached."""
        return self.cache_path.exists()

    def load(self):
        """Load Word2Vec embeddings."""
        if not self.is_downloaded():
            raise FileNotFoundError(
                f"Word2Vec model not found at {self.cache_path}.\n"
                f"Please download it first by running:\n"
                f"  python download_embeddings.py --model word2vec\n"
                f"Or to download all models:\n"
                f"  python download_embeddings.py"
            )

        print(f"Loading Word2Vec model: {self.model_name}...")
        print("Loading from cache...")
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)
            self.embeddings = data["embeddings"]
            self.vocab = data["vocab"]
            self.word_to_idx = data["word_to_idx"]

        # Build KNN index
        print("Building KNN index...")
        self.knn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="auto")
        self.knn.fit(self.embeddings)

        print(
            f"Loaded {len(self.vocab)} words with {self.embeddings.shape[1]}-dimensional embeddings"
        )

    def get_embedding(self, word: str):
        """Get embedding for a word."""
        if word not in self.word_to_idx:
            # Try lowercase
            word = word.lower()
            if word not in self.word_to_idx:
                raise KeyError(word)
        idx = self.word_to_idx[word]
        return self.embeddings[idx]


# Global model cache
_model_cache = {}


def get_embedding_model(model_type: str) -> Optional[EmbeddingModel]:
    """
    Get or create an embedding model.

    Args:
        model_type: 'glove' or 'word2vec'

    Returns:
        Loaded embedding model
    """
    if model_type not in _model_cache:
        if model_type == "glove":
            model = GloVeModel(model_name="glove-wiki-gigaword-50")
        elif model_type == "word2vec":
            model = Word2VecModel(model_name="word2vec-google-news-300")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.load()
        _model_cache[model_type] = model

    return _model_cache[model_type]
