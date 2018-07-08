# char-autoencoder

A sequence to sequence autoencoder at character level using tensorflow.

# Requirements

1. tensorflow (1.4 or similar)
2. numpy
3. Python 3.5 or similar

# Usage

1. Download the english word list using the URL provided under `preprocess.py`
2. Run `preprocess()` inside `preprocess.py`
3. Run `train.py`
4. Run `interp_demo.py` to load model and vizualize the latent space