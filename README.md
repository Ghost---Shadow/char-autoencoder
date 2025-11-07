# Character-Level Sequence-to-Sequence Autoencoder (PyTorch)

A character-level sequence-to-sequence autoencoder implemented in PyTorch. The model learns to encode short words (up to 10 characters) into a latent representation and decode them back. It can also interpolate between words in the latent space to generate intermediate representations.

## Architecture

- **Encoder**: Custom LSTM that processes input sequences character-by-character
- **Latent Space**: 128-dimensional hidden state (h) and cell state (c)
- **Decoder**: Custom LSTM that reconstructs the reversed input sequence
- **Custom LSTM**: Implements LSTM from scratch with explicit gate calculations (input, forget, output, and candidate gates)

The model learns to reconstruct reversed input sequences, which forces it to learn meaningful representations of the entire sequence.

## Requirements

- Python 3.7+
- PyTorch 1.7+ (with CUDA support optional)
- NumPy

Install dependencies:
```bash
pip install torch numpy
```

## Project Structure

```
.
├── lstm_cell.py       # Custom LSTM implementation
├── model.py           # Seq2SeqAutoencoder model
├── preprocess.py      # Data preprocessing utilities
├── train.py           # Training script
├── interp_demo.py     # Interpolation demo script
├── README.md          # This file
└── legacy_tensorflow/ # Original TensorFlow 1.x implementation
```

## Usage

### 1. Preprocess Data

Download the English word list from [SIL Linguistics](http://www-01.sil.org/linguistics/wordlists/english/) and update the `INPUT_FILE` path in `preprocess.py`.

Then run preprocessing:
```bash
python preprocess.py
```

This will create `preprocessed.npy` containing all words with length ≤ 10 characters.

### 2. Train the Model

```bash
python train.py
```

Training configuration:
- **Epochs**: 10,000 (default)
- **Batch size**: 100
- **Learning rate**: 0.001 (Adam optimizer)
- **State size**: 128
- **Sequence length**: 10 characters

The script will:
- Print training progress every 250 epochs
- Show interpolation examples between "apple", "ball", and "goat"
- Save checkpoints to `./checkpoints/`
- Display accuracy and loss metrics

**GPU Support**: The script automatically uses CUDA if available.

### 3. Visualize Latent Space

After training, run the interpolation demo:
```bash
python interp_demo.py
```

This will load the trained model and demonstrate latent space interpolation for various word triplets, showing smooth transitions between words in the learned representation space.

## Model Details

### Input Representation
- Characters are one-hot encoded (a=0, b=1, ..., z=25, space=26)
- Sequences are padded to 10 characters with space tokens at the beginning
- Input shape: `(batch_size, 10)`

### Training
- **Loss**: Cross-entropy between predictions and reversed input
- **Optimizer**: Adam with default learning rate 0.001
- **Target**: Reversed input sequence (forces the model to process the entire sequence)

### Latent Space Interpolation
Given three words A, B, C, the model:
1. Encodes each word to latent vectors (h, c)
2. Creates a 2D interpolation grid between the three points
3. Decodes each point in the grid to generate interpolated words

This allows visualization of smooth transitions in the learned latent space.

## Differences from TensorFlow Version

This PyTorch implementation maintains the same architecture and functionality as the original TensorFlow version, with the following changes:

- **Framework**: Migrated from TensorFlow 1.x to PyTorch
- **Session management**: Removed TensorFlow sessions in favor of PyTorch's imperative style
- **Checkpointing**: Uses PyTorch's native checkpoint format (`.pt` files)
- **TensorBoard**: Removed TensorBoard logging (can be re-added using PyTorch's SummaryWriter)
- **Code structure**: More modular and Pythonic structure using `nn.Module`

## Legacy TensorFlow Version

The original TensorFlow 1.x implementation is preserved in `./legacy_tensorflow/` for reference.

## Examples

After training, the model can interpolate between words like:

```
apple → ball → goat
```

Creating intermediate representations that smoothly transition between the input words in the latent space.

## License

MIT License

## References

- Original implementation: TensorFlow 1.x character autoencoder
- Word list source: [SIL International Linguistics](http://www-01.sil.org/linguistics/wordlists/english/)
