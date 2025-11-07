# Character-Level Sequence-to-Sequence Autoencoder (PyTorch)

A character-level sequence-to-sequence autoencoder implemented in PyTorch. The model learns to encode short words (up to 10 characters) into a latent representation and decode them back. It can also interpolate between words in the latent space to generate intermediate representations.

## Architecture

- **Encoder**: Custom LSTM that processes input sequences character-by-character
- **Latent Space**: 128-dimensional hidden state (h) and cell state (c)
- **Decoder**: Custom LSTM that reconstructs the reversed input sequence
- **Custom LSTM**: Implements LSTM from scratch with explicit gate calculations (input, forget, output, and candidate gates)

The model learns to reconstruct reversed input sequences, which forces it to learn meaningful representations of the entire sequence.

## Features

✨ **Automatic Data Download**: Downloads English word list automatically with multiple fallback sources
🚀 **GPU Support**: Automatic CUDA detection and usage for faster training
📊 **Progress Tracking**: Beautiful tqdm progress bars with real-time metrics
💾 **Smart Checkpointing**: Saves model checkpoints every 250 epochs to `./artifacts/`
🎨 **Latent Space Interpolation**: Visualize smooth transitions between words

## Requirements

```bash
pip install -r requirements.txt
```

## Quick Start

Run the complete pipeline with one command:
```bash
bash run_flow.sh
```

## Manual Step-by-Step

### 1. Download & Preprocess Data

The pipeline automatically downloads word lists from multiple sources with fallback support:
```bash
python download_data.py  # Downloads to ./data/wordsEn.txt
python preprocess.py     # Creates ./data/preprocessed.npy
```

**Note**: Data is downloaded automatically during training if not present. The script tries multiple sources:
- Original SIL Linguistics wordlist
- GitHub english-words repository
- Google 10000 common words
- Fallback to comprehensive built-in wordlist

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
- Show a beautiful tqdm progress bar with real-time metrics
- Print detailed results every 1000 epochs with interpolation examples
- Save checkpoints to `./artifacts/model_epoch_XXX.pt` every 1000 epochs
- Display accuracy and loss metrics
- Save final model to `./artifacts/model_final.pt`

**GPU Support**: The script automatically detects and uses CUDA if available.

### 3. Visualize Latent Space

You have two options for visualizing interpolations:

**Option A: Web Interface (Recommended)**
```bash
python app.py
```
Then open your browser to: **http://localhost:5000**

Features:
- 🎨 Beautiful interactive web interface
- ⌨️ Type your own custom words
- 📊 Visual grid display of interpolations
- 💡 Pre-loaded example word triplets
- 🚀 Real-time results

**Option B: Command Line**
```bash
python interp_demo.py
```

This loads the trained model and demonstrates latent space interpolation for various word triplets in the terminal.

**Example Output**:
```
Test case 4: ['cabbage', 'cab', 'cabin']
--------------------------------------------------------------------------------
Interpolation grid (rows: apple→ball, cols: apple→goat):

Row 0:      cabbage |      cabbage |      ccbbage |       cabbge |       cabbge |       cabbin |       cabbin |       ccabin |        cabin |        cabin
Row 1:      cabbage |       cbbbge |       cabbge |       cabbge |       cabbgn |       ccabin |        cabin |        cabin |        cabin |        cabin
Row 2:       cabbge |       cabbge |       cabbge |        abbge |        cabin |        cabin |        cabin |        cabin |        dabin |         abin
Row 3:       cabbge |        abbge |        cabge |        cabgl |        caban |        cabin |         abin |         abin |         abin |         cbin
Row 4:        jbbge |        cabge |        cabae |        dabal |         abal |         abal |         abin |         cbin |         cbin |          bin
Row 5:        dabge |         abae |         abab |         abab |         abab |         cbab |          bab |          bad |          cad |          cid
Row 6:         abab |         abab |         abab |         dbab |          bab |          bab |          cab |          cab |          cad |          cad
Row 7:         abab |         dbab |          bab |          cab |          cab |          cab |          cab |          cab |          cad |          dad
Row 8:          cab |          cab |          cab |          cab |          cab |          cab |          dab |          dab |          dad |           ad
Row 9:          cab |          cab |          cab |          cab |          dab |           ab |           ab |           ab |           ab |           ad
```

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

## Training Tips

- **Early Stopping**: You can interrupt training with Ctrl+C - checkpoints are saved every 250 epochs
- **Resume Training**: Load a checkpoint and continue training from where you left off
- **GPU Memory**: If you encounter OOM errors, reduce `BATCH_SIZE` in `train.py`
- **Faster Training**:
  - Use GPU (CUDA) if available
  - Reduce `EPOCHS` for quick experiments (e.g., 1000 epochs)
  - Increase `BATCH_SIZE` if you have enough GPU memory

## Differences from TensorFlow Version

This PyTorch implementation maintains the same architecture and functionality as the original TensorFlow version, with modern improvements:

- ✅ **Framework**: Migrated from TensorFlow 1.x to PyTorch
- ✅ **Session management**: Removed TensorFlow sessions in favor of PyTorch's imperative style
- ✅ **Checkpointing**: Uses PyTorch's native checkpoint format (`.pt` files) in `./artifacts/`
- ✅ **Progress tracking**: Added tqdm progress bars with real-time metrics
- ✅ **Auto-download**: Automatic data download with multiple fallback sources
- ✅ **Code structure**: More modular and Pythonic structure using `nn.Module`
- ✅ **GPU optimization**: Automatic CUDA detection and utilization

## Legacy TensorFlow Version

The original TensorFlow 1.x implementation is preserved in `./legacy_tensorflow/` for reference.

## Output Structure

After running the pipeline, your directory will look like:

```
.
├── data/
│   ├── wordsEn.txt           # Downloaded word list (370K+ words)
│   └── preprocessed.npy      # Processed data (248K words, shape: [248463, 10, 1])
│
├── artifacts/
│   ├── model_epoch_1000.pt   # Checkpoint at epoch 1000
│   ├── model_epoch_2000.pt   # Checkpoint at epoch 2000
│   ├── ...                   # Checkpoints every 1000 epochs
│   ├── model_epoch_10000.pt  # Checkpoint at epoch 10000
│   └── model_final.pt        # Final model (same as epoch 10000)
│
└── [source files]
```

Each checkpoint contains:
- `model_state_dict`: Model parameters
- `optimizer_state_dict`: Optimizer state (for resuming training)
- `epoch`: Current epoch number
- `loss`: Validation loss at checkpoint
- `accuracy`: Validation accuracy at checkpoint

## License

MIT License

## References

- Original implementation: TensorFlow 1.x character autoencoder
- Word list sources:
  - [SIL International Linguistics](http://www-01.sil.org/linguistics/wordlists/english/)
  - [dwyl/english-words](https://github.com/dwyl/english-words)
  - [first20hours/google-10000-english](https://github.com/first20hours/google-10000-english)
