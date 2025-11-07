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
🎨 **Latent Space Interpolation**: Visualize smooth transitions between words with parallelogram interpolation
🔄 **Multiple Models**: Compare character autoencoder with pre-trained embeddings (GloVe, Word2Vec)
📥 **Easy Model Download**: Dedicated script to download and cache pre-trained embeddings

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

### 3. (Optional) Download Pre-trained Embeddings

To compare with pre-trained word embeddings (GloVe and Word2Vec):

```bash
python download_embeddings.py
```

This will download and cache:
- **GloVe** (glove-wiki-gigaword-50): ~66MB
- **Word2Vec** (word2vec-google-news-300): ~1.6GB

You can also download specific models:
```bash
python download_embeddings.py --model glove      # Only GloVe
python download_embeddings.py --model word2vec   # Only Word2Vec
```

**Note**: These are optional and only needed if you want to compare the character autoencoder with pre-trained word embeddings.

### 4. Visualize Latent Space

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
- 🔄 Switch between models: Character Autoencoder, GloVe, or Word2Vec

**Option B: Command Line**
```bash
# Use character autoencoder (default)
python interp_demo.py

# Use GloVe embeddings
python interp_demo.py --model glove

# Use Word2Vec embeddings
python interp_demo.py --model word2vec

# Compare all models
python interp_demo.py --model all
```

This demonstrates latent space interpolation for various word triplets in the terminal.

**Example Output (Char-level-LSTM)**:

Test case 1: ['man', 'woman', 'king']
--------------------------------------------------------------------------------
Interpolation grid (2D manifold):
  Corners: [man] at [0,0], [woman] at [9,0], [king] at [0,9]
  Fourth corner [9,9] = woman + king - man (computed)

Row 0:          man |          man |          man |          kan |          kng |         wing |         king |         king |         king |         king
Row 1:          man |          man |          man |          kan |         wkng |         wing |         king |         king |         king |         king
Row 2:          man |          man |          man |         wman |         wing |         king |         king |         king |        wking |        wking
Row 3:          man |         oman |         oman |         iman |         wiag |         king |         king |        wking |        wking |        wking
Row 4:         oman |         oman |         oman |         oman |         wiag |        wwing |        wking |        wking |        wking |        wking
Row 5:         oman |         oman |         oman |         oman |        woiag |        wwing |        wking |        wking |        wking |        wking
Row 6:         oman |         oman |        woman |        woman |        woian |        wwing |        wking |        wking |        wking |       wwking
Row 7:         oman |        woman |        woman |        woman |        woian |        wwing |        wking |        wking |       wwking |       woking
Row 8:        woman |        woman |        woman |        woman |        woian |        woiag |       wowing |       woking |       woking |       woking
Row 9:        woman |        woman |        woman |        woman |       wooian |       wooiag |       wowing |       woking |       woning |       woning

**Example Output (Word2Vec)**:



**Example Output (Glove)**:

Test case 1: ['man', 'woman', 'king']
--------------------------------------------------------------------------------
Interpolation grid (2D manifold):
  Corners: [man] at [0,0], [woman] at [9,0], [king] at [0,9]
  Fourth corner [9,9] = woman + king - man (computed)

Row 0:          man |          man |          man |          man |          man |         king |         king |         king |         king |         king
Row 1:          man |          man |          man |          man |          man |         king |         king |         king |         king |         king
Row 2:          man |          man |          man |          man |          man |         king |         king |         king |         king |         king
Row 3:          man |          man |          man |          man |          man |         king |         king |         king |         king |         king
Row 4:          man |          man |          man |          man |       father |       father |         king |         king |         king |         king
Row 5:        woman |        woman |        woman |        woman |       mother |       father |         king |         king |         king |         king
Row 6:        woman |        woman |        woman |        woman |       mother |       mother |         king |         king |         king |         king
Row 7:        woman |        woman |        woman |        woman |       mother |       mother |        queen |         king |         king |         king
Row 8:        woman |        woman |        woman |        woman |       mother |       mother |        queen |        queen |         king |         king
Row 9:        woman |        woman |        woman |        woman |       mother |       mother |       mother |        queen |         king |         king

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

**Interpolation Method**: The model uses **parallelogram interpolation** to create a 2D manifold in latent space:
- **Corner [0,0]**: Word A (first input)
- **Corner [max,0]**: Word B (second input)
- **Corner [0,max]**: Word C (third input)
- **Corner [max,max]**: Computed as `B + C - A` (the opposite corner of the parallelogram)

The interpolation formula is: `point[i,j] = A + (i/max)*(B-A) + (j/max)*(C-A)`

This creates a bilinear interpolation where:
- Rows interpolate from A→B, shifted by the A→C direction
- Columns interpolate from A→C, shifted by the A→B direction
- The fourth corner is **automatically computed**, not directly encoded from an input

This allows visualization of smooth transitions in the learned latent space and exploration of the 2D manifold defined by the three input words.

**Example Use Cases**: The parallelogram interpolation works particularly well for demonstrating transformations:

**Semantic Relationships** (like the classic word2vec analogy):
- `["man", "woman", "king"]` → Fourth corner should approximate "queen" (gender transformation transfer)
  - Formula: `woman + king - man ≈ queen`
  - This is the famous word embedding analogy demonstrating that the model learns semantic relationships

**Morphological Transformations** (character-level patterns):
- `["cat", "cats", "dog"]` → Fourth corner should approximate "dogs" (plural transformation transfer)
- `["run", "running", "walk"]` → Fourth corner should approximate "walking" (gerund transformation transfer)
- `["happy", "happier", "sad"]` → Fourth corner should approximate "sadder" (comparative transformation transfer)
- `["fast", "faster", "slow"]` → Fourth corner should approximate "slower"

These examples work well because the model learns to transfer patterns (both semantic relationships and character-level patterns like adding "-s", "-ing", or "-er") across different base words.

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
