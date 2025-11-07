#!/bin/bash

################################################################################
# Character Autoencoder - Complete Pipeline
# This script runs the entire training pipeline from start to finish:
# 1. Downloads English word list
# 2. Preprocesses data
# 3. Trains the model
# 4. Runs interpolation demos
################################################################################

set -e  # Exit on error

# Color codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored headers
print_header() {
    echo -e "\n${CYAN}================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

################################################################################
# Step 0: Check Dependencies
################################################################################

print_header "Step 0: Checking Dependencies"

if ! command -v python &> /dev/null; then
    print_error "Python is not installed. Please install Python 3.7+ first."
    exit 1
fi

print_info "Python version: $(python --version)"

# Check if required packages are installed
print_info "Checking Python packages..."
if python -c "import torch" 2>/dev/null; then
    print_success "PyTorch is installed"
else
    print_warning "PyTorch not found. Installing dependencies..."
    pip install -r requirements.txt
fi

if python -c "import numpy" 2>/dev/null; then
    print_success "NumPy is installed"
fi

if python -c "import tqdm" 2>/dev/null; then
    print_success "tqdm is installed"
fi

# Check GPU availability
if python -c "import torch; print('GPU available' if torch.cuda.is_available() else 'CPU only')" | grep -q "GPU"; then
    print_success "GPU (CUDA) is available - training will be fast!"
else
    print_info "GPU not available - training will use CPU (slower)"
fi

################################################################################
# Step 1: Download Data
################################################################################

print_header "Step 1: Downloading Word List"

if [ -f "data/wordsEn.txt" ]; then
    print_success "Word list already exists at data/wordsEn.txt"
else
    print_info "Downloading English word list..."
    python download_data.py
    print_success "Word list downloaded successfully"
fi

################################################################################
# Step 2: Preprocess Data
################################################################################

print_header "Step 2: Preprocessing Data"

if [ -f "data/preprocessed.npy" ]; then
    print_warning "Preprocessed data already exists. Skipping preprocessing."
    print_info "Delete data/preprocessed.npy to rerun preprocessing."
else
    print_info "Converting words to numerical format..."
    python preprocess.py
    print_success "Data preprocessing complete"
fi

################################################################################
# Step 3: Train Model
################################################################################

print_header "Step 3: Training Model"

print_info "Starting training with the following configuration:"
echo "  - Epochs: 10,000"
echo "  - Batch Size: 100"
echo "  - Learning Rate: 0.001"
echo "  - State Size: 128"
echo ""
print_info "Checkpoints will be saved to ./artifacts/ every 1000 epochs"
print_info "Press Ctrl+C to stop training (checkpoints are saved)"
echo ""

# Check if model already exists
if [ -f "artifacts/model_final.pt" ]; then
    print_warning "Final model already exists at artifacts/model_final.pt"
    read -p "Do you want to retrain? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Skipping training. Using existing model."
        SKIP_TRAINING=true
    fi
fi

if [ -z "$SKIP_TRAINING" ]; then
    python train.py
    print_success "Training complete!"
fi

################################################################################
# Step 4: Run Interpolation Demo
################################################################################

print_header "Step 4: Latent Space Interpolation Demo"

if [ -f "artifacts/model_final.pt" ] || ls artifacts/model_epoch_*.pt 1> /dev/null 2>&1; then
    print_info "Running interpolation demo with trained model..."
    python interp_demo.py
    print_success "Demo complete!"
else
    print_error "No trained model found. Please complete training first."
    exit 1
fi

################################################################################
# Complete
################################################################################

print_header "Pipeline Complete!"

echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║  ✓ All steps completed successfully!                        ║"
echo "║                                                              ║"
echo "║  Your model is trained and ready to use.                    ║"
echo "║                                                              ║"
echo "║  Next steps:                                                 ║"
echo "║  • View checkpoints in ./artifacts/                          ║"
echo "║  • Run interpolation demo: python interp_demo.py             ║"
echo "║  • Experiment with different words in interp_demo.py         ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
