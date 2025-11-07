import numpy as np
import torch
from pathlib import Path


def stringsToArray(words, UNK_TOKEN=26):
    """
    Convert list of strings to numpy array representation.
    Each character is converted to an integer (a=0, b=1, ..., z=25, space=26).
    Words are padded to length 10 with UNK_TOKEN at the beginning.

    Args:
        words: List of strings to convert
        UNK_TOKEN: Token to use for padding (default 26 for space)

    Returns:
        numpy array of shape (num_words, 10, 1)
    """
    output = []
    for word in words:
        output.append([])
        # Pad at the beginning
        for i in range(len(word), 10):
            output[-1].append([UNK_TOKEN])
        # Convert characters to integers
        for character in word:
            output[-1].append([ord(character) - ord("a")])

    output = np.array(output)
    return output


def stringsToTensor(words, UNK_TOKEN=26, device="cpu"):
    """
    Convert list of strings to PyTorch tensor representation.

    Args:
        words: List of strings to convert
        UNK_TOKEN: Token to use for padding (default 26 for space)
        device: Device to place tensor on ('cpu' or 'cuda')

    Returns:
        torch.LongTensor of shape (num_words, 10)
    """
    array = stringsToArray(words, UNK_TOKEN)
    array = array.squeeze(-1)  # Remove last dimension
    return torch.from_numpy(array).long().to(device)


def preprocess():
    """
    Preprocess word list from text file and save as numpy array.
    Automatically downloads from: http://www-01.sil.org/linguistics/wordlists/english/
    """
    from download_data import download_wordlist

    # Download data if not present
    try:
        INPUT_FILE = download_wordlist()
    except Exception as e:
        print(f"Failed to download data: {e}")
        return

    MAX_ALLOWED_LEN = 10
    words = []

    print(f"Reading words from {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        words = f.readlines()

    words = [x.strip().lower() for x in words if x.strip()]
    print("Word count:", len(words))

    max_len = max(map(len, words))
    print("Maximum word length:", max_len)

    words = list(filter(lambda word: len(word) <= MAX_ALLOWED_LEN, words))
    words = list(filter(lambda word: word.isalpha(), words))  # Only letters
    print("%d words with len <= %d" % (len(words), MAX_ALLOWED_LEN))

    output = stringsToArray(words)

    # Save to data directory
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "preprocessed.npy"

    np.save(output_path, output)
    print(f"Saved preprocessed data to {output_path}")
    print(f"Data shape: {output.shape}")


if __name__ == "__main__":
    preprocess()
