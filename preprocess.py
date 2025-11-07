import numpy as np
import torch


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
    Download word list from: http://www-01.sil.org/linguistics/wordlists/english/
    """
    INPUT_FILE = "../../../NLP/data/wordsEn.txt"
    MAX_ALLOWED_LEN = 10
    words = []

    with open(INPUT_FILE, "r") as f:
        words = f.readlines()

    words = [x.strip() for x in words]
    print("Word count:", len(words))

    max_len = max(map(len, words))
    print("Maximum word length:", max_len)

    words = list(filter(lambda word: len(word) <= MAX_ALLOWED_LEN, words))
    print("%d words with len <= %d" % (len(words), MAX_ALLOWED_LEN))

    output = stringsToArray(words)
    np.save("./preprocessed.npy", output)
    print("Saved preprocessed data to ./preprocessed.npy")


if __name__ == "__main__":
    preprocess()
