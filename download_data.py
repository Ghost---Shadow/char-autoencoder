"""
Download English word list.
Falls back to creating a comprehensive sample dataset if online sources are unavailable.
"""
import urllib.request
import os
from pathlib import Path

DATA_DIR = Path("./data")
OUTPUT_FILE = DATA_DIR / "wordsEn.txt"

# List of alternative word list URLs
WORDLIST_URLS = [
    "http://www.institute.loni.org/lasigma/ret/products/Burkman/wordsEn.txt",
    "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt",
    "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa.txt",
]

def create_fallback_wordlist():
    """Create a comprehensive fallback word list for demonstration."""
    print("Creating fallback word list with common English words...")

    # Comprehensive word list for training
    words = []

    # Common short words
    common_words = [
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'do', 'for',
        'go', 'he', 'hi', 'i', 'if', 'in', 'is', 'it', 'me', 'my',
        'no', 'of', 'on', 'or', 'so', 'to', 'up', 'us', 'we',
    ]
    words.extend(common_words)

    # Letter patterns (good for learning character relationships)
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        words.extend([letter, letter*2, letter*3])

    # Common word families
    word_families = {
        'at': ['cat', 'bat', 'mat', 'hat', 'rat', 'sat', 'fat', 'pat', 'vat'],
        'an': ['can', 'man', 'pan', 'ran', 'tan', 'van', 'ban', 'fan'],
        'in': ['bin', 'din', 'fin', 'kin', 'pin', 'sin', 'tin', 'win'],
        'ot': ['cot', 'dot', 'got', 'hot', 'jot', 'lot', 'not', 'pot', 'rot'],
        'ed': ['bed', 'fed', 'led', 'red', 'wed', 'shed', 'sled'],
        'un': ['bun', 'fun', 'gun', 'nun', 'pun', 'run', 'sun'],
    }

    for pattern, word_list in word_families.items():
        words.extend(word_list)

    # Common nouns
    nouns = [
        'apple', 'ball', 'book', 'box', 'boy', 'cake', 'car', 'cat',
        'chair', 'coat', 'cow', 'cup', 'day', 'dog', 'door', 'duck',
        'ear', 'egg', 'eye', 'face', 'farm', 'fish', 'flower', 'foot',
        'game', 'girl', 'goat', 'hand', 'hat', 'head', 'hill', 'home',
        'horse', 'house', 'ice', 'jam', 'key', 'kite', 'lake', 'lamp',
        'leg', 'lion', 'milk', 'moon', 'mouse', 'nest', 'nose', 'orange',
        'pen', 'pig', 'queen', 'rain', 'ring', 'road', 'room', 'rose',
        'sea', 'shoe', 'snow', 'star', 'sun', 'table', 'tent', 'time',
        'toy', 'tree', 'umbrella', 'van', 'wall', 'water', 'window',
        'yard', 'year', 'zoo',
    ]
    words.extend(nouns)

    # Common verbs
    verbs = [
        'ask', 'be', 'buy', 'call', 'come', 'do', 'eat', 'fall', 'feel',
        'find', 'fly', 'get', 'give', 'go', 'have', 'hear', 'help', 'jump',
        'keep', 'know', 'learn', 'like', 'live', 'look', 'love', 'make',
        'move', 'need', 'open', 'play', 'put', 'read', 'run', 'say', 'see',
        'sell', 'send', 'show', 'sit', 'sleep', 'speak', 'stand', 'stay',
        'take', 'talk', 'tell', 'think', 'try', 'turn', 'use', 'wait',
        'walk', 'want', 'watch', 'work', 'write',
    ]
    words.extend(verbs)

    # Common adjectives
    adjectives = [
        'bad', 'big', 'black', 'blue', 'cold', 'dark', 'deep', 'fast',
        'fine', 'free', 'full', 'good', 'great', 'green', 'happy', 'hard',
        'high', 'hot', 'large', 'light', 'long', 'low', 'new', 'nice',
        'old', 'poor', 'quick', 'quiet', 'red', 'rich', 'sad', 'short',
        'slow', 'small', 'soft', 'tall', 'warm', 'white', 'wide', 'yellow',
        'young',
    ]
    words.extend(adjectives)

    # Compound-ish words
    compounds = [
        'baseball', 'basketball', 'football', 'snowball', 'sunlight',
        'moonlight', 'birthday', 'daylight', 'rainbow', 'airplane',
        'firefly', 'butterfly', 'ladybug', 'dragonfly', 'starfish',
        'jellyfish', 'goldfish', 'bluebird', 'blackbird',
    ]
    words.extend(compounds)

    # Duplicate to create larger dataset
    words = words * 50  # Create ~15000+ examples

    return words

def download_wordlist():
    """Download the English word list if it doesn't exist."""

    # Create data directory
    DATA_DIR.mkdir(exist_ok=True)

    # Check if file already exists
    if OUTPUT_FILE.exists():
        print(f"Word list already exists at {OUTPUT_FILE}")
        return str(OUTPUT_FILE)

    # Try downloading from multiple sources
    for url in WORDLIST_URLS:
        print(f"Attempting to download from {url}...")
        try:
            urllib.request.urlretrieve(url, OUTPUT_FILE)
            print(f"Successfully downloaded {OUTPUT_FILE.stat().st_size} bytes")

            # Verify the file
            with open(OUTPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                print(f"Word list contains {len(lines)} words")

            return str(OUTPUT_FILE)

        except Exception as e:
            print(f"Failed: {e}")
            continue

    # If all downloads failed, create fallback
    print("\nAll download sources failed. Creating fallback word list...")
    words = create_fallback_wordlist()

    with open(OUTPUT_FILE, 'w') as f:
        for word in words:
            f.write(f"{word}\n")

    print(f"Created fallback word list with {len(words)} words at {OUTPUT_FILE}")
    return str(OUTPUT_FILE)

if __name__ == "__main__":
    download_wordlist()
