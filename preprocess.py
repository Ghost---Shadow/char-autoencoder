import numpy as np

def stringsToArray(words,UNK_TOKEN = 26):
    output = []
    for word in words:
        output.append([])
        for i in range(len(word),10):
            output[-1].append([UNK_TOKEN])
        for character in word:
            output[-1].append([ord(character) - ord('a')])
            
    output = np.array(output)
    return output

def preprocess():
    # http://www-01.sil.org/linguistics/wordlists/english/
    INPUT_FILE = '../../../NLP/data/wordsEn.txt'
    MAX_ALLOWED_LEN = 10
    words = []

    with open(INPUT_FILE,'r') as f:
        words = f.readlines()

    words = [x.strip() for x in words] 
    print('Word count',len(words))

    max_len = max(map(len,words))
    print('Maximum word length: ',max_len)

    words = list(filter(lambda word: len(word) <= MAX_ALLOWED_LEN,words))
    print('%d words with len <= %d'%(len(words),MAX_ALLOWED_LEN))

    output = stringsToArray(words)
    np.save('./preprocessed.npy',output)
