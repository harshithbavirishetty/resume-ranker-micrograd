# utils.py

import re
import numpy as np

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return text.split()

def build_vocab(texts):
    vocab = set()
    for text in texts:
        vocab.update(tokenize(text))
    return sorted(list(vocab))

def vectorize(text, vocab):
    tokens = tokenize(text)
    return np.array([tokens.count(word) for word in vocab])
