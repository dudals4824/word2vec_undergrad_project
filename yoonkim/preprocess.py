import numpy as np
import time
import pickle
import pandas as pd
import sys
import re
from collections import defaultdict

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def load_bin_vec(fname, vocab):

    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)

    return word_vecs


def build_data_cv(textdata, cv=10, clean_string=True):
    revs = []
    vocab = defaultdict(float)

    with open("text8", "r") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            orig_rev = clean_str(" ".join(rev))
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1

            split = orig_rev.split()
            for i in range(0, len(split)):
                textSet = split[i:i + window]
                idx = int(i + window / 2)
                ans = None

                if len(textSet) < window:
                    continue
                else:
                    ans = split[idx]
                    s = ""
                    for j in range(0, window):
                        s = s + " " + str(textSet[j])
                    datum = {ans, s, window, np.random.randint(0, cv)}
                    revs.append(datum)

    return revs, vocab


# Global Variable
window = 11


if __name__ == "__main__":
    w2vModel = "word2vec-model.data-00000-of-00001"
    # vector file input
    textdata = "text8"
    start = time.time()
    print("loading data...")
    revs, vocab = build_data_cv(textdata, cv=10, clean_string=True)
    print("elapsed time: " + str(time.time() - start))

    max_l = window
    print("data loaded!")
    print("number of inputs: " + str(len(revs)))
    print("vocab size: " + str(len(vocab)))
    print("max sentence length :" + str(max_l))
    print("loading word2vec vectors...")

    w2v = load_bin_vec(w2vModel, vocab)
