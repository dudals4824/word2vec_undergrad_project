import numpy as np
import re
import pandas as pd
import itertools
from collections import Counter
from gensim.models import Word2Vec


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
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
    return string.strip().lower()

model = Word2Vec.load("./data/rt-polaritydata/wvModel")
wv = model.wv

window = 11


data = []
with open("./data/rt-polaritydata/text8", "r") as f:
    for line in f:
        rev = []
        rev.append(line.strip())
        orig_rev = list(clean_str(" ".join(rev)).split())

        for i in range(0, len(orig_rev)):
            seq = list()
            k = i
            while len(seq) < window:
                if orig_rev[k] in wv.vocab:
                    seq.append(wv.vocab[orig_rev[k]].index)
                    k += 1
                else:
                    k += 1
                    continue
            label = seq[5]
            data.append([seq, label])

    df = pd.DataFrame(data, columns=["sequence", "label"])
    df.to_csv("./data/rt-polaritydata/data.csv", encoding="utf-8", index=False)