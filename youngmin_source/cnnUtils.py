import numpy as np
import pandas as pd
import re, sys
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

def next_batch(start, batch_size):
    # start : epoch에서의 index를 뜻함.
    batch_xs = list()
    batch_ys = list()
    traindf = pd.read_csv("./data/rt-polaritydata/tt.csv")
    model = Word2Vec.load("./data/rt-polaritydata/wvModel")
    wv = model.wv
    for row in traindf.itertuples(index=True):

        if start <= int(row[0]) <= start + batch_size - 1:
            # 받은 row들을 1 X 300 데이터로 바꿔줘야한다.
            x = []
            seq = getattr(row, "sequence")
            seq = clean_str(seq)
            seq = seq.split(" , ")

            for elem in seq:
                x.append(wv.vectors[int(elem)])
            batch_xs.append(x)

            y = np.zeros(50095, dtype="int32")
            y[int(row[2])] = 1
            batch_ys.append(y)

        if int(row[0]) >= start + batch_size - 1:

            start = start + batch_size
            return start, batch_xs, batch_ys
