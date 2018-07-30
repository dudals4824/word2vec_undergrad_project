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


def load_data_and_labels(positive_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    model = Word2Vec.load("./data/rt-polaritydata/wvModel")
    wv = model.wv
    print(len(wv.vocab))
    df = pd.read_csv("./data/rt-polaritydata/data.csv")

    x_text = list()
    y = list()
    for row in df.itertuples(index=True):
        seq = getattr(row, "sequence")
        seq = clean_str(seq)
        seq = seq.split(" , ")
        x = list()
        for i in seq:
            x.append(int(i))

        x_text.append(x)
        y.append(getattr(row, "label"))


    # print(len(wv.vocab))
    # print(wv.vocab['the'].index)


    # 없는단어 처리하는 예외처리 라인
    # if 'UNK' in wv.vocab:
    #     print(wv.vocab['UNK'].index)
    # else:
    #     print(-123456778iiㅑ)

    # revs = []
    # window = 11
    # with open(positive_data_file, "r") as f:
    #     for line in f:
    #         rev = []
    #         rev.append(line.strip())
    #         orig_rev = list(clean_str(" ".join(rev)).split())
    #
    #         for i in range(0, len(orig_rev)):
    #             s = orig_rev[i:i + window]
    #             idx = int(i + window / 2)
    #             anslist = np.zeros(len(wv.vocab) + 1)
    #             ans = None
    #
    #             if len(s) < window:
    #                 continue
    #             else:
    #                 ans = orig_rev[idx]
    #                 w = str()
    #                 for k in range(0, len(s)):
    #                     if k == len(s)-1:
    #                         w = w + s[k]
    #                     else:
    #                         w = w + s[k] + " "
    #                 # print(ans)
    #                 revs.append(w)
    #                 if ans in wv.vocab:
    #                     anslist[wv.vocab[ans].index] = 1
    #                     y.append(wv.vocab[ans].index)
    #                     print(len(y))
    #                 else:
    #                     anslist[len(wv.vocab)] = 1
    #                     y.append(len(wv.vocab))
    #                     print(len(y))
    #
    #
    # # Load data from files
    # # positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    # # positive_examples = [s.strip() for s in positive_examples]
    # # negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    # # negative_examples = [s.strip() for s in negative_examples]
    # # Split by words
    # x_text = revs
    # x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    # _는 문장 하나를 뜻함.
    # positive_labels = answers
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
            print(shuffled_data)
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
