import re
import pandas as pd
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

model = Word2Vec.load("./wvModel")
wv = model.wv

window = 11

traindata = []
testdata = []
with open("./text8", "r") as f:
    for line in f:
        rev = []
        rev.append(line.strip())
        orig_rev = list(clean_str(" ".join(rev)).split())
        datanum = 1
        for i in range(0, len(orig_rev)):
            print("진행중...")
            seq = ""
            label = 0
            k = i

            num = 0
            while num < window:
                if num == 5 and (orig_rev[k] in wv.vocab):
                    label = wv.vocab[orig_rev[k]].index
                    num += 1
                elif orig_rev[k] in wv.vocab:
                    seq = seq + str(wv.vocab[orig_rev[k]].index) + " "
                    num += 1
                k += 1

            if datanum % 10 == 0:
                testdata.append([seq, label])
            else:
                traindata.append([seq, label])

    traindf = pd.DataFrame(traindata, columns=["sequence", "label"])
    testdf = pd.DataFrame(testdata, columns=["sequence", "label"])
    traindf.to_csv("./train.csv", encoding="utf-8", index=False)
    testdf.to_csv("./test.csv", encoding="utf-8", index=False)
