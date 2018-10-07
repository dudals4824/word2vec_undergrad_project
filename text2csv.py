import pandas as pd
from gensim.models import Word2Vec
from collections import OrderedDict

model = Word2Vec.load("./wordmodel/subsampled_wvModel")
wv = model.wv

window = 10


train_data = []
valid_data = []
test_data = []

f = open("./text8/test_text8", "r")
orig_rev = f.readline().split(" ")

# frequency table : S
freqtable = dict()

for t in orig_rev:
    if t not in freqtable.keys():
        freqtable[t] = 1
    else:
        freqtable[t] += 1

freqtable = OrderedDict(sorted(freqtable.items(), key=lambda x: x[1], reverse=True))

# frequency table : E
mk_data_idx = 0
for i in range(0, len(orig_rev)):
    seq = list()
    k = i
    while len(seq) < window:
        # out of index 처리
        if k == len(orig_rev):
            seq.clear()
            break
        # 단어가 50000개 이내에 존재할 때, 5개 미만이거나 5개 초과일 경우에는 input으로 단어를 채워줌.
        if (orig_rev[k] in wv.vocab) and (len(seq) < 5):
            seq.append(wv.vocab[orig_rev[k]].index)
            k += 1
        elif (orig_rev[k] in wv.vocab) and (len(seq) > 5):
            if seq[5] == " ":
                seq.pop(5)
            seq.append(wv.vocab[orig_rev[k]].index)
            k += 1
        # 단어가 50000개 이내에 존재할 때, sequence의 길이가 5이면 label을 찾아줘야할 차례
        elif (orig_rev[k] in wv.vocab) and len(seq) == 5:
            # 15000개 이내에 있는지 확인하는 부분
            idx = 0
            for f in freqtable:
                # k번째 단어가 15000개 안에 있다면
                # label로 저장하고 break
                if idx < 15000 and f == orig_rev[k]:
                    label = wv.vocab[orig_rev[k]].index
                    k += 1
                    seq.append(" ")
                    break
                # k번째 단어가 15000개 안에 없다면 break
                elif idx >= 15000:
                    k += 1
                    break
                idx += 1
        else:
            k += 1
            continue
    if mk_data_idx == 10:
        mk_data_idx = 0
    if len(seq) >= window:
        seq_str = ""
        for s in seq:
            seq_str += str(s)
            seq_str += " "
        if mk_data_idx % 8 == 0:
            valid_data.append([seq_str, label])
        elif mk_data_idx % 9 == 0:
            test_data.append([seq_str, label])
        else:
            train_data.append([seq_str, label])
    mk_data_idx += 1


    traindf = pd.DataFrame(train_data, columns=["sequence", "label"])
    validdf = pd.DataFrame(valid_data, columns=["sequence", "label"])
    testdf = pd.DataFrame(test_data, columns=["sequence", "label"])
    traindf.to_csv("./data/new_train_data.csv", encoding="utf-8", index=False)
    validdf.to_csv("./data/new_valid_data.csv", encoding="utf-8", index=False)
    testdf.to_csv("./data/new_test_data.csv", encoding="utf-8", index=False)