# 1. 원본  text8 라인으로 나누기 (1000라인) -> context, target 구분
# 2. 나뉜 라인에서 context 부분에 동일한 lemmatizer 적용
# 3. target 단어가 a 와 같이 한 단어일 경우 pass (최소 2단어 이상인 경우만 취급)
# 4. target의 에러는 다음과 같은 case로 나누어 적용 -> 걍 1개로
#        word length = n, word index = i
# 	1) (n/2 - 1)<= i <(n/2 + n/2)까지 rand char로 대체
# random.choice(string.ascii_lowercase)

import nltk
from nltk.corpus import wordnet
from nltk import WordNetLemmatizer
from gensim.models import Word2Vec
import pandas as pd
import random
import string


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

model = Word2Vec.load("./data/new_50kwvModel")
wv = model.wv
text4test = open("./data/text4test", "r")
# lemmatize를 위한 Wordnet Lemmatizer 선언
lemmatizer = WordNetLemmatizer()
# csv로 만들기위한 list 선언
test_data = list()
# 첫번째 라인 처리
while len(test_data) < 10000:
    line = text4test.readline().split("\n")
    line = line[0].split(" ")
    datum = ""
    for i in range(0, 11):
        if i != 5:
            token = nltk.word_tokenize(line[i])
            tagged = nltk.pos_tag(token)
            if len(tagged) == 0:
                break
            w = tagged[0][0]
            tag = tagged[0][1]
            wntag = get_wordnet_pos(tag)
            if wntag is None:  # not supply tag in case of None
                lemma = lemmatizer.lemmatize(word=w)
            else:
                lemma = lemmatizer.lemmatize(word=w, pos=wntag)

            if lemma in wv.vocab:
                datum += str(wv.vocab[lemma].index)
                datum += " "
            else:
                datum = ""
                break
    if datum != "":
        label = line[5]
        spel = line[5]
        char_len = len(spel)
        if char_len > 1:
            broken_str = ""
            for idx in range(0, char_len):
                if (int(char_len / 2) - 1) <= idx < ((int(char_len / 2) * 2) -1):
                    broken_str += chr(random.randint(33, 126))
                else:
                    broken_str += spel[idx]
            datum += broken_str
        elif char_len == 1:
            datum += label
        test_data.append([datum, label])

testdf = pd.DataFrame(test_data, columns=["sequence", "origin_word"])
testdf.to_csv("./data/test_for_eval_10k.csv", encoding="utf-8", index=False)

