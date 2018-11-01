import nltk
from nltk.corpus import wordnet
from nltk import WordNetLemmatizer
from gensim.models import Word2Vec
import pandas as pd

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


text = open("./data/text8", "r")
text = text.readline()
textlist = text.split(" ")

lemmatizer = WordNetLemmatizer()
tokens = nltk.word_tokenize(text)
tagged = nltk.pos_tag(tokens)

model = Word2Vec.load("./wordmodel/new_50kwvModel")
wv = model.wv

inverse_dict = dict()

for w, tag in tagged:
    wntag = get_wordnet_pos(tag)
    if wntag is None:  # not supply tag in case of None
        lemma = lemmatizer.lemmatize(word=w)
    else:
        lemma = lemmatizer.lemmatize(word=w, pos=wntag)

    if lemma in wv.vocab:
        if lemma in inverse_dict.keys():
            pre = inverse_dict[lemma]
            if w in pre.split(" "):
                continue
            else:
                pre += " " + w
                inverse_dict[lemma] = pre
        else:
            inverse_dict[lemma] = w

df = pd.DataFrame.from_dict(inverse_dict, orient='index')
df.to_csv("./data/inverse_dictionary.csv")