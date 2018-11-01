import nltk
from nltk.corpus import wordnet
from nltk import WordNetLemmatizer

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

newtext = open("./data/new_new_text8", "w")
newtextstr = ""
lemmatizer = WordNetLemmatizer()
tokens = nltk.word_tokenize(text)
tagged = nltk.pos_tag(tokens)
for w, tag in tagged:
    wntag = get_wordnet_pos(tag)
    if wntag is None:  # not supply tag in case of None
        lemma = lemmatizer.lemmatize(word=w)
    else:
        lemma = lemmatizer.lemmatize(word=w, pos=wntag)
    newtextstr += " "
    newtextstr += lemma

newtext.write(newtextstr)

