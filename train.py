from gensim.models import Word2Vec
import matplotlib.pyplot as pyplot
import sklearn.decomposition.pca as PCA
import os

# class MySentences(object):
#     def __init__(self, dirname):
#         self.dirname = dirname
#
#     def __iter__(self):
#         for fname in os.listdir(self.dirname):
#             for line in open(os.path.join(self.dirname, fname), 'rt', encoding='UTF8'):
#                 yield line.split()

# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
             ['this', 'is', 'the', 'second', 'sentence'],
             ['yet', 'another', 'sentence'],
             ['one', 'more', 'sentence'],
             ['and', 'the', 'final', 'sentence']]
# sg가 0이면 Skip-gram, 1이면 CBOW
# hs가 1이면 hierachiclal softmax가 사용되고, 0이면 negative sampling
model = Word2Vec(sentences, size=400, min_count=5, workers=4, iter=100, hs=1)
# model.train(sentences, total_examples=model.corpus_count, epochs=model.iter, start_alpha=0.01, end_alpha=0.001,
#             word_count=0)
model.save('../wordModel/vectors.txt')

# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

