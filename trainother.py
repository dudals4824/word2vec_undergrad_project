from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import time
import logging
# define training data
sentences = [['Anarchism', 'does', 'not', 'offer'],
             ['a', 'fixed', 'body', 'of', 'doctrine'],
             ['from', 'a', 'single', 'particular', 'world'],
             ['view', ',', 'instead', 'fluxing', 'and'],
             ['flowing', 'as', 'a', 'philosophy'],
             ['Many', 'types', 'and', 'traditions', 'of'],
             ['anarchism', 'exist', 'not', 'all', 'of'],
             ['which', 'are', 'mutually', 'exclusive'],
             ['Anarchist', 'schools', 'of', 'thought', 'can'],
             ['differ', 'fundamentally', ',', 'supporting', 'anything'],
             ['from', 'extreme', 'individualism'],
             ['to', 'complete', 'collectivism']]
# train model
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
start = time.time()
model = Word2Vec(sentences, size=100, workers=4, iter=100, min_count=1, sg=0)

model.train(sentences, total_examples=model.corpus_count, epochs=model.iter, start_alpha=0.01, end_alpha=0.001,
            word_count=0)
print(time.time() - start)
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
