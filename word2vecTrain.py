from gensim.models import Word2Vec
import time
import logging
import os


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for subdir in os.listdir(self.dirname):
            print(subdir)
            for fname in os.listdir(self.dirname + subdir):
                print(fname)
                for line in open(os.path.join(self.dirname + subdir, fname), 'rt', encoding='UTF8'):
                    yield line.split()


# define training data
sentences = MySentences("../sentences/")
# train model
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
start = time.time()
model = Word2Vec(sentences, size=100, workers=8, iter=15, min_count=5, sg=1, min_alpha=0.01)
print(time.time() - start)
model.save('../wordModel/vectors.txt')

