from gensim.models import Word2Vec
import numpy as np
import os

model = Word2Vec.load('../wordModel/vectors.txt')


l1 = np.sum(model.wv.vectors[[0, 1, 2, 3]], axis=0)

print(l1)

print(model.predict_output_word(context_words_list=['Many', 'types', 'and', 'of', 'anarchism'], topn=10))


# def predict_output_word(self, context_words_list, topn=10):
#     """Report the probability distribution of the center word given the context words
#     as input to the trained model.
#     Parameters
#     ----------
#     context_words_list : :obj: `list` of :obj: `str`
#         List of context words
#     topn: int
#         Return `topn` words and their probabilities
#     Returns
#     -------
#     :obj: `list` of :obj: `tuple`
#         `topn` length list of tuples of (word, probability)
#     """
#     if not self.negative:
#         raise RuntimeError(
#             "We have currently only implemented predict_output_word for the negative sampling scheme, "
#             "so you need to have run word2vec with negative > 0 for this to work."
#         )
#
#     if not hasattr(self.wv, 'vectors') or not hasattr(self.trainables, 'syn1neg'):
#         raise RuntimeError("Parameters required for predicting the output words not found.")
#     # self: model
#     # model.wv.vocab : word vector의 전체 vocabulary (사전)
#     word_vocabs = [self.wv.vocab[w] for w in context_words_list if w in self.wv.vocab]
#     if not word_vocabs:
#         warnings.warn("All the input context words are out-of-vocabulary for the current model.")
#         return None
#     # context word 배열의 index 배열
#     word2_indices = [word.index for word in word_vocabs]
#
#     # context word 들의 index들을 이용해 전체 word vector 확률값을 합산한다.
#     # cbow_mean이 0일경우 그 합을 그대로 이용하고, 1일 경우에는 평균값을 이용한다.
#     # cbow_mean이 1인것은 학습 모델을 cbow로 적용시켰을때 한함.
#     l1 = np_sum(self.wv.vectors[word2_indices], axis=0)
#     if word2_indices and self.cbow_mean:
#         l1 /= len(word2_indices)
#
#     # propagate hidden -> output and take softmax to get probabilities
#     # 확률을 얻기위해 softmax를 활용한다
#     prob_values = exp(dot(l1, self.trainables.syn1neg.T))
#     prob_values /= sum(prob_values)
#     # topn을 이용해 확률값을 기준으로 내림차순으로 정렬한다.
#     top_indices = matutils.argsort(prob_values, topn=topn, reverse=True)
#     # returning the most probable output words with their probabilities
#     return [(self.wv.index2word[index1], prob_values[index1]) for index1 in top_indices]