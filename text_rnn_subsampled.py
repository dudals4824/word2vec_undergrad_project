import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from gensim.models import Word2Vec
import pandas as pd
import random
import math
import os

model = Word2Vec.load("./data/50kwvModel")
wv = model.wv
CHECK_POINT_DIR = './rnn_runs/checkpoint'
learning_rate = 0.001
epoch_num = 151
cell_num = 512
sequence_len = 10
word_num = 47915 #47915
vector_per_word = 128
batch_size = 256 #256


def next_batch(traindf, freqtable, start_index):
    data = traindf.values
    data_size = len(traindf)
    cur_idx = start_index

    x_zip = []
    y_zip = []

    while len(x_zip) < batch_size:
        target_word = wv.index2word[int(data[cur_idx][1])]
        subsampled_prob = (1 - math.sqrt((1e-5 / freqtable[target_word]))) * 100
        rand = random.randint(1, 100)

        if rand > subsampled_prob:
            input_x = [int(x) for x in data[cur_idx][0].split(" ")[0:10]]
            y = np.zeros(word_num, dtype=np.int32)
            y[int(data[cur_idx][1])] = 1
            x_zip.append(input_x)
            y_zip.append(y)

        cur_idx += 1

        if cur_idx == data_size:
            break
    start_index = cur_idx
    return x_zip, y_zip, start_index

def batch_validation(validationdf, freqtable, start_index):
    x_dev = []
    y_dev = []
    data = validationdf.values
    data_size = len(validationdf)
    cur_idx = start_index

    while len(x_dev) < batch_size:
        target_word = wv.index2word[int(data[cur_idx][1])]
        subsampled_prob = (1 - math.sqrt((1e-5 / freqtable[target_word]))) * 100
        rand = random.randint(1, 100)

        if rand > subsampled_prob:
            input_x = [int(x) for x in data[cur_idx][0].split(" ")[0:10]]
            y = np.zeros(word_num, dtype=np.int32)
            y[int(data[cur_idx][1])] = 1
            x_dev.append(input_x)
            y_dev.append(y)

        cur_idx += 1

        if cur_idx == data_size:
            break
    start_index = cur_idx

    return x_dev, y_dev, start_index


def train(_traindf, _dic):
    loss = 0
    correct = 0
    start_idx = 0
    train_step = 0
    _traindf = shuffle(_traindf)
    while True:
        x_batch, y_batch, start_idx = next_batch(_traindf, _dic, start_idx)
        if len(x_batch) == batch_size:
            train_step += 1
        elif len(x_batch) < batch_size:
            break
        feed_dict = {X: x_batch, Y: y_batch}
        train_accuracy, train_loss, _ = sess.run([correct_check, sampled_softmax, minimize], feed_dict)

        loss += train_loss
        correct += train_accuracy

    return correct/(batch_size*train_step + 1), loss

def validation(_validationdf, _dic):
    loss = 0
    correct = 0
    start_idx = 0
    dev_step = 0
    _validationdf = shuffle(_validationdf)
    while True:
        x_dev, y_dev, start_idx = next_batch(_validationdf, _dic, start_idx)
        if len(x_dev) == batch_size:
            dev_step += 1
        elif len(x_dev) < batch_size:
            break
        feed_dict = {X: x_dev, Y: y_dev}
        valid_accuracy, valid_loss, _ = sess.run([correct_check, sampled_softmax, minimize], feed_dict)

        loss += valid_loss
        correct += valid_accuracy

    return correct/(batch_size*dev_step + 1), loss

def run():
    traindf = pd.read_csv("./data/new_train_data.csv")
    validationdf = pd.read_csv("./data/new_valid_data.csv")
    # frequency table : S
    f = open("./data/new_text8");
    line = f.readline();
    arr = line.split(" ");
    dic = dict();

    for a in arr:
        if a in dic.keys():
            dic[a] += 1
        else:
            dic[a] = 1
    # frequency table: E

    for epoch in range(1, epoch_num):
        train_accuracy, train_loss = train(traindf, dic)
        valid_accuracy, valid_loss = validation(validationdf, dic)

        print("epoch: ", epoch)
        print("train accuracy | loss: " + str(train_accuracy)+ " | " + str(train_loss))
        print("validation accruacy | loss: "+ str(valid_accuracy)+" | " + str(valid_loss))

        summary = sess.run(merged, {train_accuracy_tensorboard: train_accuracy,
                                    train_loss_tensorboard: train_loss,
                                    valid_accuracy_tensorboard: valid_accuracy,
                                    valid_loss_tensorboard: valid_loss})
        writer.add_summary(summary, epoch)

        print("Saving network...")
        if not os.path.exists(CHECK_POINT_DIR):
            os.makedirs(CHECK_POINT_DIR)
        saver.save(sess, CHECK_POINT_DIR, global_step=epoch)

with tf.device('/cpu:0'):
    X = tf.placeholder(tf.int32, [None, sequence_len])
    Y = tf.placeholder(tf.int32, [None, word_num])

    word2vec = tf.Variable(wv.vectors, name="word2vec") #Variable
    embedded_chars = tf.nn.embedding_lookup(word2vec, X)
    embedded_chars_reshape = tf.reshape(embedded_chars, (-1, sequence_len, vector_per_word))

    cell = tf.nn.rnn_cell.LSTMCell(cell_num, state_is_tuple=True)  # GRUCell, RNNCell
    val, state = tf.nn.dynamic_rnn(cell, embedded_chars_reshape, dtype=tf.float32)

    # val = tf.reshape(val, (-1, time_ * cell_num))
    # last_val = val[:, -cell_num:] # shape = batch, cell_num
    # 위 두줄은 아래 코드와 의미가 같음
    last_val = state.h

    W = tf.get_variable('w', shape=[cell_num, word_num], initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.Variable(tf.constant(0.0, shape=[word_num]))
    output = tf.matmul(last_val, W) + bias

    sampled_softmax = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=tf.transpose(W),
                                                              biases = bias,
                                                              inputs = last_val,
                                                              num_classes=word_num,
                                                              num_sampled=15,
                                                              num_true=1,
                                                              labels=tf.reshape(tf.argmax(Y, 1), [-1, 1])))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    minimize = optimizer.minimize(sampled_softmax)

    correct_check = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1)), tf.int32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

train_accuracy_tensorboard = tf.placeholder(tf.float32)
train_loss_tensorboard = tf.placeholder(tf.float32)
valid_accuracy_tensorboard = tf.placeholder(tf.float32)
valid_loss_tensorboard = tf.placeholder(tf.float32)

train_accuracy_summary = tf.summary.scalar("train_accuracy", train_accuracy_tensorboard)
train_loss_summary = tf.summary.scalar("train_loss", train_loss_tensorboard)
valid_accuracy_summary = tf.summary.scalar("valid_accuracy", valid_accuracy_tensorboard)
valid_loss_summary = tf.summary.scalar("valid_loss", valid_loss_tensorboard)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./rnn_runs/', sess.graph)

# run
run()
