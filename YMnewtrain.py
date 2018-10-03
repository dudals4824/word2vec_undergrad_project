import tensorflow as tf
import pandas as pd
from gensim.models import Word2Vec
import random
import math
import numpy as np

model = Word2Vec.load("./data/subsampled_wvModel")
wv = model.wv
learning_rate = 0.001
epoch_num = 101
cell_num = 128
sequence_len = 10
word_num = 14920
vector_per_word = 128
num_filters = 128
batch_size = 64

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
            y = np.zeros(14920, dtype=np.int32)
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
            y = np.zeros(14920, dtype=np.int32)
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
    while True:
        x_batch, y_batch, start_idx = next_batch(_traindf, _dic, start_idx)
        if len(x_batch) == batch_size:
            train_step += 1
            print(train_step)
        elif len(x_batch) < batch_size:
            break
        feed_dict = {X: x_batch, Y: y_batch}
        train_accuracy, train_loss, _ = sess.run([correct_check, sampled_softmax, minimize], feed_dict)

        loss += train_loss
        correct += train_accuracy

    return correct / (batch_size * train_step), loss

def validation(_validationdf, _dic):
    loss = 0
    correct = 0
    start_idx = 0
    dev_step = 0
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

    return correct / (batch_size * dev_step + 1), loss

def run():
    traindf = pd.read_csv("./data/newtrain.csv")
    validationdf = pd.read_csv("./data/validation.csv")
    # frequency table : S
    f = open("./data/text8");
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
        print("train accuracy | loss: " + str(train_accuracy) + " | " + str(train_loss))
        print("validation accruacy | loss: " + str(valid_accuracy) + " | " + str(valid_loss))

        summary = sess.run(merged, {train_accuracy_tensorboard: train_accuracy,
                                    train_loss_tensorboard: train_loss,
                                    valid_accuracy_tensorboard: valid_accuracy,
                                    valid_loss_tensorboard: valid_loss})
        writer.add_summary(summary, epoch)

with tf.device('/cpu:0'), tf.name_scope("embedding"):
    X = tf.placeholder(tf.int32, [None, sequence_len], name="X")
    Y = tf.placeholder(tf.int32, [None, word_num], name="Y")
    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    word2vec = tf.constant(wv.vectors, name="word2vec")
    embedded_chars = tf.nn.embedding_lookup(word2vec, X)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

    pooled_outputs = []
with tf.name_scope("CNN_filter_size_3"):
    # CNN filter size 3
    filter_shape = [3, 128, 1, 128]
    W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W1")
    b1 = tf.Variable(tf.constant(0.1, shape=[128]), name="b1")
    conv1 = tf.nn.conv2d(embedded_chars_expanded, W1, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
    h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu1")
    maxpool1 = tf.nn.max_pool(h1, ksize=[1, 8, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="maxpool1")
    pooled_outputs.append(maxpool1)
with tf.name_scope("CNN_filter_size_4"):
    # CNN filter size 4
    filter_shape = [4, 128, 1, 128]
    W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W2")
    b2 = tf.Variable(tf.constant(0.1, shape=[128]), name="b2")
    conv2 = tf.nn.conv2d(embedded_chars_expanded, W2, strides=[1, 1, 1, 1], padding="VALID", name="conv2")
    h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")
    maxpool2 = tf.nn.max_pool(h2, ksize=[1, 7, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="maxpool2")
    pooled_outputs.append(maxpool2)
with tf.name_scope("CNN_filter_size_5"):
    # CNN filter size 5
    filter_shape = [5, 128, 1, 128]
    W3 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W3")
    b3 = tf.Variable(tf.constant(0.1, shape=[128]), name="b3")
    conv3 = tf.nn.conv2d(embedded_chars_expanded, W3, strides=[1, 1, 1, 1], padding="VALID", name="conv3")
    h3 = tf.nn.relu(tf.nn.bias_add(conv3, b3), name="relu3")
    maxpool3 = tf.nn.max_pool(h3, ksize=[1, 6, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="maxpool3")
    pooled_outputs.append(maxpool3)

num_filters_total = num_filters * 3
h_pool = tf.concat(pooled_outputs, 3)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

with tf.name_scope("dropout"):
    h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

with tf.name_scope("output"):
    sampled_W = tf.get_variable("sampled_W", shape=[num_filters_total, word_num],
                                initializer=tf.contrib.layers.xavier_initializer())
    sampled_b = tf.Variable(tf.constant(0.0, shape=[word_num]), name="sampled_b")

    output = tf.matmul(h_drop, sampled_W) + sampled_b

with tf.name_scope("loss/cost"):
    sampled_softmax = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=tf.transpose(sampled_W),
                                                                biases=sampled_b,
                                                                inputs=h_drop,
                                                                num_classes=14920,
                                                                num_sampled=15,
                                                                num_true=1,
                                                                labels=tf.reshape(tf.argmax(Y, 1), [-1, 1])))

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    minimize = optimizer.minimize(sampled_softmax)

with tf.name_scope("accuracy"):
    correct_check = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1)), tf.int32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_accuracy_tensorboard = tf.placeholder(tf.float32)
train_loss_tensorboard = tf.placeholder(tf.float32)
valid_accuracy_tensorboard = tf.placeholder(tf.float32)
valid_loss_tensorboard = tf.placeholder(tf.float32)

train_accuracy_summary = tf.summary.scalar("train_accuracy", train_accuracy_tensorboard)
train_loss_summary = tf.summary.scalar("train_loss", train_loss_tensorboard)
valid_accuracy_summary = tf.summary.scalar("valid_accuracy", valid_accuracy_tensorboard)
valid_loss_summary = tf.summary.scalar("valid_loss", valid_loss_tensorboard)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./cnn_runs/', sess.graph)

# run
run()