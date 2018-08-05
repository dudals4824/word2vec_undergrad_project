import tensorflow as tf
from cnnClassifier import cnnUtils

tf.set_random_seed(777)

learning_rate = 0.001
training_epochs = 20
batch_size = 16

keep_prob = tf.placeholder(tf.float32)

# 이전에 데이터 들어가는 부분이 있어야함.
# 여기서 None은 몇 라인이 들어갈지 정해질거임
X = tf.placeholder(tf.float32, [batch_size, 11, 300], name="x_input")
# X를 받는대로 reshape를 하면 단어 11개에 대해서 300차원으로 표현되고
# -1은 개수로 들어갈거임 None이랑 비슷한 의미
X_reshape = tf.reshape(X, [-1, 11, 300, 1])

# 우리가 softmax하고 싶은 종류는 총 50095개
# 답이 50095개니까 50095개로
Y = tf.placeholder(tf.float32, [None, 50095], name="y_input")

# filter_size가 3, 300차원, color는 1, layer의 개수 32개
with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([3, 300, 1, 32], stddev=0.01), name="Weight1")

    L1 = tf.nn.conv2d(X_reshape, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 3, 300, 1], strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    W1_hist = tf.summary.histogram("Weight1", W1)
    L1_hist = tf.summary.histogram("Layer1", L1)

with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal([4, 300, 32, 64], stddev=0.01), name="Weight2")
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 4, 300, 1], strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

    W2_hist = tf.summary.histogram("Weight2", W2)
    L2_hist = tf.summary.histogram("Layer2", L2)

with tf.name_scope("layer3") as scope:
    W3 = tf.Variable(tf.random_normal([5, 300, 64, 128], stddev=0.01), name="Weight3")
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 4, 300, 1], strides=[1, 2, 2, 1], padding='SAME')
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

    W3_hist = tf.summary.histogram("Weight3", W3)
    L3_hist = tf.summary.histogram("Layer3", L3)

    L3_flat = tf.reshape(L3, [-1, 2*38*128])

with tf.name_scope("layer4") as scope:
    W4 = tf.get_variable("W4", shape=[2*38*128, 625],
                     initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([625]), name="Bias")
    L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
    L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

    b4_hist = tf.summary.histogram("Bias4", b4)
    L4_hist = tf.summary.histogram("Layer4", L4)

with tf.name_scope("layer5") as scope:
    W5 = tf.get_variable("W5", shape=[625, 50095],
                     initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([50095]), name="Bias5")
    logits = tf.matmul(L4, W5) + b5

    b5_hist = tf.summary.histogram("Bias5", b5)
    logits_hist = tf.summary.histogram("Logits", logits)


with tf.name_scope("cost/loss") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

with tf.name_scope("train") as scope:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
# tensorboard --logdir=./logs/xor_logs
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/cnn_logs")
writer.add_graph(sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

print("Learning started, It takes long.")
for epoch in range(training_epochs):
    avg_cost = 0
    start = 0
    # 임의로 초기화
    train_len = 32
    total_batch = int(train_len / batch_size)
    for i in range(0, total_batch):
        start, batch_xs, batch_ys = cnnUtils.next_batch(start, batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.5}
        summary, c, _ = sess.run([merged_summary, cost, optimizer], feed_dict=feed_dict)
        writer.add_summary(summary, global_step=i)
        avg_cost += c / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

saver.save(sess, './runs')
print("Learning Finished!")


start = 0
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
start, batch_xs, batch_ys = cnnUtils.next_batch(start, batch_size)
print('Accuracy:', sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1}))
