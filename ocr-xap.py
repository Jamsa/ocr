# -*- coding: utf-8 -*-
"""
tf CNN+LSTM+CTC 训练识别不定长数字字符图片
@author: zhujie
"""

# import dataset
import gen_image
import numpy as np
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# 训练最大轮次
num_epochs = 10000

num_hidden = 64
num_layers = 2  # 1

# obj = gen_id_card()

num_classes = len(gen_image.char_set) + 1  # + 1  # 10位数字 + blank + ctc blank

# 初始化学习速率
INITIAL_LEARNING_RATE = 1e-2  # 3
DECAY_STEPS = 5000
REPORT_STEPS = 100
LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor
MOMENTUM = 0.9


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.5)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride=(1, 1), padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding=padding)


def max_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1], strides=[1, stride[0], stride[1], 1], padding='SAME')


def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1], strides=[1, stride[0], stride[1], 1], padding='SAME')


# 定义CNN网络，处理图片，
def convolutional_layers(inputs):
    # 第一层卷积层, 32*256*3 => 16*128*48
    W_conv1 = weight_variable([5, 5, 3, 48])
    b_conv1 = bias_variable([48])
    # x_expanded = tf.expand_dims(inputs, 3)
    h_conv1 = tf.nn.relu(conv2d(inputs, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2))

    # return inputs, h_pool1
    # 第二层, 16*128*48 => 16*64*64
    W_conv2 = weight_variable([5, 5, 48, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2, ksize=(2, 1), stride=(2, 1))

    # 第三层, 16*64*64 => 8*32*128
    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool(h_conv3, ksize=(2, 2), stride=(2, 2))

    return h_pool3


def get_train_model(inputs, keep_prob):
    print(inputs.get_shape())
    cnn_outputs = convolutional_layers(inputs)

    N, H, W, C = cnn_outputs.get_shape().as_list()  # [batch_size,height,width,features]
    print('cnn_outputs shape:' + str(cnn_outputs.get_shape().as_list()))
    N = tf.shape(cnn_outputs)[0]
    W = tf.shape(cnn_outputs)[2]

    # 1维向量 序列长度 [batch_size,]
    seq_len = tf.fill([N], W)

    transposed = tf.transpose(cnn_outputs, perm=[0, 2, 1, 3])  # [batch_size,width,height,features]
    print('cnn_output transposed shape:' + str(transposed.get_shape().as_list()))

    rnn_inputs = tf.reshape(transposed, [N, W, H * C])
    print('rnn_inputs shape:' + str(rnn_inputs.get_shape().as_list()))

    # 定义LSTM网络
    # cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    # stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)  # 没使用的变量
    cells = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(num_hidden), keep_prob) for _ in range(num_layers)]
    stack = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)  # 没使用的变量
    outputs, _ = tf.nn.dynamic_rnn(stack, rnn_inputs, seq_len, dtype=tf.float32)  # outputs 形状与 inputs 相同
    # outputs = tf.nn.dropout(outputs, keep_prob)


    outputs = tf.reshape(outputs, [-1, num_hidden])  # (batch_size*256,64)
    W = tf.Variable(tf.truncated_normal([num_hidden, num_classes],
                                        stddev=0.1), name="W")  # (64,12)
    b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")

    logits = tf.matmul(outputs, W) + b  # (batch_size*256,12) 概率分布值
    logits = tf.reshape(logits, [N, -1, num_classes])  # (batch_size,256,12)
    logits = tf.transpose(logits, (1, 0, 2))  # (256, batch_size,12)
    return logits, seq_len


def train():
    keep_prob = tf.placeholder(tf.float32)
    # 输入数据，shape [batch_size, 32, 256, 3]
    img_height = 32
    img_channel = 3
    inputs = tf.placeholder(tf.float32, [None, img_height, None, img_channel])
    # 定义ctc_loss需要的稀疏矩阵co
    targets = tf.sparse_placeholder(tf.int32)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                               global_step,
                                               DECAY_STEPS,
                                               LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)
    logits, seq_len = get_train_model(inputs, keep_prob)

    loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)

    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=MOMENTUM).minimize(cost, global_step=global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len,
                                                      merge_repeated=False)  # decode包含标签值（应该是seq_len的，包含重复标签）

    init = tf.global_variables_initializer()

    with tf.Session() as sess:  # 启动创建的模型
        sess.run(init)
        for i in range(50000):  # 开始训练模型，循环训练5000次
            train_inputs, sparse_targets = gen_image.get_next_batch(16)
            feed_dict = {inputs: train_inputs, targets: sparse_targets, keep_prob: 1.0}
            sess.run([optimizer, global_step], feed_dict=feed_dict)  # 执行训练
            if i % 100 == 0:
                test_inputs, sparse_targets = gen_image.get_next_batch(1)
                feed_dict = {inputs: test_inputs, targets: sparse_targets, keep_prob: 1.0}
                loss_, decoded_, learning_rate_ = sess.run([cost, decoded, learning_rate], feed_dict=feed_dict)  # 执行训练
                print(loss_, learning_rate_)
                print(decoded_[0][1], sparse_targets[1])


if __name__ == '__main__':
    train()
