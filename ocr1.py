# -*- coding: utf-8 -*-
"""
tf CNN+LSTM+CTC 训练识别不定长数字字符图片
@author: zhujie
"""


import dataset

import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


#定义一些常量
#图片大小，32 x 256
OUTPUT_SHAPE = (32,256)

#训练最大轮次
num_epochs = 10000

num_hidden = 64
num_layers = 2 #1

#obj = gen_id_card()

num_classes = len(dataset.DICT) + 1 #+ 1  # 10位数字 + blank + ctc blank

#初始化学习速率
INITIAL_LEARNING_RATE = 1e-2#3
DECAY_STEPS = 5000
REPORT_STEPS = 100
LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor
MOMENTUM = 0.9

DIGITS='0123456789'
BATCHES = 10
BATCH_SIZE = 64
TRAIN_SIZE = BATCHES * BATCH_SIZE


def decode_sparse_tensor(sparse_tensor):
    rows = list()
    current_row = list()
    current_row_num = 0
    for i, row in enumerate(sparse_tensor[0]):
        row_num = row[0]
        if row_num != current_row_num:
            rows.append(current_row)
            current_row = list()
            current_row_num = row_num
        current_row.append(DIGITS[sparse_tensor[1][i]])
    rows.append(current_row)
    return rows

def report_accuracy(decoded_list, test_targets):
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)
    true_numer = 0

    if len(original_list) != len(detected_list):
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
              " test and detect length desn't match")
        return
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        if hit:
            true_numer = true_numer + 1
    print("Test Accuracy:", true_numer * 1.0 / len(original_list))

#转化一个序列列表为稀疏矩阵
def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), xrange(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64) #此shape应与sequences相同吧？


    return indices, values, shape


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.5)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride=(1, 1), padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1],padding=padding)

def max_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],strides=[1, stride[0], stride[1], 1], padding='SAME')

def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1],strides=[1, stride[0], stride[1], 1], padding='SAME')


#定义CNN网络，处理图片，
def convolutional_layers():
    #输入数据，shape [batch_size, 32, 256, 3]
    inputs = tf.placeholder(tf.float32, [BATCH_SIZE, OUTPUT_SHAPE[0], OUTPUT_SHAPE[1], 3])

    #第一层卷积层, 32*256*3 => 16*128*48
    W_conv1 = weight_variable([5, 5, 3, 48])
    b_conv1 = bias_variable([48])
    #x_expanded = tf.expand_dims(inputs, 3)
    x_expanded = inputs
    h_conv1 = tf.nn.relu(conv2d(x_expanded, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2))

    #return inputs, h_pool1
    #第二层, 16*128*48 => 16*64*64
    W_conv2 = weight_variable([5, 5, 48, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2, ksize=(2, 1), stride=(2, 1))

    #第三层, 16*64*64 => 8*32*128
    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool(h_conv3, ksize=(2, 2), stride=(2, 2))

    return inputs, h_pool3

    #全连接
    W_fc1 = weight_variable([16 * 8 * OUTPUT_SHAPE[1], OUTPUT_SHAPE[1]])
    b_fc1 = bias_variable([OUTPUT_SHAPE[1]])

    conv_layer_flat = tf.reshape(h_pool3, [-1, 16 * 8 * OUTPUT_SHAPE[1]])

    features = tf.nn.relu(tf.matmul(conv_layer_flat, W_fc1) + b_fc1)
    #（batchsize,256）
    shape = tf.shape(features)
    features = tf.reshape(features, [shape[0], OUTPUT_SHAPE[1], 1])  # batchsize * outputshape * 1
    return inputs,features

def get_train_model():
    keep_prob = tf.placeholder(tf.float32)
    #features = convolutional_layers()
    #print features.get_shape()

    # (batch_size, 256, 32)
    # inputs = tf.placeholder(tf.float32, [None, None, OUTPUT_SHAPE[0]])

    # (batch_size,8,32,128)
    inputs,cnn_outputs = convolutional_layers()
    shape = cnn_outputs.get_shape().as_list() # [batch_size,height,width,features]
    print('cnn_outputs shape:'+str(shape))
    #定义ctc_loss需要的稀疏矩阵
    targets = tf.sparse_placeholder(tf.int32)

    #1维向量 序列长度 [batch_size,]
    #seq_len = tf.placeholder(tf.int32, [None])
    seq_len = tf.fill([shape[0]], shape[2])

    transposed = tf.transpose(cnn_outputs, perm=[0,2,1,3])  #[batch_size,width,height,features]
    print('cnn_output transposed shape:'+str(transposed.get_shape().as_list()))

    rnn_inputs = tf.reshape(transposed,[shape[0],shape[2],shape[1]*shape[3]])
    print('rnn_inputs shape:'+str(rnn_inputs.get_shape().as_list()))

    #定义LSTM网络
    #cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    #stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)  # 没使用的变量
    cells = [tf.contrib.rnn.DropoutWrapper( tf.contrib.rnn.GRUCell(num_hidden),keep_prob) for _ in range(num_layers)]
    stack = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)  # 没使用的变量
    outputs, _ = tf.nn.dynamic_rnn(stack, rnn_inputs, seq_len, dtype=tf.float32)  # outputs 形状与 inputs 相同
    #outputs = tf.nn.dropout(outputs, keep_prob)

    shape = rnn_inputs.get_shape().as_list()
    batch_s, max_timesteps = shape[0], shape[1]

    outputs = tf.reshape(outputs, [-1, num_hidden])  #(batch_size*256,64)
    W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1), name="W")  #(64,12)
    b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")

    logits = tf.matmul(outputs, W) + b  # (batch_size*256,12) 概率分布值

    logits = tf.reshape(logits, [batch_s, -1, num_classes])  # (batch_size,256,12)
    logits = tf.transpose(logits, (1, 0, 2))  # (256, batch_size,12)
    return logits, inputs, targets, seq_len, W, b,keep_prob

def train(ds):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                               global_step,
                                               DECAY_STEPS,
                                               LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)
    logits, inputs, targets, seq_len, W, b,keep_prob = get_train_model()

    loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)

    #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=MOMENTUM).minimize(cost, global_step=global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)  # decode包含标签值（应该是seq_len的，包含重复标签）

    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    init = tf.global_variables_initializer()

    def do_report():
        test_inputs, test_targets, _, test_seq_len = ds.get_next_batch(BATCH_SIZE, gray_scale=False, transpose=False)
        test_feed = {inputs: test_inputs,
                     targets: test_targets,
                     seq_len: test_seq_len,
                     keep_prob: 1.0}
        dd, log_probs, accuracy = session.run([decoded[0], log_prob, acc], test_feed)
        report_accuracy(dd, test_targets)
        # decoded_list = decode_sparse_tensor(dd)

    def do_batch():
        train_inputs, train_targets, _, train_seq_len = ds.get_next_batch(BATCH_SIZE,gray_scale=False, transpose=False)
        feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len,keep_prob:0.75}

        b_loss, b_targets, b_logits, b_seq_len, b_cost, steps, ded, _ = session.run([loss, targets, logits, seq_len, cost, global_step, decoded, optimizer], feed)

        #print b_loss
        #print b_targets, b_logits, b_seq_len
        #print b_cost, steps
        if steps > 0 and steps % REPORT_STEPS == 0:
            do_report()
            #save_path = saver.save(session, "ocr.model", global_step=steps)
            # print(save_path)
        return b_cost, steps

    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        for curr_epoch in range(num_epochs):
            print("Epoch.......", curr_epoch)
            train_cost = train_ler = 0
            for batch in range(BATCHES):
                start = time.time()
                c, steps = do_batch()
                train_cost += c * BATCH_SIZE
                seconds = time.time() - start
                print("Step:", steps, ", batch seconds:", seconds)

            train_cost /= TRAIN_SIZE

            train_inputs, train_targets, _, train_seq_len = ds.get_next_batch(BATCH_SIZE,gray_scale=False, transpose=False)
            val_feed = {inputs: train_inputs,
                        targets: train_targets,
                        seq_len: train_seq_len,
                        keep_prob: 1.0}

            val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)

            log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}s, learning_rate = {}"
            print(log.format(curr_epoch + 1, num_epochs, steps, train_cost, train_ler, val_cost, val_ler, time.time() - start, lr))


if __name__ == '__main__':
    print(decode_sparse_tensor([[[0,0],[2,2]],[0,1,2,3,4,5,6],[0]]))

    ds = dataset.DataSet('../dataline')
    inputs, sparse_targets, labels, seq_len = ds.get_next_batch(2, gray_scale=False, transpose=False) #get_next_batch(2)
    print(inputs.shape)
    results = decode_sparse_tensor(sparse_targets)
    print(results)

    train(ds)