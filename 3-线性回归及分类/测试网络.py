# !/user/bin/env python
# -*- coding:utf-8 -*-
# author:Parker   time: 2018/7/31

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# # 批次的大小
# batch_size = 128
# n_batch = mnist.train.num_examples // batch_size
#
# x = tf.placeholder(tf.float32, [None,784])
# y = tf.placeholder(tf.float32, [None, 10])
#
# # 创建一个简单的神经网络
# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([1, 10]))
# prediction = tf.nn.softmax(tf.matmul(x,W) + b)
#
# # 代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
# # 梯度下降法
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#
# # 初始化变量
# init = tf.global_variables_initializer()
#
# # 得到一个布尔型列表，存放结果是否正确
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1)) #argmax 返回一维张量中最大值索引
#
# # 求准确率
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 把布尔值转换为浮点型求平均数
#
# with tf.Session() as sess:
#     sess.run(init)
#     for epoch in range(21):
#         for batch in range(n_batch):
#             # 获得批次数据
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#             sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
#         acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels})
#         print("Iter " + str(epoch) + " Testing Accuracy: " + str(acc))

#TODO 网络有问题，没有优化成功
batch_size = 256
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 第一层  784->200
W1 = tf.Variable(tf.truncated_normal([784,200],0,(784*200)**0.5),tf.float32)
# W1 = tf.Variable(tf.zeros([784, 200]))
b1 = tf.Variable(tf.zeros([1, 200]))
layer1 = tf.matmul(x, W1) + b1

# 第二层 200->10
W2 = tf.Variable(tf.truncated_normal([200,10],0,2000*0.5,tf.float32))
# W2 = tf.Variable(tf.zeros([200, 10]))
b2 = tf.Variable(tf.zeros([1, 10]))
output = tf.matmul(layer1, W2) + b2
prediction = tf.nn.softmax(output)

loss = tf.reduce_mean(tf.square(y - prediction))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ",Accuracy: " + str(acc))
