import os
import tensorflow as tf
from PIL import Image
from nets import nets_factory
import numpy as np

CHAR_SET = [str(i) for i in range(10)]

CHAR_SET_LEN = len(CHAR_SET)

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
BATCH_SIZE = 128
TFRECORD_FILE = "captcha/train.tfrecord"
EPOCH = 30
SAMPLE_NUM = 5800
LOOP_TIMES = EPOCH * SAMPLE_NUM // BATCH_SIZE


def read_and_decode(filename, batch_size=BATCH_SIZE, shuffle_batch=True):
    min_after_dequeue = 2000
    capacity = (min_after_dequeue + batch_size * 4) * 2
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'label0': tf.FixedLenFeature([], tf.int64),
                                           'label1': tf.FixedLenFeature([], tf.int64),
                                           'label2': tf.FixedLenFeature([], tf.int64),
                                           'label3': tf.FixedLenFeature([], tf.int64)
                                       })
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [224, 224])
    # 预处理
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    # 获取label
    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)
    if shuffle_batch:
        images, labels0, labels1, labels2, labels3 = tf.train.shuffle_batch([image, label0, label1, label2, label3],
                                                                            batch_size=batch_size,
                                                                            capacity=capacity,
                                                                            num_threads=1,
                                                                            min_after_dequeue=min_after_dequeue)
    else:
        images, labels0, labels1, labels2, labels3 = tf.train.batch([image, label0, label1, label2, label3],
                                                                    batch_size=batch_size,
                                                                    capacity=capacity,
                                                                    num_threads=1,
                                                                    min_after_dequeue=min_after_dequeue)
    # 返回图片及字符串形式的验证码，字符串会后面处理
    return images, labels0, labels1, labels2, labels3


x = tf.placeholder(tf.float32, [None, 224, 224])
y0 = tf.placeholder(tf.float32, [None])
y1 = tf.placeholder(tf.float32, [None])
y2 = tf.placeholder(tf.float32, [None])
y3 = tf.placeholder(tf.float32, [None])
lr = tf.Variable(0.003, dtype=tf.float32)

images_batch, labels0_batch, labels1_batch, labels2_batch, labels3_batch = read_and_decode(TFRECORD_FILE,
                                                                                           batch_size=BATCH_SIZE)
train_network_fn = nets_factory.get_network_fn('alexnet_v2_captcha_multi',
                                               num_classes=CHAR_SET_LEN,
                                               weight_decay=0.0005,
                                               is_training=True)

X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
logits0, logits1, logits2, logits3, end_points = train_network_fn(X)

# 这里的y0,y1,y2,y3 需要索引值，即上面的placeholder需要传入数字
one_hot_labels0 = tf.one_hot(indices=tf.cast(y0, tf.int32), depth=CHAR_SET_LEN)
one_hot_labels1 = tf.one_hot(indices=tf.cast(y1, tf.int32), depth=CHAR_SET_LEN)
one_hot_labels2 = tf.one_hot(indices=tf.cast(y2, tf.int32), depth=CHAR_SET_LEN)
one_hot_labels3 = tf.one_hot(indices=tf.cast(y3, tf.int32), depth=CHAR_SET_LEN)

# 计算loss
loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits0, labels=one_hot_labels0))
loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=one_hot_labels1))
loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=one_hot_labels2))
loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits3, labels=one_hot_labels3))

# 计算总的loss
total_loss = (loss0 + loss1 + loss2 + loss3) / 4.0
# 优化total_loss
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)

# 计算准确率
correct_prediction0 = tf.equal(tf.argmax(one_hot_labels0, 1), tf.argmax(logits0, 1))
accuracy0 = tf.reduce_mean(tf.cast(correct_prediction0, tf.float32))

correct_prediction1 = tf.equal(tf.argmax(one_hot_labels1, 1), tf.argmax(logits1, 1))
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

correct_prediction2 = tf.equal(tf.argmax(one_hot_labels2, 1), tf.argmax(logits2, 1))
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))

correct_prediction3 = tf.equal(tf.argmax(one_hot_labels3, 1), tf.argmax(logits3, 1))
accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners()

    for i in range(LOOP_TIMES):
        try:
            b_image, b_label0, b_label1, b_label2, b_label3 = sess.run(
                [images_batch, labels0_batch, labels1_batch, labels2_batch, labels3_batch])

            sess.run(optimizer, feed_dict={x: b_image,
                                           y0: b_label0,
                                           y1: b_label1,
                                           y2: b_label2,
                                           y3: b_label3})
            if i % 20 == 0:
                if i % 500 == 0:
                    sess.run(tf.assign(lr, lr*0.8))
                acc0, acc1, acc2, acc3, loss_ = sess.run([accuracy0, accuracy1, accuracy2, accuracy3, total_loss],
                                                         feed_dict={
                                                             x: b_image,
                                                             y0: b_label0,
                                                             y1: b_label1,
                                                             y2: b_label2,
                                                             y3: b_label3})
                learning_rate = sess.run(lr)
                Iter_epoch = i//(SAMPLE_NUM//EPOCH) + 1
                print("Iter: %d, Epoch: %d,  Loss: %.3f,  Accuracy: %.2f, %.2f, %.2f, %.2f, Learning_rate: %.5f" % (
                i, Iter_epoch, loss_, acc0, acc1, acc2, acc3, learning_rate))
                if acc0 > 0.9 and acc1 > 0.9 and acc2 > 0.9 and acc3 > 0.9:
                    if i % 200 == 0:
                        saver.save(sess, 'captcha/models/cap.model', global_step=i)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
    coord.request_stop()
    coord.join(threads)