import tensorflow as tf
from tensorflow import Variable
from tensorflow import placeholder


def weight_variable(shape):
    """
    :param shape: size_w * size_h * ch_in * ch_out
    :return:
    """
    init = tf.truncated_normal(
        shape=shape, dtype=tf.float32, stddev=0.01)
    return Variable(init)


def bias_variable(shape):
    """
    :param shape: [ch]
    :return:
    """
    init = tf.constant(
        0.01, dtype=tf.float32, shape=shape)
    return Variable(init)


def conv2d(x, W, stride):
    """
    :param x:
    :param W:
    :param stride:
    :return:
    """
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME") # SAME: / stride


def max_pool_22(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def graph():
    # input
    s = placeholder(tf.float32, shape=[None, 80, 80, 4])

    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_22(h_conv1)

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_pool2 = max_pool_22(h_conv2)

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 1) + b_conv3)
    h_pool3 = max_pool_22(h_conv3)

    h_conv2_flan = tf.reshape(h_pool3, [-1, 2 * 2 * 64])

    W_fc1 = weight_variable([2 * 2 * 64, 16])
    b_fc1 = bias_variable([16])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flan, W_fc1) + b_fc1)

    W_q = weight_variable([16, 2])
    b_q = bias_variable([2])
    q = tf.matmul(h_fc1, W_q) + b_q

    return s, q






