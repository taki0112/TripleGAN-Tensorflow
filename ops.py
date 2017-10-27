import tensorflow as tf
from tflearn import global_avg_pool
from tensorflow.contrib.layers import variance_scaling_initializer
import numpy as np

he_init = variance_scaling_initializer()

"""
weight normalization
Kaming he initialization -> weight_normalization -> it uses kernel_initialization
"""

def weight_norm(x, output_dim) :
    input_dim = int(x.get_shape()[-1])
    g = tf.get_variable('g_scalar', shape=[1], initializer=tf.ones_initializer())
    w = tf.get_variable('weight', shape=[input_dim, output_dim], initializer=he_init)
    w_init = g * tf.nn.l2_normalize(w, dim=[0, 1])  # SAME dim=1

    return w_init

def conv_layer(x, filter_size, kernel, stride=1, padding='SAME', wn=True, layer_name="conv"):
    with tf.name_scope(layer_name):
        if wn:
            w_init = weight_norm(x, filter_size)
            x = tf.layers.conv2d(inputs=x, filters=filter_size, kernel_size=kernel, kernel_initializer=w_init, strides=stride, padding=padding)
        else :
            x = tf.layers.conv2d(inputs=x, filters=filter_size, kernel_size=kernel, kernel_initializer=he_init, strides=stride, padding=padding)
        return x


def deconv_layer(x, filter_size, kernel, stride=1, padding='SAME', wn=False, layer_name='deconv'):
    with tf.name_scope(layer_name):
        if wn :
            w_init = weight_norm(x, filter_size)
            x = tf.layers.conv2d_transpose(inputs=x, filters=filter_size, kernel_size=kernel, kernel_initializer=w_init, strides=stride, padding=padding)
        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=filter_size, kernel_size=kernel, kernel_initializer=he_init, strides=stride, padding=padding)
        return x


def linear(x, unit, wn=True, layer_name='linear'):
    with tf.name_scope(layer_name):
        if wn :
            w_init = weight_norm(x, unit)
            x = tf.layers.dense(inputs=x, units=unit, kernel_initializer=w_init)
        else :
            x = tf.layers.dense(inputs=x, units=unit, kernel_initializer=he_init)
        return x


def nin(x, unit, wn=True, layer_name='nin'):
    # https://github.com/openai/weightnorm/blob/master/tensorflow/nn.py
    with tf.name_scope(layer_name):
        s = list(map(int, x.get_shape()))
        x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
        x = linear(x, unit, wn, layer_name)
        x = tf.reshape(x, s[:-1] + [unit])


        return x


def gaussian_noise_layer(x, std=0.15):
    noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=std, dtype=tf.float32)
    return x + noise

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')


def max_pooling(x, kernel, stride):
    return tf.layers.max_pooling2d(x, pool_size=kernel, strides=stride, padding='VALID')


def flatten(x):
    return tf.contrib.layers.flatten(x)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def sigmoid(x):
    return tf.nn.sigmoid(x)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.nn.tanh(x)


def conv_concat(x, y):
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()

    return concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], axis=3)


def concat(x, axis=1):
    return tf.concat(x, axis=axis)


def reshape(x, shape):
    return tf.reshape(x, shape=shape)


def batch_norm(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)


def dropout(x, rate, is_training):
    return tf.layers.dropout(inputs=x, rate=rate, training=is_training)
