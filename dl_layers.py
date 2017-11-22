import tensorflow as tf
import numpy as np

def doubleConvLayer(input1, input2, num_filters, kernel_height, kernel_width, name, in_channels = 1):
    with tf.name_scope(name):
        theta = tf.Variable(initial_value=tf.random_uniform([kernel_height, kernel_width, in_channels, num_filters], -1.0, 1.0),
                            dtype=tf.float32,
                            name="theta")
        b = tf.Variable(tf.zeros([num_filters]), name="biases")
        conv_1 = tf.nn.conv2d(input=input1, filter=theta, strides=[1, 1, 1, 1], padding="VALID")
        conv_2 = tf.nn.conv2d(input=input2, filter=theta, strides=[1, 1, 1, 1], padding="VALID")
        conv_1 = tf.nn.bias_add(conv_1, b)
        conv_2 = tf.nn.bias_add(conv_2, b)
        return tf.nn.elu(conv_1), tf.nn.elu(conv_2)

def doubleConvLayer_sigmoid(input1, input2, num_filters, kernel_height, kernel_width, name, in_channels=1):
    with tf.name_scope(name):
        theta = tf.Variable(
            initial_value=tf.random_uniform([kernel_height, kernel_width, in_channels, num_filters], -1.0, 1.0),
            dtype=tf.float32,
            name="theta")
        b = tf.Variable(tf.zeros([num_filters]), name="biases")
        conv_1 = tf.nn.conv2d(input=input1, filter=theta, strides=[1, 1, 1, 1], padding="VALID")
        conv_2 = tf.nn.conv2d(input=input2, filter=theta, strides=[1, 1, 1, 1], padding="VALID")
        conv_1 = tf.nn.bias_add(conv_1, b)
        conv_2 = tf.nn.bias_add(conv_2, b)
        return tf.nn.sigmoid(conv_1), tf.nn.sigmoid(conv_2)


def doubleLayer(input1, input2, hidden_nodes, name):
    with tf.name_scope(name):
        input_size = int(input1.get_shape()[1])
        theta = tf.Variable(initial_value=tf.random_uniform([input_size, hidden_nodes], -1.0, 1.0),
                            dtype=tf.float32,
                            name="theta")
        b = tf.Variable(tf.zeros([hidden_nodes]), name="biases")
        layer1 = tf.matmul(input1, theta) + b
        layer2 = tf.matmul(input2, theta) + b
        return tf.nn.elu(layer1), tf.nn.elu(layer2)

def doubleLayer_nc(input1, input2, hidden_nodes, name, features_per_step):
    with tf.name_scope(name):
        input_size = int(input1.get_shape()[1])
        theta = tf.constant(value=noncompensatory_weights(input_size, hidden_nodes, features_per_step),
                            shape=([input_size, hidden_nodes]), name = "theta")
        b = tf.Variable(tf.zeros([hidden_nodes]), name="biases")
        theta = tf.cast(theta, tf.float32)
        layer1 = tf.matmul(input1, theta) + b
        layer2 = tf.matmul(input2, theta) + b
        return tf.nn.elu(layer1), tf.nn.elu(layer2)

def doubleLayerBinary(input1, input2, hidden_nodes, name):
    with tf.name_scope(name):
        input_size = int(input1.get_shape()[1])
        theta = tf.Variable(initial_value=tf.random_uniform([input_size, hidden_nodes], -1.0, 1.0),
                            dtype=tf.float32,
                            name="theta")
        b = tf.Variable(tf.zeros([hidden_nodes]), name="biases")
        layer1 = tf.matmul(input1, theta) + b
        layer2 = tf.matmul(input2, theta) + b
        return tf.nn.elu(layer1/tf.abs(layer1)), tf.nn.elu(layer2/tf.abs(layer2))


def doubleLayer_sigmoid(input1, input2, hidden_nodes, name):
    with tf.name_scope(name):
        input_size = int(input1.get_shape()[1])
        theta = tf.Variable(initial_value=tf.random_uniform([input_size, hidden_nodes], -1.0, 1.0),
                            dtype=tf.float32,
                            name="theta")
        b = tf.Variable(tf.zeros([hidden_nodes]), name="biases")
        layer1 = tf.matmul(input1, theta) + b
        layer2 = tf.matmul(input2, theta) + b
        return tf.nn.sigmoid(layer1), tf.nn.sigmoid(layer2)


def double_to_singleLayer(input_1, input_2, hidden_nodes, name):
    with tf.name_scope(name):
        input_size = int(input_1.get_shape()[1])
        theta = tf.Variable(initial_value=tf.random_uniform([input_size, hidden_nodes], -1.0, 1.0),
                            dtype=tf.float32,
                            name="theta")
        b = tf.Variable(tf.zeros([hidden_nodes]), name="biases")
        layer1 = tf.matmul(input_1, theta) + b
        layer2 = tf.matmul(input_2, -theta) - b
        return (layer1) + (layer2), layer1

def double_to_singleLayer_ew(input_1, input_2, hidden_nodes, name):
    with tf.name_scope(name):
        input_size = int(input_1.get_shape()[1])
        theta = tf.ones([input_size, hidden_nodes])
        layer1 = tf.matmul(input_1, theta)
        layer2 = tf.matmul(input_2, -theta)
        return (layer1) + (layer2), layer1

def noncompensatory_weights(length, height, features_per_step):
    weights = []
    w = 0.000025
    for i in range(0, length):
        weights.append(w)
        if((i + 1) % features_per_step == 0):
            w = w *2
    weights_t = np.array(weights)
    for i in range(1, height):
        weights_t = np.concatenate((weights_t, np.array(weights)))
    return weights_t.reshape(-1, length)

def double_to_singleLayer_nc(input_1, input_2, hidden_nodes, name, features_per_step):
    with tf.name_scope(name):
        input_size = int(input_1.get_shape()[1])
        theta = tf.constant(value = noncompensatory_weights(input_size, hidden_nodes, features_per_step),  shape = ([input_size, hidden_nodes]))
        theta = tf.cast(theta, tf.float32)
        layer1 = tf.matmul(input_1, theta)
        layer2 = tf.matmul(input_2, -theta)
        return (layer1) + (layer2), layer1, layer2

def double_to_singleLayer_binary(input_1, input_2, hidden_nodes, name, features_per_step):
    with tf.name_scope(name):
        input_size = int(input_1.get_shape()[1])
        theta = tf.constant(value=noncompensatory_weights(input_size, hidden_nodes, features_per_step),
                            shape=([input_size, hidden_nodes]))
        theta = tf.cast(theta, tf.float32)
        compared = (input_1 - input_2) / tf.abs(input_1 - input_2)
        compared = tf.where(tf.is_nan(compared), tf.zeros_like(compared), compared)
        return tf.matmul(compared, theta)

def double_to_singleLayer_sigmoid(input_1, input_2, hidden_nodes, name):
    with tf.name_scope(name):
        input_size = int(input_1.get_shape()[1])
        theta = tf.Variable(initial_value=tf.random_uniform([input_size, hidden_nodes], -1.0, 1.0),
                            dtype=tf.float32,
                            name="theta")
        b = tf.Variable(tf.zeros([hidden_nodes]), name="biases")
        layer1 = tf.matmul(input_1, theta) + b
        layer2 = tf.matmul(input_2, -theta) - b
        return tf.nn.sigmoid(layer1) + tf.nn.sigmoid(layer2), tf.nn.sigmoid(layer1)


def singleLayer(input, hidden_nodes, name):
    with tf.name_scope(name):
        input_size = int(input.get_shape()[1])
        theta = tf.Variable(initial_value=tf.random_uniform([input_size, hidden_nodes], -1.0, 1.0),
                            dtype=tf.float32,
                            name="theta")
        b = tf.Variable(tf.zeros([hidden_nodes]), name="biases")
        layer1 = tf.matmul(input, theta) + b
        return tf.nn.elu(layer1)

def singleLayer(input, hidden_nodes, name):
    with tf.name_scope(name):
        input_size = int(input.get_shape()[1])
        theta = tf.Variable(initial_value=tf.random_uniform([input_size, hidden_nodes], -1.0, 1.0),
                            dtype=tf.float32,
                            name="theta")
        b = tf.Variable(tf.zeros([hidden_nodes]), name="biases")
        layer1 = tf.matmul(input, theta) + b
        return tf.nn.relu(layer1)