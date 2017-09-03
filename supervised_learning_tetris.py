import tensorflow as tf

import _thread
import pygame as pyg

import load_functions as lf


#Can take several minutes depending on size of input:
# lf.loadAndParseData("data/supervised_learning_tetris_10x10_training_small.csv", "data/parsed")
# lf.loadAndParseData("data/supervised_learning_tetris_10x10_test_small.csv", "data/parsed_test")
# lf.loadParsedReshapeAndFlatten("data/parsed", "data/train_input", "data/train_output")
# lf.loadParsedReshapeAndFlatten("data/parsed_test", "data/test_input", "data/test_output")

input = lf.pickleLoad("data/test_input")
output = lf.pickleLoad("data/test_output")
input_test = lf.pickleLoad("data/train_input")
output_test = lf.pickleLoad("data/train_output")

#data is now a flattened input of 1's and 0's and output is the values of each option
m, n = input.shape
m_test, n_test = input_test.shape

print("m",m)
print("n",n)
print("m_test",m_test)
print("n_test",n_test)

input_with_bias = np.float32(np.c_[np.ones((m,1)), input.reshape(m,n)])
input_test_with_bias = np.float32(np.c_[np.ones((m_test,1)), input_test.reshape(m_test,n)])

print("data ready")
#we start defining tensorflow model:
learning_rate = 0.001
n_epochs = 5000

print("x:",input_with_bias.shape)
print("y:",output.shape)

tf.constant(input_with_bias, dtype=tf.float32,  name = "x")
tf.constant(output, dtype=tf.float32,  name = "y")

a = tf.Variable(initial_value= tf.random_uniform([n+1, 3400], -1.0, 1.0), name = "a")
z = tf.Variable(initial_value= tf.random_uniform([n+1, 3400], -1.0, 1.0), name = "zeta")


theta = tf.Variable(initial_value= tf.random_uniform([n+1, 34], -1.0, 1.0), name = "theta")
y_pred = tf.matmul(input_with_bias, theta, name = "predictions")

error = y_pred - output
error_test = tf.matmul(input_test_with_bias, theta) - output_test

error_max = tf.cast(tf.reduce_max(y_pred, axis = 1), tf.float32) - tf.cast(tf.reduce_max(output, axis = 1), tf.float32)
error_max_test = tf.cast(tf.reduce_max(tf.matmul(input_test_with_bias, theta), axis = 1), tf.float32) - tf.cast(tf.reduce_max(output_test, axis = 1), tf.float32)

msemax = tf.reduce_mean(error_max, name = "msemax")
msemax_test = tf.reduce_mean(error_max_test, name = "msemax_test")

mse = tf.reduce_mean(tf.square(error), name = "mse")
mse_test = tf.reduce_mean(tf.square(error_test), name = "mse_test")

gradients = tf.gradients(msemax, [theta])[0]

optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = 0.9)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


def train():
    global sess
    # We run the session:
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            sess.run(training_op)
            if (epoch % 1000 == 0):
                print(y_pred.eval().shape)
                print("fit MSE:", mse.eval())
                print("fit MSE max:", msemax.eval())
                print("test MSE:", mse_test.eval())
                # print("test MSE max:", msemax_test.eval())
        save_path = saver.save(sess, "models/my_model_final.ckpt")


#train()

## DEEP NETWORK

from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_inputs = 3400
n_hidden1 = 1000
n_hidden2 = 300
n_hidden3 = 100
n_outputs = 34

n_filters = 10
channels = 34

# input_test = input_test.reshape([-1, 34, 10, 10])
input_test = np.reshape(np.ravel(input_test, order='F'), [-1, 10, 10, 34], order='F')


# X = tf.placeholder(dtype=tf.float32, shape = (None, n_inputs), name = "x")
X_conv = tf.placeholder(dtype=tf.float32, shape = (None, 10, 10, 34), name = "x")
y = tf.placeholder(dtype=tf.float32, shape = (None, n_outputs), name = "y")



def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X, W) + b
        if activation=="elu":
            return tf.nn.elu(z)
        else:
            return z

with tf.name_scope("dnn"):
    # Create n_filters filters of 5x5
    filters = tf.random_uniform([5, 5, 34, n_filters], -1, 1)
    print("filters shape:", filters.shape)
    convolution = tf.nn.conv2d(X_conv, filters, strides=[1, 1, 1, 1], padding="SAME")
    conv_flat = tf.nn.elu(tf.reshape(convolution, [-1, 1000]))
    hidden1 = neuron_layer(conv_flat, n_hidden1, name ="hidden1", activation="elu")
    hidden2 = neuron_layer(hidden1, n_hidden2, name ="hidden2", activation="elu")
    logits = neuron_layer(hidden2, n_outputs, name ="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(
                       labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

    learning_rate = 0.05
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    labels = tf.argmax(y, 1)
    correct = tf.nn.in_top_k(logits, labels, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 4000
batch_size = 100

accuracy_summary = tf.summary.scalar('accuracy', accuracy)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

def next_batch(batch_id):
    indices = range(batch_id, min(batch_id+batch_size, m))
    # x_batch = input[indices, :].reshape(-1, 10, 10, 34)
    x_batch = np.reshape(np.ravel(input[indices, :], order='F'), [-1, 10, 10, 34], order='F')
    y_batch = output[indices].reshape(-1,34)
    return x_batch, y_batch

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        batch_id = 0
        for iteration in range(m // batch_size):
                X_batch, y_batch = next_batch(batch_id)
                batch_id += 1
                sess.run(training_op, feed_dict={X_conv: X_batch, y: y_batch})
                if batch_id % 10 == 0:
                    summary_str = accuracy_summary.eval(feed_dict={X_conv: X_batch, y: y_batch})
                    step = epoch * m/batch_size + batch_id
                    file_writer.add_summary(summary_str, step)
        if(epoch % 2 == 0):
            batch_id = 0
            # for iteration in range(m // batch_size):
            #     X_batch, y_batch = next_batch(batch_id)
            acc_train = accuracy.eval(feed_dict={X_conv: X_batch, y: y_batch})
            #     batch_id += 1
            print(epoch, "batch ", batch_id, " train accuracy:", acc_train)
            acc_test = accuracy.eval(feed_dict={X_conv: input_test, y: output_test})
            print(epoch, "Test accuracy:", acc_test)
            # fil = filters.eval(feed_dict={X_conv: input_test, y: output_test})
            # for i in range(n_filters):
            #     for x in range(5):
            #         for y in range(5):
            #             print(fil[x,y,0,0], end="")
            #         print()

    save_path = saver.save(sess, "./my_model_final.ckpt")
file_writer.close()
