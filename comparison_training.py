import load_functions as lf
import tensorflow as tf
import numpy as np

# lf.loadAndParseData("data/supervised_learning_tetris_10x10_training_small.csv", "data/parsed")
# print("turning into pairwise, this may take a while ...")
# lf.loadParsedAndMakePairwiseComparisons("data/parsed", "data/pairwise")

pw_full = lf.pickleLoad("data/pairwise")
pw_all = np.array_split(pw_full, 2)
pw = pw_all[0]
pw_test = pw_all[1]

total_size = len(pw)
n = 1000

labels = np.float32(pw[:,2])

boardsize_flat = 100

x = tf.placeholder(dtype=tf.float32, shape = (None, boardsize_flat*2 +1), name = "x")
# x2 = tf.placeholder(dtype=tf.float32, shape = (None, boardsize_flat +1),  name = "x2")
y = tf.placeholder(dtype=tf.float32, shape = (None, 1),  name = "y")

# theta1 = tf.Variable(initial_value= tf.random_uniform([boardsize_flat + 1, n], -1.0, 1.0), dtype= tf.float32, name = "theta1")

def doubleConvLayer(input1, input2, filterwidth, filterheight, channels, name):
    with tf.name_scope(name):
        #todo
        return;

def doubleLayer(input1, input2, hidden_nodes, name):
    with tf.name_scope(name):
        input_size = int(input1.get_shape()[1])
        theta = tf.Variable(initial_value=tf.random_uniform([input_size, hidden_nodes], -1.0, 1.0), dtype=tf.float32,
                             name="theta")
        # theta_2 = tf.Variable(initial_value=tf.random_uniform([input_size, hidden_nodes], -1.0, 1.0), dtype=tf.float32,
        #                     name="theta2")
        b = tf.Variable(tf.zeros([hidden_nodes]), name="biases")
        # b_2 = tf.Variable(tf.zeros([hidden_nodes]), name="biases_2")
        layer1 = tf.matmul(input1, theta) + b
        layer2 = tf.matmul(input2, theta) + b
        return tf.nn.relu(layer1), tf.nn.relu(layer2)

def singleLayer(input, hidden_nodes, name):
    with tf.name_scope(name):
        input_size = int(input.get_shape()[1])
        theta = tf.Variable(initial_value=tf.random_uniform([input_size, hidden_nodes], -1.0, 1.0),
                            dtype=tf.float32,
                            name="theta")
        # theta_2 = tf.Variable(initial_value=tf.random_uniform([input_size, hidden_nodes], -1.0, 1.0), dtype=tf.float32,
        #                     name="theta2")
        b = tf.Variable(tf.zeros([hidden_nodes]), name="biases")
        # b_2 = tf.Variable(tf.zeros([hidden_nodes]), name="biases_2")
        layer1 = tf.matmul(input, theta) + b
        return tf.nn.relu(layer1)

# l1_1,l2_1 = doubleLayer(x1, x2, 100, "l1")
# l1_2,l2_2 = doubleLayer(l1_1,l2_1, 50, "l2")
# l1_f,l2_f = doubleLayer(l1_1,l2_1, 10, "lf")

l1 = singleLayer(x, 100, "l1")
l2 = singleLayer(l1, 50, "l2")
# l3 = singleLayer(l2, 50, "l3")
# l4 = singleLayer(l3, 50, "l4")
# l5 = singleLayer(l4, 50, "l5")
lf = singleLayer(l2, 10, "l6")

input_size = int(lf.get_shape()[1])
theta_f = tf.Variable(initial_value= tf.random_uniform([input_size, 1], -1.0, 1.0), dtype= tf.float32, name = "theta_f")
b = tf.Variable(tf.zeros([1]), name="biases_f")
prediction = tf.matmul(lf, theta_f) + b

error = y - prediction

right_comparisons = 1 - tf.reduce_mean(tf.sign(tf.abs(tf.sign(y) - tf.sign(prediction))))
mse = tf.reduce_mean(tf.square(error), name = "mse")

learning_rate = 0.0001
n_epochs = 1500
n_batches = 1000
batch_size = 100
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(mse)

def fetch_batch(batch_size, batch_index):
    # random_indices = np.random.choice(pw.shape[0], batch_size, replace=False)
    indices = range(batch_index, min(batch_index + batch_size, len(pw)))
    x_batch1 = np.array(list(pw[indices,0]))
    x_batch1 = np.float32(np.c_[x_batch1, np.ones(batch_size)])
    x_batch2 = np.array(list(pw[indices,1]))
    x_batch2 = np.float32(np.c_[x_batch2, np.ones(batch_size)])
    y_batch = np.float32(pw[indices,2].reshape(batch_size, 1))
    return x_batch1, x_batch2, y_batch

def fetch_batch_single(batch_size, batch_index):
    # random_indices = np.random.choice(pw.shape[0], batch_size, replace=False)
    indices = range(batch_index, min(batch_index + batch_size, len(pw)))
    x1 = np.array(list(pw[indices,0]))
    x2 = np.array(list(pw[indices,1]))
    x_batch = np.hstack((x1, x2))
    x_batch = np.c_[x_batch, np.ones(batch_size)]
    y_batch = np.float32(pw[indices,2].reshape(batch_size, 1))
    return x_batch, y_batch

def fetch_test_single():
    size = len(pw_test)
    x_batch = np.hstack((np.array(list(pw_test[:,0])), np.array(list((pw_test[:,1])))))
    x_batch = np.c_[x_batch, np.ones(size)]
    y_batch = np.float32(pw_test[:,2].reshape(size, 1))
    return x_batch, y_batch


def fetch_test():
    # random_indices = np.random.choice(pw.shape[0], batch_size, replace=False)
    size = len(pw_test)
    x_batch1 = np.array(list(pw_test[:,0]))
    x_batch1 = np.float32(np.c_[x_batch1, np.ones(size)])
    x_batch2 = np.array(list(pw_test[:,1]))
    x_batch2 = np.float32(np.c_[x_batch2, np.ones(size)])
    y_batch = np.float32(pw_test[:,2].reshape(size, 1))
    return x_batch1, x_batch2, y_batch



init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    print("everything initialized")
    for epoch in range(n_epochs):
        if (epoch % 100 == 0):
            x_batch, y_batch = fetch_batch_single(batch_size * n_batches, 1)
            print("fit MSE:", mse.eval(feed_dict= {x: x_batch, y: y_batch}))
            print("fit right comparisons", right_comparisons.eval(feed_dict= {x: x_batch, y: y_batch}))
            x_batch, y_batch = fetch_test_single()
            print("test right comparisons:", right_comparisons.eval(feed_dict= {x: x_batch, y: y_batch}))

        for batch_index in range(n_batches):
            x_batch, y_batch = fetch_batch_single(batch_size, batch_index)
            sess.run(training_op, feed_dict= {x: x_batch, y: y_batch})
    save_path = saver.save(sess, "models/my_model_comparison_final.ckpt")
