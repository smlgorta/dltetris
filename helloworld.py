import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape

n_epochs = 1000
learning_rate = 0.01
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]
row_sums = housing_data_plus_bias.max(axis = 0)
scaled_housing_data_plus_bias = housing_data_plus_bias / row_sums

X = tf.placeholder(dtype=tf.float32, shape=(None, n +1), name="X")
y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y")

# X = tf.constant(scaled_housing_data_plus_bias, dtype= tf.float32, name = "X")
# y = tf.constant(housing.target.reshape(-1, 1), dtype= tf.float32, name = "y")

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name = "theta")
y_pred = tf.matmul(X, theta, name = "predictions")

error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name = "mse")
# gradients = 2/m * tf.matmul(tf.transpose(X), error)
gradients = tf.gradients(mse, [theta])[0]

# training_op = tf.assign(theta, theta - learning_rate * gradients)
optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = 0.9)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

def fetch_batch(epoch, batch_index, batch_size):
    random_indices = np.random.choice(scaled_housing_data_plus_bias.shape[0], batch_size, replace=False)
    indices = range(batch_index, min(batch_index + batch_size, m))
    x_batch = scaled_housing_data_plus_bias[indices,:]
    y_batch = housing.target[indices].reshape(-1,1)
    return x_batch, y_batch

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:  # checkpoint every 100 epochs
            save_path = saver.save(sess, "/tmp/my_model.ckpt")

        for batch_index in range(n_batches):
            x_batch, y_batch = fetch_batch(epoch= epoch, batch_index= batch_index, batch_size= batch_size)
            sess.run(training_op, feed_dict= {X: x_batch, y: y_batch})

    print("theta:", theta.eval())
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")