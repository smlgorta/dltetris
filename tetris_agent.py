import tensorflow as tf
import numpy as np
import pandas as pd
import ast
import pickle

class TetrisAgent:
    def __init__(self):
        nx = 3400
        ny = 34
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, nx + 1), name="X")


        theta = tf.Variable(tf.random_uniform([nx + 1, 34], -1.0, 1.0), name="theta")
        self.y_pred = tf.matmul(self.X, theta, name="predictions")

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        self.sess = tf.Session()
        self.sess.run(init)
        saver.restore(self.sess, "models/my_model_final.ckpt")
        print("model restored", theta.eval(session = self.sess))
        print("theta shape", theta.shape)

    def play(self, input):
        y = self.y_pred.eval(session = self.sess, feed_dict={self.X: input})
        return y

    def pick(self, placements):
        input = np.array(placements).reshape(1, 3400)
        input = np.c_[1, input]
        y = self.play(input)
        picked = max((v, i) for i, v in enumerate(y[0]))[1]
        return picked