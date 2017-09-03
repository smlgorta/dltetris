import tensorflow as tf
import numpy as np
import pandas as pd
import ast
import pickle

class TetrisAgent:
    def __init__(self):

        boardsize_flat = 100

        self.x = tf.placeholder(dtype=tf.float32, shape=(None, boardsize_flat*2 + 1), name="x")
        # self.x2 = tf.placeholder(dtype=tf.float32, shape=(None, boardsize_flat + 1), name="x2")
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y")

        def doubleLayer(input1, input2, hidden_nodes, name):
            with tf.name_scope(name):
                input_size = int(input1.get_shape()[1])
                theta = tf.Variable(initial_value=tf.random_uniform([input_size, hidden_nodes], -1.0, 1.0),
                                    dtype=tf.float32,
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

        l1 = singleLayer(self.x, 50, "l2")
        l2 = singleLayer(l1, 10, "lf")

        input_size = int(l2.get_shape()[1])
        theta_f = tf.Variable(initial_value=tf.random_uniform([input_size, 1], -1.0, 1.0), dtype=tf.float32,
                              name="theta_f")
        b = tf.Variable(tf.zeros([1]), name="biases_f")
        self.prediction = tf.matmul(l2, theta_f) + b

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(init)
        saver.restore(self.sess, "models/my_model_comparison_final.ckpt")
        print("model restored")

    def play(self, placements):
        flattened_placements = []
        for i in range(len(placements)):
            print(i)
            for r in range(10):
                for c in range(10):
                    print(placements[i][r][c], end = " ")
                print("")
            print("")
            p = np.array(placements[i]).reshape(1,100)
            flattened_placements.append(p)
        bestplacement = 0;
        for i in range(len(flattened_placements)):
                if(i != bestplacement):
                    pi = np.hstack((np.array(list(flattened_placements[i])), np.array(list(flattened_placements[bestplacement]))))
                    pi = np.c_[pi, np.ones(1)]
                    y = self.prediction.eval(session=self.sess, feed_dict={self.x: pi})
                    if(y > 0):
                        print(i, " better than ", bestplacement, " by ", y)
                        bestplacement = i
        return bestplacement

    def pick(self, placements):
        y = self.play(placements)
        return y