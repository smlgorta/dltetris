import tensorflow as tf
import dl_layers as dl
import numpy as np
import pandas as pd
import ast
import pickle
import random
import copy
import dl_model

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class TetrisAgent:
    def __init__(self, model, sess):

        boardsize_flat = 100
        self.pool = []
        self.pool_size_max = 300
        self.batch_size = 100
        self.num_batches = 10
        self.epoch = 150
        self.gamma = 0
        self.iter = 0
        self.model = model
        self.sess = sess

        # self.x = tf.placeholder(dtype=tf.float32, shape=(None, boardsize_flat*2 + 1), name="x")
        # self.x1 = tf.placeholder(dtype=tf.float32, shape=(None, boardsize_flat), name="x1")
        # self.x2 = tf.placeholder(dtype=tf.float32, shape=(None, boardsize_flat), name="x2")
        # self.x1_conv = tf.placeholder(dtype=tf.float32, shape=(None, 10, 10, 1), name="x1_conv")
        # self.x2_conv = tf.placeholder(dtype=tf.float32, shape=(None, 10, 10, 1), name="x2_conv")
        # self.y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y")
        # self.x_conv = tf.placeholder(dtype=tf.float32, shape=(None, 10, 10, 2), name="x")

        # n_filters_1 = 25
        # self.conv1 = tf.layers.conv2d(self.x_conv, filters = n_filters_1, kernel_size=[6, 6], padding="same", activation=tf.nn.relu)
        # n_filters_2 = 15
        # self.conv2 = tf.layers.conv2d(self.conv1, filters = n_filters_2, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        # self.input_layer_1, self.input_layer_2 = doubleConvLayer(tf.reshape(self.x1, [-1, 10, 10, 1]), tf.reshape(self.x2, [-1, 10, 10, 1]),  num_filters= 30, kernel_height=5, kernel_width=5, name="input_conv")
        # self.layer_3_1, self.layer_3_2 = doubleLayer(tf.reshape(self.input_layer_1, [-1, 30 * 6 * 6]), tf.reshape(self.input_layer_2, [-1, 30 * 6 * 6]), 1, "l1")

        # self.layer_4_1, self.layer_4_2 = doubleLayer(self.layer_3_1, self.layer_3_2, 100, "l3")

        # self.l3, self.single_state_pred = double_to_singleLayer(tf.reshape(self.input_layer_1, [-1, 30 * 6 * 6]), tf.reshape(self.input_layer_2, [-1, 30 * 6 * 6]), 2, "l4")
        # self.l3, self.single_state_pred = double_to_singleLayer(self.layer_3_1, self.layer_3_2, 2, "l4")

        # input_layer_1, input_layer_2 = dl.doubleConvLayer(tf.reshape(self.x1, [-1, 10, 10, 1]), tf.reshape(self.x2, [-1, 10, 10, 1]),
        #                                                num_filters=30, kernel_height=5, kernel_width=5,
        #                                                name="input_conv")
        # layer_3_1, layer_3_2 = dl.doubleConvLayer(tf.reshape(input_layer_1, [-1, 6, 6, 30]),
        #                                        tf.reshape(input_layer_2, [-1, 6, 6, 30]), 30, 5, 5, in_channels=30,
        #                                        name="l2")
        # layer_4_1, layer_4_2 = dl.doubleLayer(tf.reshape(layer_3_1, [-1, 2 * 2 * 30]),
        #                                    tf.reshape(layer_3_2, [-1, 2 * 2 * 30]), 30, "l3")
        # layer_5_1, layer_5_2 = dl.doubleLayer(layer_4_1, layer_4_2, 10, "l4")
        # self.l3, self.single_state_pred = dl.double_to_singleLayer_nc(layer_5_1, layer_5_2, 1, "l5", features_per_step=1)
        # self.l3, self.single_state_pred = double_to_singleLayer(layer_5_1, layer_5_2, 1, "l5")
        # self.l3, self.single_state_pred = double_to_singleLayer_ew(layer_5_1, layer_5_2, 1, "l5")

        # self.l4 = singleLayer(self.l3, 20, "l4")
        # self.lf = singleLayer(self.l3, 2, "lf")

        # self.l2 = singleLayer(tf.reshape(self.conv2, [-1, n_filters_2 * 100]), 30, "l2")
        # self.l3 = singleLayer(self.l2, 50, "l3")
        # self.lf = singleLayer(self.l3, 50, "lf")

        # input_size = int(self.l2.get_shape()[1])
        #
        # self.theta_f = tf.Variable(initial_value=tf.random_uniform([input_size, 2], -1.0, 1.0), dtype=tf.float32,
        #                       name="theta_f")
        # self.b = tf.Variable(tf.zeros([1]), name="biases_f")
        # self.prediction = tf.nn.softmax(tf.matmul(self.l2, self.theta_f) + self.b)
        # self.prediction = self.l3

        # self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        #     labels=self.y, logits=self.l3, name='xentropy')
        # self.loss = tf.reduce_mean(self.cross_entropy, name='xentropy_mean')
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)
        # self.error = self.y - self.prediction
        # self.mse = tf.reduce_mean(tf.square(self.error), name="mse")
        # self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate)
        # self.training_op = self.optimizer.minimize(self.mse)

        # self.init = tf.global_variables_initializer()
        # saver = tf.train.Saver()
        # self.sess = tf.Session()
        # self.sess.run(self.init)
        # saver.restore(self.sess, model_path)
        # print("model restored: ", model_path)

    def play(self, placements):
        flattened_placements = copy.deepcopy(placements)
        bestplacement = 0;

        indices_range = range(len(flattened_placements))
        zipped = list(zip(indices_range, flattened_placements))
        random.shuffle(zipped)
        indices = []
        indices[:], flattened_placements[:] = zip(*zipped)

        for i in range(len(flattened_placements)):
                if(i != bestplacement):
                    # pi = np.hstack((np.array(list(flattened_placements[i])), np.array(list(flattened_placements[bestplacement]))))
                    # pi = np.vstack((np.array(list(flattened_placements[i])).reshape(-1),
                    #                np.array(list(flattened_placements[bestplacement])).reshape(-1))).reshape((-1,), order='F')  # Interleaves two states

                    x1 = np.append([np.array(list(flattened_placements[i])).reshape(-1)], []).reshape(
                        [-1, 100])
                    x2 = np.append([np.array(list(flattened_placements[bestplacement])).reshape(-1)],
                                   []).reshape([-1, 100])

                    x1_conv = np.array(list(flattened_placements[i])).reshape([-1, 10, 10, 1])
                    x2_conv = np.array(list(flattened_placements[bestplacement])).reshape([-1, 10, 10, 1])

                    # pi = pi.reshape(1, 10, 10, 2)

                    # y = self.prediction.eval(session=self.sess, feed_dict={self.x_conv: pi})
                    y = self.prediction.eval(session=self.sess, feed_dict={self.x1: x1, self.x2: x2})

                    # pi2 = np.vstack((np.array(list(flattened_placements[bestplacement])).reshape(-1),
                    #                np.array(list(flattened_placements[i])).reshape(-1))).reshape((-1,), order='F')
                    # pi2 = pi2.reshape(1, 10, 10, 2)
                    #
                    # y2 = self.prediction.eval(session=self.sess, feed_dict={self.x_conv: pi2})

                    if(y[0][0] > 0.5):
                        # print(i, " better than ", bestplacement, " by ", y)
                        bestplacement = i
        return indices[bestplacement]

    def play_valuebased(self, placements):
        flattened_placements = copy.deepcopy(placements)
        values = []
        for i in range(len(flattened_placements)):
            values.append(self.value(flattened_placements[i]))
        return np.argmax(values)

    def value(self, flattened_placement):
        x1 = np.append([np.array(list(flattened_placement)).reshape(-1)], []).reshape(
            [-1, 100])
        y = self.model.single_state_pred.eval(session=self.sess, feed_dict={self.model.x1: x1, self.model.x2: x1})
        return(y[0][0])

    def playKnockout(self, placements):

        flattened_placements = copy.deepcopy(placements)
        indices_range = range(len(flattened_placements))
        zipped = list(zip(indices_range, flattened_placements))
        random.shuffle(zipped)
        indices = []
        indices[:], flattened_placements[:] = zip(*zipped)
        alive = list(range(len(flattened_placements)))
        lastroundlen = len(alive) + 1

        while(lastroundlen != len(alive)):
            toremove = list()
            for match in range(int(len(alive) / 2)):

                x1 = np.append([np.array(list(flattened_placements[alive[match]])).reshape(-1)],[]).reshape([-1,100])
                x2 = np.append([np.array(list(flattened_placements[alive[len(alive) -match-1]])).reshape(-1)],[]).reshape([-1,100])

                # pi = np.vstack((np.array(list(flattened_placements[alive[match]])).reshape(-1),
                #                 np.array(list(flattened_placements[alive[len(alive) -match-1]])).reshape(-1))).reshape((-1,), order='F')  # Interleaves two states

                # pi = pi.reshape(1, 10, 10, 2)
                # pi = pi.reshape(1, 200)
                # pi = np.append([pi], [1])
                # pi = pi.reshape(1, 201)

                # y = self.prediction.eval(session=self.sess, feed_dict={self.x_conv: pi})
                y = self.prediction.eval(session=self.sess, feed_dict={self.x1: x1, self.x2: x2})

                # pi2 = np.vstack((np.array(list(flattened_placements[alive[len(alive) -match-1]])).reshape(-1),
                #                 np.array(list(flattened_placements[alive[match]])).reshape(-1))).reshape((-1,), order='F')  # Interleaves two states
                # pi2 = pi2.reshape(1, 10, 10, 2)
                # pi2 = pi2.reshape(1, 200)
                # pi2 = np.append([pi2], [1])
                # pi2 = pi2.reshape(1, 201)


                # y2 = self.prediction.eval(session=self.sess, feed_dict={self.x_conv: pi2})

                if(y[0][0] > 0.5):
                    toremove.append(alive[len(alive) -match -1])
                elif(y[0][0] < 0.5):
                    toremove.append(alive[match])

            lastroundlen = len(alive)
            alive = [x for x in alive if x not in toremove]
            # print(alive)
        return indices[random.choice(alive)]

    def add_to_experience_pool(self, placements, experienced_reward, reward_experienced):
        # print("experiencing reward: ", sum(experienced_reward))
        # if(len(placements) == 1 or max(experienced_reward) == 0):
        if(len(placements) == 1):
            return;
        else:
            if(not reward_experienced and random.random() > 0):
                return;
            # printing experience summarized
            # index = np.argmax(experienced_reward)
            # print(placements[index])
            # print(" better than")
            # if(index > 0):
            #     print(placements[index - 1])
            # else:
            #     print(placements[index + 1])
            # print("__")
            if(len(self.pool) == 0):
                self.pool.insert(0, [placements, experienced_reward])
            else:
                self.pool.insert(random.randint(0, len(self.pool)-1), [placements, experienced_reward])

        if(len(self.pool) > self.pool_size_max):
            self.pool.pop()


    def train_from_pool(self):
        if(len(self.pool) < self.batch_size):
            print("pool size:", len(self.pool))
            return;

        print("pool ready, size ", len(self.pool),". Let's train!, iter:", self.iter)
        self.iter = self.iter + 1
        exp_batch = random.sample(self.pool, self.batch_size)
        ytot = []
        pitot = []
        for exp in exp_batch:
            placements = exp[0]
            experienced_reward = exp[1]

            max_exp_index = np.argmax(experienced_reward)
            max_reward = np.max(experienced_reward)
            for i in range(len(placements)):
                for j in range(len(placements)):
                    if(i == j):
                        continue;


                    # if(random.random() > 0.5):
                    x1 = np.append([np.array(list(placements[i])).reshape(-1)], []).reshape([-1])
                    x2 = np.append([np.array(list(placements[j])).reshape(-1)], []).reshape([-1])
                    # x1_conv = np.array(list(placements[i])).reshape([10, 10])
                    # x2_conv = np.array(list(placements[j])).reshape([10, 10])
                    # vx1 = self.value(np.array(list(placements[i])).reshape(-1))
                    # vx2 = self.value(np.array(list(placements[j])).reshape(-1))

                    # pi = np.vstack((np.array(list(placements[i])).reshape(-1),
                    #                 np.array(list(placements[j])).reshape(-1))).reshape((-1,), order='F') #Interleaves two states
                    # pi = pi.reshape(10, 10, 2)
                    # if(max_reward > 0):
                    ytot.append(experienced_reward[i]-experienced_reward[j])
                    # ytot.append(softmax([experienced_reward[i], experienced_reward[j]]))
                    pitot.append(np.array([x1,x2]))
                    # else:

                    # x12_conv = np.array(list(placements[j])).reshape([10, 10])
                    # x22_conv = np.array(list(placements[i])).reshape([10, 10])


                    # if(max_reward > 0):
                    ytot.append(experienced_reward[j]-experienced_reward[i])
                    # ytot.append(softmax([experienced_reward[j], experienced_reward[i]]))
                    pitot.append(np.array([x2,x1]))

                    # else:
                    #     ytot.append([0.5, 0.5])
                    # pi = pi.reshape(200)
                    # pi = np.append([pi], [1])
                    # pi = pi.reshape(201)


                    # pi2 = np.vstack((np.array(list(placements[j])).reshape(-1),
                    #                np.array(list(placements[i])).reshape(-1))).reshape((-1,), order='F') #Interleaves two states
                    # pi2 = pi2.reshape(10, 10, 2)
                    # if (max_reward > 0):
                    # ytot.append(softmax([experienced_reward[j], experienced_reward[i]]))
                    # else:
                    #     ytot.append([0.5, 0.5])

                    # pi2 = pi2.reshape(200)
                    # pi2 = np.append([pi2], [1])
                    # pi2 = pi2.reshape(201)

                    # pitot.append(pi2)

        ytot = np.array(ytot)
        pitot = np.array(pitot)

        s = np.arange(ytot.shape[0])
        np.random.shuffle(s)

        ytot = ytot[s]
        pitot = pitot[s]

        def fetch_batch(batch_index):
            mini_batch_size = int(len(ytot) / self.num_batches)
            init = batch_index*mini_batch_size
            end = (batch_index+1) * mini_batch_size
            return ytot[init : end].reshape(-1, 1), pitot[init : end]

        # # printing one comparison (x_conv):
        # print(pitot.shape)
        # print(pitot[0, :, :, 0])
        # print("_")
        # print(pitot[0, :, :, 1])
        # print(ytot[0])
        # print("____")
        # print(pitot[1, :, :, 0])
        # print("_")
        # print(pitot[1, :, :, 1])
        # print(ytot[1])
        # print()

        def right_choices():
            right_vb = 0
            right_l = 0
            chance = 0
            total = 0
            for exp in exp_batch:
                if(max(exp[1]) > 0):
                    total += 1
                    chosen_id_vb = self.play_valuebased(exp[0])
                    chosen_id_l = self.play(exp[0])
                    chance += 1/len(exp[0])
                    if(np.argmax(exp[1]) == chosen_id_vb):
                        right_vb += 1
                    if(np.argmax(exp[1]) == chosen_id_l):
                        right_l += 1

            return right_vb / (total+1), right_l / (total+1), chance / (total+1)

        # sess.run(self.init)

        print("pairs ready, starting to backpropagate")
        for i in range(self.epoch):
            # error = ytot - self.prediction
            if(i % 100 == 0 or i + 1 == self.epoch):
                # print("cross entropy:", loss.eval(feed_dict={self.x: pitot}, session = self.sess))
                rightchoices_vb, rightchoices_list, chance = right_choices()
                print("right_choices_vb:", rightchoices_vb, "right_choices_l:", rightchoices_list," chance:" , chance)
                rightchoices_vb, rightchoices_list, chance = right_choices()
                # print("right_choices_vb:", rightchoices_vb, "right_choices_l:", rightchoices_list," chance:" , chance)
                # rightchoices_vb, rightchoices_list, chance = right_choices()
                # print("right_choices_vb:", rightchoices_vb, "right_choices_l:", rightchoices_list," chance:" , chance)
                meaningful = np.where(ytot[:] != 0)
                if(len(meaningful[0]) > 0):
                    # right_paired_comparisons = sum(np.argmax(ytot[meaningful[0],:], axis = 1) == np.argmax(self.prediction.eval(feed_dict={self.x_conv: pitot[meaningful[0],:]}, session = self.sess),  axis = 1))
                    # pred = self.prediction.eval(feed_dict={self.x1: pitot[meaningful[0],0], self.x2: pitot[meaningful[0],1]}, session = self.sess)
                    # print(pred)
                    # right_paired_comparisons = sum(np.argmax(ytot[meaningful[0],:], axis = 1) == np.argmax(pred,  axis = 1))
                    right_paired_comparisons = tf.reduce_mean(tf.sign(tf.abs(tf.sign(ytot[meaningful]) - tf.sign(self.prediction, tf.float32))))
                    print(i," - right paired comparisons:", right_paired_comparisons, " out of ", len(ytot[meaningful]) ,":", right_paired_comparisons/len(ytot[meaningful[0]]))

            for batch_index in range(0, self.num_batches):
                by, bx = fetch_batch(batch_index)
                # self.sess.run(self.training_op, feed_dict={self.x_conv: bx, self.y: by})
                self.sess.run(self.training_op, feed_dict={self.x1: bx[:, 0],
                                                           self.x2: bx[:, 1],
                                                           self.y: by})
        #discard data:
        self.pool = []

    def pick(self, placements, cols = [], rots = []):
        y = self.play_valuebased(placements)
        return y
