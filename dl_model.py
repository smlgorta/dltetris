import dl_layers as dl
import tensorflow as tf

class Model:
    def __init__(self, model, numfeats, learning_rate, type):
        boardsize_flat = 100
        model = model
        numfeats = numfeats
        learning_rate = learning_rate
        type = type

        self.x1 = tf.placeholder(dtype=tf.float32, shape=(None, boardsize_flat), name="x1")
        self.x2 = tf.placeholder(dtype=tf.float32, shape=(None, boardsize_flat), name="x2")
        self.y = tf.placeholder(dtype=tf.float32, shape = (None, 1),  name = "y")

        self.x1 = tf.where(self.x1 == 0, -tf.ones_like(self.x1), self.x1)
        self.x2 = tf.where(self.x2 == 0, -tf.ones_like(self.x2), self.x2)

        if(model == "conv"):
            input_layer_1, input_layer_2 = dl.doubleConvLayer(tf.reshape(self.x1, [-1, 10, 10, 1]), tf.reshape(self.x2, [-1, 10, 10, 1]),  num_filters= 30, kernel_height=5, kernel_width=5, name="input_conv")
            layer_3_1, layer_3_2 = dl.doubleConvLayer(tf.reshape(input_layer_1, [-1, 6, 6, 30]), tf.reshape(input_layer_2, [-1, 6, 6, 30]), 30, 5, 5, in_channels= 30, name="l2")
            layer_4_1, layer_4_2 = dl.doubleLayer(tf.reshape(layer_3_1, [-1, 2 * 2 * 30]), tf.reshape(layer_3_2, [-1, 2 * 2 * 30]), 100, "l3")
            layer_5_1, layer_5_2 = dl.doubleLayer(layer_4_1, layer_4_2, numfeats, "l4")
        elif(model == "ff"):
            input_layer_1, input_layer_2 = dl.doubleLayer(self.x1, self.x2, 100, name="input")
            layer_4_1, layer_4_2 = dl.doubleLayer(input_layer_1, input_layer_2, 100, "l2")
            layer_5_1, layer_5_2 = dl.doubleLayer(layer_4_1, layer_4_2, numfeats, "l3")


        if(type == "full"):
            self.l3, self.single_state_pred = dl.double_to_singleLayer(layer_5_1, layer_5_2, 1, "l5")
        elif(type == "nc"):
            self.l3, self.single_state_pred, self.single_state_pred2 = dl.double_to_singleLayer_nc(layer_5_1, layer_5_2, 1, "l5", features_per_step= 1)
        else:
            self.l3, self.single_state_pred = dl.double_to_singleLayer_ew(layer_5_1, layer_5_2, 1, "l5")


        prediction = self.l3
        self.prediction_binary = dl.double_to_singleLayer_binary(layer_4_1, layer_4_2, numfeats, "l4", features_per_step= 1)
        tf.stop_gradient(self.prediction_binary)
        error = self.y - prediction
        self.right_comparisons = 1 - tf.reduce_mean(tf.sign(tf.abs(tf.sign(self.y) - tf.sign(prediction))))
        self.right_comparisons_binary = 1 - tf.reduce_mean(tf.sign(tf.abs(tf.sign(self.y) - tf.sign(self.prediction_binary))))
        self.mse = tf.reduce_mean(tf.square(error), name = "mse")

        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        self.training_op = optimizer.minimize(self.mse)

        def noncompensantion():
            layer_5_1