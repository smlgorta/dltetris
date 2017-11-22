import tensorflow as tf
import dl_model
import numpy as np
import sqlite3 as lite
import json
import random
import sys
import tetris

con = lite.connect("/home/simon/PycharmProjects/dltetris/states.db")
con.row_factory = lite.Row
cursor = con.cursor()

boardsize_flat = 100
learning_rate = 0.1
n_epochs = 500
n_batches = 100
batch_size = 1000
testsetsize = 20000
numfeats = int(sys.argv[2])
model_type = str(sys.argv[3])
run_code = repr(random.random())
last_layer_type = str(sys.argv[1])
namefile = '/home/simon/PycharmProjects/dltetris/data/comparison_training/'+model_type+'/comparison_training_'+last_layer_type+'_'+str(numfeats)+'feats_'+run_code+'.txt'
# namefile = '/home/simon/PycharmProjects/dltetris/data/comparison_training/'+model_type+'/temp.txt'
print("saving to file ", namefile)
f = open(namefile, 'w')

mod = dl_model.Model(model_type, numfeats, learning_rate, last_layer_type)

def loadData():
    lowerindex = 1
    upperindex = n_batches * batch_size + testsetsize
    x_batch_1 = []
    x_batch_2 = []
    y_batch = []
    cursor.execute("SELECT * FROM pairwise_comparisons WHERE rowid >=? AND rowid <=?", (lowerindex, upperindex))
    comparisons = cursor.fetchall()
    for comp in comparisons:
        b1 = json.loads(comp["stateaction_1"])
        b2 = json.loads(comp["stateaction_2"])
        v = comp["value"]
        x_batch_1.append(np.concatenate((np.array(b1),[]), axis = 0))
        x_batch_2.append(np.concatenate((np.array(b2),[]), axis = 0))
        y_batch.append(v)

    randomize = np.arange(len(y_batch))
    np.random.shuffle(randomize)
    print(np.array(y_batch).shape)
    print(np.array(x_batch_1).shape)
    print(np.array(x_batch_2).shape)
    return np.array(x_batch_1)[randomize], np.array(x_batch_2)[randomize], np.array(y_batch).reshape(-1,1)[randomize]

print("loading data")
data_x1, data_x2, data_y = loadData()
print("data loaded")


def fetch_batch_single(batch_size, batch_index):
    indices = range(batch_index*batch_size, batch_index*batch_size + batch_size)
    x_batch_1 = data_x1[indices]
    x_batch_2 = data_x2[indices]
    y_batch = data_y[indices]
    return x_batch_1, x_batch_2, y_batch

def fetch_testset():
    indices = range(n_batches*batch_size, n_batches*batch_size + testsetsize)
    x_batch_1 = data_x1[indices]
    x_batch_2 = data_x2[indices]
    y_batch = data_y[indices]
    return x_batch_1, x_batch_2, y_batch

init = dl_model.tf.global_variables_initializer()
saver = dl_model.tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    print("everything initialized")
    for epoch in range(n_epochs):
        if (epoch % 10 == 0):
            x_batch_1_t, x_batch_2_t, y_batch_t = fetch_testset()
            x_batch_1, x_batch_2, y_batch = fetch_batch_single(batch_size, 1)
            f.write(str(epoch) +",fit,mse," + str(mod.mse.eval(feed_dict= {mod.x1: x_batch_1, mod.x2: x_batch_2, mod.y: y_batch}))+"\n")
            f.write(str(epoch) + ",fit,comparisons," + str(mod.right_comparisons.eval(feed_dict= {mod.x1: x_batch_1, mod.x2: x_batch_2, mod.y: y_batch}))+"\n")
            f.write(str(epoch) + ",test,mse,"+ str(mod.mse.eval(feed_dict= {mod.x1: x_batch_1_t, mod.x2: x_batch_2_t, mod.y: y_batch_t}))+"\n")
            f.write(str(epoch) + ",test,comparisons,"+ str(mod.right_comparisons.eval(feed_dict= {mod.x1: x_batch_1_t, mod.x2: x_batch_2_t, mod.y: y_batch_t}))+"\n")
            mean_score = tetris.playGames(50, mod, sess)
            f.write(str(epoch) + ",test,meanscore,"+str(mean_score)+"\n")
            f.flush()
            print("fit MSE:", mod.mse.eval(feed_dict= {mod.x1: x_batch_1, mod.x2: x_batch_2, mod.y: y_batch}))
            print(epoch, "fit right comparisons", mod.right_comparisons.eval(feed_dict= {mod.x1: x_batch_1, mod.x2: x_batch_2, mod.y: y_batch}))
            print("test MSE:", mod.mse.eval(feed_dict= {mod.x1: x_batch_1_t, mod.x2: x_batch_2_t, mod.y: y_batch_t}))
            print("test right comparisons:", mod.right_comparisons.eval(feed_dict= {mod.x1: x_batch_1_t, mod.x2: x_batch_2_t, mod.y: y_batch_t}))
            print("test right comparisons ttb:", mod.right_comparisons_binary.eval(feed_dict= {mod.x1: x_batch_1_t, mod.x2: x_batch_2_t, mod.y: y_batch_t}))
            print("mean score: ", mean_score)

        for batch_index in range(n_batches):
            x_batch_1, x_batch_2, y_batch = fetch_batch_single(batch_size, batch_index)
            sess.run(mod.training_op, feed_dict= {mod.x1: x_batch_1, mod.x2: x_batch_2, mod.y: y_batch})
    save_path = saver.save(sess, "models/my_model_comparison_final.ckpt")
    print("saved: ",save_path)
    f.close()
