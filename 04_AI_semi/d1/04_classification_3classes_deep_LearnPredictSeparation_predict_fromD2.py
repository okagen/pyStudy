# -*- coding: utf-8 -*-
# TensorFlow 2次元点の3分類（ディープラーニング）
# モデルを読み込んで、予測

import tensorflow as tf
import numpy as np

# データ読込
def readCSV(filepath):
    ret = np.loadtxt(filepath, delimiter=",")
    return ret

if __name__=='__main__':

    # 予測用データ読込
    prediction_data = readCSV("./dataset/tf_3classes_prediction.csv")
    print("prediction_data :", prediction_data.shape)

    ########################## グラフをリセットする
    tf.reset_default_graph()
    ##########################
    
    # モデル作成時と同じ構成を再現する
    num_classes = 3
    n_inputs = 2
    n_hidden = 3
    n_outputs = 3
    
    x = tf.placeholder(tf.float32, shape=[None, n_inputs], name="x")
    supervisor = tf.placeholder(tf.float32, shape=[None, n_outputs], name="supervisor")
    
    # 1層目
    weight1 = tf.Variable(tf.truncated_normal([n_inputs, n_hidden], stddev=0.1), name="w1")
    bias1 = tf.Variable(tf.constant(0.1, shape=[n_hidden]), name="b1")
    h = tf.nn.sigmoid(tf.matmul(x, weight1) + bias1)
    
    # 2層目
    weight2 = tf.Variable(tf.truncated_normal([n_hidden, n_outputs], stddev=0.1), name="w2")
    bias2 = tf.Variable(tf.constant(0.1, shape=[n_outputs]), name="b2")
    y = tf.nn.softmax(tf.matmul(h, weight2) + bias2)
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=supervisor, logits=y))
    train_step = tf.train.AdamOptimizer(0.5).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(supervisor, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    init_op = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        # 学習はせずにパラメータを読み込む
        saver.restore(sess,"./model/model_3classes_deep")
    
        # 読み込んだパラメータで予測
        prediction_result = y.eval(feed_dict={
            x: prediction_data
        })
        print(prediction_result)
    
        saveresult = np.concatenate((prediction_data, prediction_result), axis = 1)
        np.savetxt("./result/04_tf_3classes_deep_result.csv", saveresult, delimiter=",")