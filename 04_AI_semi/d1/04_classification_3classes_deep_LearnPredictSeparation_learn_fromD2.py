# -*- coding: utf-8 -*-
# TensorFlow 2次元点の3分類（ディープラーニング）
# 学習してモデルを生成

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

# データ読込
def readCSV(filepath):
    ret = np.loadtxt(filepath, delimiter=",")
    return ret

# 1,2列目を取得（入力データ）
def getData(data):
    ret = data[:,0:2]
    return ret

# 3列目を取得（教師ラベル）し、ベクトル化
def getSupervisor(data):
    #ret = np.reshape(data[:,2],[-1,1])
    #-1を使用すると、元の要素数に合わせて自動で適切な値が設定される。    
    ret = np.reshape(data[:,2],[-1])
    return ret

# 重み 標準偏差=0.1の乱数で生成する。
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

# バイアス shapeで与えられた
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

if __name__=='__main__':
    # 学習データ読込
    train_dataset = readCSV( "./dataset/tf_3classes_training.csv")
    print("train_dataset :", train_dataset.shape)
    # 検証データ読込
    validation_dataset = readCSV("./dataset/tf_3classes_validation.csv")
    print("validation_dataset :", validation_dataset.shape)

    # 学習データをデータと教師ラベルに分ける
    train_data = getData(train_dataset)
    print("train_data :", train_data.shape)
    train_supervisor = getSupervisor(train_dataset)
    print("train_supervisor :", train_supervisor.shape)
    
    # 検証データをデータと教師ラベルに分ける    
    validation_data = getData(validation_dataset)
    print("validation_data :", validation_data.shape)
    validation_supervisor = getSupervisor(validation_dataset)    
    print("validation_supervisor :", validation_supervisor.shape)
    
    ########################## グラフをリセットする
    tf.reset_default_graph()
    ##########################
    
    # 学習用　教師ラベルをOneHot形式に変形する　３分類
    num_classes = 3
    train_supervisor_onehot = tf.one_hot(train_supervisor, depth=num_classes, on_value=1, off_value=0)
    print("train_supervisor_onehot :", train_supervisor_onehot.shape)
    # 検証用　教師ラベルをOneHot形式に変形する　３分類
    validation_supervisor_onehot = tf.one_hot(validation_supervisor, depth=num_classes, on_value=1, off_value=0)
    print("validation_supervisor_onehot :", validation_supervisor_onehot.shape)

    # ニューラルネットワークの形を定義　
    n_inputs = 2 # インプットは２ノード
    n_hidden = 3 # 隠れ層は３ノード
    n_outputs = 3 # アウトプットは３ノード
    
    # 入力層のプレイスフォルダ作成
    x = tf.placeholder(tf.float32, shape = [None, n_inputs], name="x")
    supervisor = tf.placeholder(tf.float32, shape = [None, n_outputs], name="supervisor")
    
    # 隠れ層　1層目 2行3列
    weight1 = weight_variable([n_inputs, n_hidden], name="w1")
    bias1 = bias_variable([n_hidden], name="b1")
    h = tf.nn.sigmoid(tf.matmul(x, weight1) + bias1)
    
    # 隠れ層　2層目 3行3列
    weight2 = weight_variable([n_hidden, n_outputs], name="w2")
    bias2 = bias_variable([n_outputs], name="b2")
    y = tf.nn.softmax(tf.matmul(h, weight2) + bias2)
    
    # 誤差関数を作成
    # 交差エントロピー誤差のsoftmax値を計算して、cross entropyを計算
    # supervisorデータはＮ行３列（３次元）なので、tf.nn.softmax_cross_entropy_with_logitsの戻り値が複数になる。
    # よってreduce_meanで平均値を計算する。
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=supervisor, logits=y))
    
    # 誤差最適化手法（Optimizer）を作成　ここでは勾配降下法のうちのAdamアルゴリズムを使う 学習率＝0.5
    train_step = tf.train.AdamOptimizer(0.5).minimize(cross_entropy)
    
    # 検証のための誤差計算方法の作成
    # argmaxの使い方・・・
    # array([[4, 9, 1, 5],
    #        [3, 5, 6, 2],
    #        [9, 8, 1, 0]])
    # axis=0の場合、縦方向の成分の中で最大値のインデックスをとっていく
    # a0 = np.argmax(tmp, axis=0)
    # a0:[2, 0, 1, 0]
    # axis=1の場合、横方向の成分の中で最大値のインデックスをとっていく
    # a1 = np.argmax(tmp, axis=1)
    # a1:[1, 2, 0]
    # correct_prediction: ３分類した結果、一番可能性が大きいと計算されたものが、教師データの分類と一致した場合1、そうでない場合0
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(supervisor, 1))
    
    # accuracy: correct_predictionの結果を32bit floatに変換し、平均を求める    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # 精度表示用のＤａｔａＦｒａｍｅを作成
    df = DataFrame(columns=['Accuracy'])

    # セッションの実行
    init_op = tf.global_variables_initializer()

    ########################## 以下の1行をつける
    saver = tf.train.Saver()
    ##########################

    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(1000):
            # 順番をシャッフル
            x_shuffled, y_shuffed = shuffle(train_data, train_supervisor_onehot.eval())
            
            sess.run(train_step, feed_dict={
                x: x_shuffled,
                supervisor: y_shuffed
            })
    
            training_accuracy = accuracy.eval(feed_dict={
                x: validation_data,
                supervisor: validation_supervisor_onehot.eval()
            })
            if(i%1==0):
               print(i,"training_accuracy : ", training_accuracy)
               df = df.append(Series([training_accuracy], index=['Accuracy']), ignore_index=True)

    
        #　精度をグラフ表示
        df.plot(title='Training accuracy', style='--', grid=True, ylim=(0, 1.1))
        plt.show()
        
        ########################## 下の１行で保存．いくつかファイルができるのでディレクトリにした方がよい
        saver.save(sess,"./model/model_3classes_deep")
        ##########################

