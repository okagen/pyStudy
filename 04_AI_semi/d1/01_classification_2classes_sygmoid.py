# -*- coding: utf-8 -*-
# TensorFlow 2次元点群の2分類（ロジスティック回帰）

import tensorflow as tf
import numpy as np
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
    ret = np.reshape(data[:,2],[-1,1])
    return ret

# 重み 標準偏差=0.1の乱数で生成する。
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# バイアス shapeで与えられた
def bias_variable(shape):
    # constantの使い方
    # x = tf.constant([1., 2., 3., 4., 5., 6., 7., 8., 9.], shape=[3, 3])
    # [[ 1.  2.  3.]
    #  [ 4.  5.  6.]
    #  [ 7.  8.  9.]]
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

if __name__=='__main__':
    # 学習データ読込
    train_dataset = readCSV( "./dataset/tf_2classes_training.csv")
    # 検証データ読込
    validation_dataset = readCSV("./dataset/tf_2classes_validation.csv")
    # 予測用データ読込
    prediction_data = readCSV("./dataset/tf_2classes_prediction.csv")
    
    # 学習データをデータと教師ラベルに分ける
    train_data = getData(train_dataset)
    train_supervisor = getSupervisor(train_dataset)
    
    # 検証データをデータと教師ラベルに分ける    
    validation_data = getData(validation_dataset)
    validation_supervisor = getSupervisor(validation_dataset)    
    
    n_inputs  = 2
    n_outputs= 1    
    
    # 重みの変数を作成
    weight = weight_variable([n_inputs, n_outputs])
    
    # バイアスの変数を作成   
    bias = bias_variable([n_outputs])
    
    # 入力データ用のplaceholderを作成　N行2列
    x = tf.placeholder(tf.float32, shape = [None, n_inputs], name="x")
    
    # 教師ラベル用のplaceholderを作成　N行1列
    supervisor = tf.placeholder(tf.float32, shape = [None, n_outputs], name="supervisor")
    
    # ニューラルネットワークの計算を作成
    # .matmulは行列の掛け算を行う。
    # シグモイド関数を活性化関数に採用
    y = tf.nn.sigmoid(tf.matmul(x, weight) + bias)
    
    # 誤差関数を作成　交差エントロピー誤差 （他にも二乗誤差関数などあり）
    # En=-Σ[k=1..n]{pk*ln(q)−(1−pk)*ln(1−(q))}
    # 「supervisor」は正解のラベルで、「y」は学習した結果。この差が小さくなるような処理。
    cross_entropy = - tf.reduce_sum(supervisor * tf.log(y) + (1 - supervisor) * tf.log(1 - y))
    
    # 誤差最適化手法（Optimizer）を作成　ここでは勾配降下法のうちの急速降下法を使う　学習率＝0.1
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
    
    # 検証のための誤差計算方法の作成
    # correct_prediction: yが0.5より大きければ1、そうでなければ0として、supervisorの値と等しいかどうかを求める
    # accuracy: correct_predictionの結果を32bit floatに変換し、平均を求める
    correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), supervisor)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 精度表示用のＤａｔａＦｒａｍｅを作成
    df = DataFrame(columns=['Accuracy'])

    # セッションの実行
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(10):
            sess.run(train_step, feed_dict={
                x:train_data,
                supervisor:train_supervisor
            })
    
            training_accuracy = accuracy.eval(feed_dict={
                x:validation_data,
                supervisor:validation_supervisor
            })
 
            #　精度表示
            print(i,"training_accuracy : ", training_accuracy)
            df = df.append(Series([training_accuracy], index=['Accuracy']), ignore_index=True)
    
        prediction_result=y.eval(feed_dict={
            x:prediction_data
        })
        
        #　精度をグラフ表示
        df.plot(title='Training accuracy', style='--', grid=True, ylim=(0, 1.1))
        plt.show()    
    
        # モデルを可視化
        # summary_writer =  tf.summary.FileWriter('01_classification_2class_sygmoid', graph_def=sess.graph_def)
        
        # 予測値は0～1の実数なので、0.5より大きければ1にする
        for i in range(len(prediction_result)):
            if prediction_result[i] > 0.5:
                prediction_result[i] = 1
            else:
                prediction_result[i] = 0
                
        # 書き出しのために予測データの次の列に予測結果を追加する
        saveresult = np.c_[prediction_data,prediction_result]
        
        # 予測結果書き出し
        np.savetxt("./result/tf_2classes_sigmoid_result.csv",saveresult,delimiter=",")