from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

# mnistのデータは要素数784(=28×28)のベクトルとして読み込まれる
#Extracting ./dataset\train-images-idx3-ubyte.gz
#Extracting ./dataset\train-labels-idx1-ubyte.gz
#Extracting ./dataset\t10k-images-idx3-ubyte.gz
#Extracting ./dataset\t10k-labels-idx1-ubyte.gz
mnist_data_dir = "./dataset"
mnist = input_data.read_data_sets(mnist_data_dir, one_hot=True)

# 入力と教師データ用のプレイスホルダー
# [None, 784]は2次元のテンソル。Noneは次元の指定なしの意味 N行784列
x = tf.placeholder(tf.float32, [None, 784])
# ラベル　N行104列
t = tf.placeholder(tf.float32, [None, 10])

# ネットワークの定義
# すべての要素が0のTensorを生成
weight = tf.Variable(tf.zeros([784, 10]))
# 計算時にbloadcastigされ、Ｎ行10列になる。
bias = tf.Variable(tf.zeros([10]))

# tf.matmul(x, weight) : N行784列　x 784行10列　⇒ N行10列
#  + bias : N行10列 + Ｎ行10列　　⇒ N行10列
# softmax :  N行10列が出力される。[0の確率, １の確率,・・・, 9の確率]　x　N行
y = tf.nn.softmax(tf.matmul(x, weight) + bias)

# 誤差関数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y))

# 最適化手法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 学習評価用
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 精度表示用のＤａｔａＦｒａｍｅを作成
df = DataFrame(columns=['Accuracy'])

# セッションの実行
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        # ミニバッチを100として学習データと正解ラベルを取得
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # train_stepで定義した関数に関連する計算を全て行う。
        sess.run(train_step, feed_dict={
            x: batch_xs,
            t: batch_ys
        })

        # 学習評価用のデータ（学習には不使用）で評価
        if(i%10==0):
            training_accuracy = accuracy.eval(feed_dict={
                x: mnist.test.images,
                t: mnist.test.labels
            })
            print(i, " accuracy ", training_accuracy)
            df = df.append(Series([training_accuracy], index=['Accuracy']), ignore_index=True)
            
            
#　精度をグラフ表示
df.plot(title='Training accuracy', style='--', grid=True, ylim=(0, 1.1))
plt.show()                  