import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

# 重み
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# バイアス
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 畳み込み
def conv2d(x, W):
    # x : [画像の枚数、縦、横、チャンネル数]
    # W : [縦、横、チャンネル数、フィルター数]
    # strides:フィルタの適用範囲。
    # padding:フィルタ適用時に画像領域が足りない時どうするか。
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# プーリング
def max_pool_2x2(x):
    # ksize : プーリングサイズ 3x3にしたい場合は[1, 3, 3, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# mnistのデータは要素数784(=28×28)のベクトルとして読み込まれる
mnist_data_dir = "./dataset"
mnist = input_data.read_data_sets(mnist_data_dir, one_hot=True)

# ニューラルネットワークの定義
x = tf.placeholder(tf.float32, [None, 784]) # input
supervisor = tf.placeholder(tf.float32, [None, 10]) # output
# 上下左右の隣接情報が必要⇒28x28にしている。
x_image = tf.reshape(x, [-1, 28, 28, 1]) # 28×28に変形

# 1層目
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 2層目
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全結合層1
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 過学習を防ぐためのドロップアウト設定
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全結合層2つ目。
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=supervisor, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 学習の検証
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(supervisor, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 精度表示用のＤａｔａＦｒａｍｅを作成
df = DataFrame(columns=['Accuracy'])

# セッションの実行
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('data', sess.graph)
    for i in range(5000):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={
            x: batch_xs,
            supervisor: batch_ys,
            keep_prob:0.5
        })

        if(i%100==0):
            testbatch_x,testbatch_y = mnist.test.next_batch(100)
            training_accuracy = accuracy.eval(feed_dict={
                x:testbatch_x,
                supervisor:testbatch_y,
                keep_prob:0.5
            })
            print(i," accuracy ",training_accuracy)
            df = df.append(Series([training_accuracy], index=['Accuracy']), ignore_index=True)
            
#　精度をグラフ表示
df.plot(title='Training accuracy', style='--', grid=True, ylim=(0, 1.1))
plt.show()   
            