from tensorflow.examples.tutorials.mnist import input_data
import cv2
import numpy as np

# mnistのデータは要素数784(=28×28)のベクトルとして読み込まれる
mnist_data_dir = "./dataset"
mnist = input_data.read_data_sets(mnist_data_dir, one_hot=True)

for i in range(len(mnist.train.images)):
    img = np.reshape(mnist.train.images[i],newshape=[28,28])
    cv2.imshow("imgae", img)
    k = cv2.waitKey(100)
    if k == 27:  # Escを押したら終了
        break

cv2.destroyAllWindows()