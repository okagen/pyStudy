# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 22:00:41 2017

@author: 10007434
"""
# opencvをインストールしておく
# conda install -c menppo opencv3=3.1.0

import cv2
import urllib.request as req

# スクレイピングで画像を取ってくる
url = 'https://www.python.org/static/img/python-logo.png'
img1 = 'Step19_py.png'
req.urlretrieve(url, img1)

# イメージを読み込む
img2 = cv2.imread(img1)

# ネガポジ反転
img3 = 255 - img2

# 処理結果をファイルに書き出す
cv2.imwrite("Step19_py_negaposi.png", img3)

#-------------------------------------------
# イメージをグレースケールで読み込む
img4 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)

# 閾値で白黒二値化を行う
th = 120
img4[img4 > th] = 255
img4[img4 < th] = 0
# 処理結果をファイルに書き出す
cv2.imwrite("Step19_py_bin.png", img4)

#-------------------------------------------
# イメージをグレースケールで読み込む
img5 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)

# OpenCVのフィルタを用いて白黒二値化を行う
ret, img5 = cv2.threshold(img5, 100, 250, cv2.THRESH_BINARY)

# 処理結果をファイルに書き出す
cv2.imwrite("Step19_py_bin_openCV.png", img5)

#-------------------------------------------
# イメージをグレースケールで読み込む
img6 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)

# 輪郭取得
img6 = cv2.Canny(img6, 80, 200)
# ネガポジ反転
img6 = 255 - img6

# 処理結果をファイルに書き出す
cv2.imwrite("Step19_py_bin_canny.png", img6)












