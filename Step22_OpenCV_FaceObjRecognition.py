# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 07:43:48 2017

@author: 10007434
"""


# オンラインマニュアル
# http://docs.opencv.org/2.4/modules/refman.html
# インストール
# pip install opencv-python

import cv2
imgname = 'Step22_OpenCV_FaceRec.jpg'
imgname_gray = 'Step22_OpenCV_FaceRec-gray.jpg'
imgname_bin = 'Step22_OpenCV_FaceRec-bin.jpg'
imgname_edge = 'Step22_OpenCV_FaceRec-edge.jpg'
imgname_faceRecog = 'Step22_OpenCV_FaceRec_FaceRecognition.jpg'

# オリジナルファイル
img_org = cv2.imread(imgname)

# グレースケールで画像を読み込む
img_gray = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
# ファイルに書き出す
cv2.imwrite(imgname_gray, img_gray)


# --------------------------
# 白黒画像を作成
# ２値化した画像を作る
ret, img_bit = cv2.threshold(img_gray, 100, 250, cv2.THRESH_BINARY)
# ファイルに書き出す
cv2.imwrite(imgname_bin, img_bit)
print(ret)


# --------------------------
# エッジ画像を作成
# エッジの抽出
img_edge = cv2.Canny(img_gray, 10, 300)
# ネガポジ反転
img_edge = 255 - img_edge
# ファイルに書き出す
cv2.imwrite(imgname_edge, img_edge)


# --------------------------
# 顔認証
# 顔認証の為の特徴データファイル（カスケードファイル）
#cascade_path = "haarcascade_frontalface_default.xml"
cascade_path = "haarcascade_frontalface_alt.xml"
#cascade_path = "haarcascade_frontalface_alt2.xml"
#cascade_path = "haarcascade_frontalface_alt_tree.xml"

# 顔認証
cascade=cv2.CascadeClassifier(cascade_path)
face_list = cascade.detectMultiScale(img_gray)

# 認証に失敗した場合
if len(face_list) == 0:
    print("Fail to recognize faces")
    quit()
    
# 認証した顔の範囲に赤色の枠を描画
for (x, y, w, h) in face_list:
    print('The coodinate of the face = ', x, y, w, h)
    color = (0, 255, 255)
    pen_w = 12
    cv2.rectangle(img_org, (x, y), (x+w, y+h), color, thickness = pen_w)
    
# ファイルに書き出す
cv2.imwrite(imgname_faceRecog, img_org)

"""    
# -------------------------
# 物体認識
img_org2 = cv2.imread('Step22_OpenCV_ObjRec.JPG')

# グレースケールに変換
img_gray2 = cv2.cvtColor(img_org2, cv2.COLOR_BGR2GRAY)
cv2.imwrite('Step22_OpenCV_ObjRec_gray_org.jpg', img_gray2) 

# 模様を消す為にぼかしを入れる
img_gray2 = cv2.GaussianBlur(img_gray2, (11, 11), 0)
cv2.imwrite('Step22_OpenCV_ObjRec_gray_blur.jpg', img_gray2) 

# 二値化
img_bit2 = cv2.threshold(img_gray2, 100, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imwrite('Step22_OpenCV_ObjRec_bit.jpg', img_bit2) 

# 輪郭を抽出
cnts = cv2.findContours(img_bit2,
                        cv2.RETR_LIST,
                        cv2.CHAIN_APPROX_SIMPLE)[1]

# 輪郭に枠線を描画
for pt in cnts:
    x, y, w, h = cv2.boundingRect(pt)
    if w < 50: continue
    print('The coodinate of the Object =', x, y, w, h)
    cv2.rectangle(img_org2, (x,y), (x+w, y+h), (0, 255, 0), 3)

# ファイルに書き出す
cv2.imwrite('Step22_OpenCV_ObjRec_ObjectRecogunition.jpg', img_org2)

"""
