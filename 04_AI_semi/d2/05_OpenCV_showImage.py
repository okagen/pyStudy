import cv2
import numpy as np
import csv

filepath = "./dataset/imageAndList/"
filepath_filelist=filepath+"image_list.csv"

filelist=[]
supervisors=[]

# csvファイルの読み込み方法はいろいろあると思いますので、
# 好きなものを使ってください
with open(filepath_filelist,  newline='') as f:
    dataReader = csv.reader(f)

    for row in dataReader:
        filelist.append(row[0])
        supervisors.append(int(row[1]))


# ファイルリスト内の画像を一つずつ読み込んで連続表示
train_data = []
for i in filelist:
    img = cv2.imread(filepath+i,-1)
    train_data.append(img)
    cv2.imshow("imgae",img)
    k = cv2.waitKey(500)

    if k == 27:
        break
print(np.shape(train_data)) # (15,50,50,3)と表示される。15(枚)×50(縦)×50(横)×3(チャンネル)
cv2.destroyAllWindows()