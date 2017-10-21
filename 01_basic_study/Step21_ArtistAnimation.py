# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 09:20:43 2017

@author: 10007434
"""

"""
# sinカーブをアニメーション表示
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ims = []

# 10枚のグラフをあらかじめ作成し、ims配列に保存
for i in range(10):
    t = np.arange(0, 361, 10) * np.pi/180
    ims.append(plt.plot(np.sin(t+i)))

# Figureオブジェクトに予め作成したグラフを1000ミリ秒ごとに描画
ani = animation.ArtistAnimation(fig, ims, interval=1000)
plt.show()
"""

# sinカーブをアニメーション表示
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Figureオブジェクト
fig = plt.figure()
i=0

# datが無いと「ellipse() takes 0 positional arguments but 1 was given」となる。
def ellipse(dat):
    global i
    i = (i + 1) % 100
    plt.cla()
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    t = np.arange(0, 361, 10) * np.pi/180
    plt.scatter(np.cos(t)*i*0.1, np.sin(t)*i*0.1)


# plotを30ミリ秒ごとに実行する
ani = animation.FuncAnimation(fig, ellipse, interval=30)
plt.show()

    