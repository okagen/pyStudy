# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 09:21:08 2017

@author: 10007434
"""

"""
# sinカーブを描画
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0, 360) * np.pi/180
y = np.sin(x)
plt.plot(x, y)
plt.show()
"""

# click eventを試す
import matplotlib.pyplot as plt
def onclick(event):
    plt.scatter(event.xdata, event.ydata)

fig = plt.figure()
plt.xlim([0, 10])
plt.ylim([0, 10])
fig.canvas.mpl_connect('button_press_event', onclick)

