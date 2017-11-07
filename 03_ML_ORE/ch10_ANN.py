# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 07:22:01 2017

@author: 10007434
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris= load_iris()
X = iris.data[:, (2, 3)] # petal length, petal witdth

# iris.targetリスト内の値が「0」
# before ->[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 2 2 2 2]
# after  ->[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0]
y = (iris.target == 0).astype(np.int) # Iris Setrosa

per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)

y_pred = per_clf.predict([[2, 0.5]])
print("y_pred", y_pred)

