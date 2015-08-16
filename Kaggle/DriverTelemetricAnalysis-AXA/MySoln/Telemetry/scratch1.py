# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 20:56:00 2014

@author: robbie
"""

import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1,3.2], [-2, -1,9], [-3, -2,6], [1, 1,6], [2, 1,3], [3, 2,1]])
pca = PCA(n_components=2)
pca.fit(X)
PCA(copy=True, n_components=2, whiten=False)
print(pca.explained_variance_ratio_) 


