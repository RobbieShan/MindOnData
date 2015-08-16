import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.decomposition import PCA as PCA

datapath = 'G:/Continuing Education/Research & Presentations/Self - Machine Learning/Kaggle/OttoProductClassification/Data/'
trainfile = 'train.csv'
testfile = 'test.csv'

trd = pd.read_csv(datapath+trainfile)
nptrd = trd.values

pca = PCA(n_components=10)
pca.fit(nptrd[:,range(1,94)])
X = pca.transform(nptrd[:,range(1,94)])
sum(pca.explained_variance_ratio_)

# Most of the features are highly skewed i.e. their 75% value ranges when we do td.describe() is 0 while their max is much higher. 
# This indicates that only a few values are non-zero for most features.
# This could mean that these features are actually categorical variables that are encoded in the test data.. could .. not sure

forest = rfc(n_estimators=5)
#forest = forest.fit(nptrd[:,range(1,94)],nptrd[:,-1])
forest = forest.fit(X,nptrd[:,-1])

#temp = forest.predict(nptrd[:,range(1,94)])
temp = forest.predict(X)
sum(temp == nptrd[:,-1]) / 61878.0

# Need to spend some time checking for overfit - using some elbow techiques maybe

# Working with the Test Data

ted = pd.read_csv(datapath+testfile)
npted = ted.values

Xt = pca.transform(npted[:,range(1,94)])

#output = forest.predict(npted[:,range(1,94)])
output = forest.predict(Xt)

a = pd.DataFrame(output)
a['id'] = a.index+1
a.columns = ['values','id']
a['count'] = 1
final = a.pivot(index='id',columns='values',values='count')
final = final.fillna(0)

final.to_csv(datapath+'sub.csv')