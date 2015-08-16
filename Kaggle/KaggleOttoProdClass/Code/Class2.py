import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.decomposition import PCA as PCA
from sklearn.cross_validation import train_test_split as tts



datapath = 'G:/Continuing Education/Research & Presentations/Self - Machine Learning/Kaggle/OttoProductClassification/Data/'
trainfile = 'train.csv'
testfile = 'test.csv'

trd = pd.read_csv(datapath+trainfile)
trd = trd.values

# Split data into training and cross-validation dataset
nptrd, npcvd = tts(trd,test_size=0.33)


# Train the model

pca = PCA(n_components=40)
pca.fit(nptrd[:,range(1,94)])
X = pca.transform(nptrd[:,range(1,94)])
PCAExplained = sum(pca.explained_variance_ratio_)

# Most of the features are highly skewed i.e. their 75% value ranges when we do td.describe() is 0 while their max is much higher. 
# This indicates that only a few values are non-zero for most features.
# This could mean that these features are actually categorical variables that are encoded in the test data.. could .. not sure

forest = rfc(n_estimators=500,criterion = 'entropy' , n_jobs=-1,min_samples_split=5,min_samples_leaf=5,max_depth=20)
#forest = forest.fit(nptrd[:,range(1,94)],nptrd[:,-1])
forest = forest.fit(X,nptrd[:,-1])

#temp = forest.predict(nptrd[:,range(1,94)])
temp = forest.predict(X)
TrainError = sum(temp == nptrd[:,-1]) / (len(nptrd)*1.0)

# Need to spend some time checking for overfit - using some elbow techiques maybe


# Cross validate the model using the cross validation dataset

XCv = pca.transform(npcvd[:,range(1,94)])
#output = forest.predict(npted[:,range(1,94)])
outputCv = forest.predict(XCv)
CrossValidError = sum(outputCv == npcvd[:,-1]) / (len(npcvd)*1.0)


print('############################################')
print('PCA eExplained        ::', PCAExplained)
print('Training Error        ::', TrainError)
print('Cross Validation Error::', CrossValidError)
print('############################################')

# Score the Test Data using the best model

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