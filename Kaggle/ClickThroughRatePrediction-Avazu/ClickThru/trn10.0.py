# Avazu CTR prediction
# SGD Logistic regression + one hot encoder. Score: 0.414
import pandas as pd
import numpy as np
from datetime import datetime, date, time
#from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDClassifier

trainfile  = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/train.csv'
trainpath  = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/train/'

testpath   = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/test/'
testfile   = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/test/test.csv'

outputpath = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/testoutput/'
outputfile = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/testoutput/submission.csv'

cols = ["C1","banner_pos","site_category", "device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21", "hour"]

from scipy import sparse


class OneHotEncoder():
    """
    OneHotEncoder takes data matrix with categorical columns and
    converts it to a sparse binary matrix doing one-of-k encoding.

    Parts of code borrowed from Paul Duan (www.paulduan.com)
    Licence: MIT (https://github.com/pyduan/amazonaccess/blob/master/MIT-LICENSE)
    """

    def __init__(self):
        self.keymap = None

    def fit(self, x):
        self.keymap = []
        for col in x.T:
            uniques = set(list(col))
            self.keymap.append(dict((key, i) for i, key in enumerate(uniques)))

    def partial_fit(self, x):
        """
        This method can be used for doing one hot encoding in mini-batch mode.
        """
        if self.keymap is None:
            self.fit(x)
        else:
            for i, col in enumerate(x.T):
                uniques = set(self.keymap[i].keys() + (list(col)))
                self.keymap[i] = dict((key, i) for i, key in enumerate(uniques))

    def transform(self, x):
        if self.keymap is None:
            self.fit(x)

        outdat = []
        for i, col in enumerate(x.T):
            km = self.keymap[i]
            num_labels = len(km)
            spmat = sparse.lil_matrix((x.shape[0], num_labels))
            for j, val in enumerate(col):
                if val in km:
                    spmat[j, km[val]] = 1
            outdat.append(spmat)
        outdat = sparse.hstack(outdat).tocsr()
        return outdat
        
# add two columns for hour and a weekday
def dayhour(timestr):
    d = datetime.strptime(str(x), "%y%m%d%H")
    return [float(d.weekday()), float(d.hour)]
 
enc = OneHotEncoder()
 
# Fit OneHotEncoder small batch at the time
# This implementation of encoder that supports partial fitting borrowed from Mahendra Kariya.
train = pd.read_csv(trainfile, chunksize = 1000000, iterator = True)
for chunk in train:
    chunk = chunk[cols]
    chunk = chunk.join(pd.DataFrame([dayhour(x) for x in chunk.hour], columns=["wd","hr"]))
    chunk.drop("hour", axis=1, inplace = True)
    enc.partial_fit(np.array(chunk))
 
# Train the classifier
clf = SGDClassifier(loss="log")
train = pd.read_csv(trainfile, chunksize = 1000000, iterator = True)
all_classes = np.array([0, 1])
for chunk in train:
    y_train = chunk["click"]
    chunk = chunk[cols]
    chunk = chunk.join(pd.DataFrame([dayhour(x) for x in chunk.hour], columns=["wd", "hr"]))
    chunk.drop("hour", axis=1, inplace = True)
    Xcat = enc.transform(np.array(chunk))
    clf.partial_fit(Xcat, y_train, classes=all_classes)
    
# Create a submission file
X_test = pd.read_csv(testfile, usecols = cols + ["id"])
X_test = X_test.join(pd.DataFrame([dayhour(x) for x in X_test.hour], columns=["wd", "hr"]))
X_test.drop("hour", axis=1, inplace = True)
 
X_enc_test = enc.transform(X_test)
 
y_pred = clf.predict_proba(X_enc_test)[:, 1]
with open(outputfile, "w") as f:
    f.write("id,click\n")
    for idx, xid in enumerate(X_test.id):
        f.write(str(xid) + "," + "{0:.10f}".format(y_pred[idx]) + "\n")
f.close()