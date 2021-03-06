from __future__ import division
import sys

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier as gbc


np.random.seed(17411)

datapath = '/home/robbie/Hacking/Kaggle/KaggleOttoProdClass/Data/'
trainfile = 'train.csv'
testfile = 'test.csv'

def logloss_mc(y_true, y_prob, epsilon=1e-15):
    """ Multiclass logloss
    This function is not officially provided by Kaggle, so there is no
    guarantee for its correctness.
    """
    # normalize
    y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    # get probabilities
    y = [y_prob[i, j] for (i, j) in enumerate(y_true)]
    ll = - np.mean(np.log(y))
    return ll


def load_train_data(path=None, train_size=0.8):
    path = sys.argv[1] if len(sys.argv) > 1 else path
    if path is None:
        try:
            # Unix
            df = pd.read_csv(datapath+trainfile)
        except IOError:
            # Windows
            df = pd.read_csv(datapath+trainfile)
    else:
        df = pd.read_csv(path)
        
    X = df.values.copy()
    np.random.shuffle(X)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X[:, 1:-1], X[:, -1], train_size=train_size,
    )
    print(" -- Loaded data.")
    return (X_train.astype(float), X_valid.astype(float),
            y_train.astype(str), y_valid.astype(str))


def load_test_data(path=None):
    path = sys.argv[2] if len(sys.argv) > 2 else path
    if path is None:
        try:
            # Unix
            df = pd.read_csv(datapath+testfile)
        except IOError:
            # Windows
            df = pd.read_csv(datapath+testfile)
    else:
        df = pd.read_csv(path)
    X = df.values
    X_test, ids = X[:, 1:], X[:, 0]
    return X_test.astype(float), ids.astype(str)


def train():
    X_train, X_valid, y_train, y_valid = load_train_data()
    # Number of trees, increase this to beat the benchmark ;)
   
#   n_estimators = 10
#    clf = RandomForestClassifier(n_estimators=n_estimators)
   
    
#    for r in np.arange(3,50,3):
#        
#        tclf = gbc(n_estimators=50, learning_rate=0.35,max_depth=5,max_features=0.7,min_samples_leaf=6,min_samples_split=r)    
#        print(" --------- Testing Stuff -------------", "min_samples_split=", r)
#        tclf.fit(X_train, y_train)
#        ty_prob = tclf.predict_proba(X_valid)
#    
#        tencoder = LabelEncoder()
#        ty_true = tencoder.fit_transform(y_valid)
#        assert (tencoder.classes_ == tclf.classes_).all()
#    
#        tscore = logloss_mc(ty_true, ty_prob)
#        print(" -- Multiclass logloss While Testing Stuff: {:.4f}.".format(tscore))
        
        
    
    clf = gbc(n_estimators=25, learning_rate=0.18,min_samples_leaf=6, max_features=0.8,subsample=0.9,verbose=2,max_depth=10)  
#    clf = gbc(n_estimators=20, learning_rate=.13, min_samples_leaf=6,subsample=.8,max_features=0.6,verbose=2,max_depth=20)    
    
    print(" -- Start training Random Forest Classifier.")
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_valid)

    
    print(" -- Finished training.")

    encoder = LabelEncoder()
    y_true = encoder.fit_transform(y_valid)
    assert (encoder.classes_ == clf.classes_).all()

    score = logloss_mc(y_true, y_prob)
    print(" -- Multiclass logloss on validation set: {:.4f}.".format(score))

    return clf, encoder


def make_submission(clf, encoder, path='my_submission.csv'):
    path = sys.argv[3] if len(sys.argv) > 3 else path
    X_test, ids = load_test_data()
    y_prob = clf.predict_proba(X_test)
    with open(path, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + list(map(str, probs.tolist())))
            f.write(probas)
            f.write('\n')
    print(" -- Wrote submission to file {}.".format(path))


def main():
    print(" - Start.")
    clf, encoder = train()
    make_submission(clf, encoder)
    print(" - Finished.")


if __name__ == '__main__':
    main()