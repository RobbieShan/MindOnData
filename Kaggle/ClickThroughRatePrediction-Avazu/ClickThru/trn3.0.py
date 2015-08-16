#import os
import csv
#import matplotlib.pyplot as plt
#import numpy as np
#import scipy as sp
#import itertools
#from collections import Counter
#import math
#from sklearn.cluster import KMeans
#from sklearn.cluster import DBSCAN
#from sklearn.cluster import AgglomerativeClustering
#from sklearn import metrics
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
#import sys

testrun = True
trainY = []

def traindata(a_traindata,a_yvalues):
    print "###############################In TrainData########################################"
    rfc = RandomForestClassifier(n_estimators = 10)
    rfc.fit(a_traindata,a_yvalues)
    return rfc


def writepreds(a_outputpath,a_modelclickdata):
    print "###############################In WritePreds########################################"
    if testrun == True:
            f = open(a_outputpath + 'PredictsTest.csv', 'w')  
    else: 
            f = open(a_outputpath + 'Predicts.csv', 'w')       
    f.writelines('id,click\n')    
    for arow in a_modelclickdata:
        f.writelines(str(arow[0]) + ',' + str(arow[1]) + '\n')        
    f.close()    
    
    
def main(): 
    projpath = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/'
    trainfilename = 'train.csv'
    testfilename = 'test.csv' 
    outputpath = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/'
    
    print "###############################In ImportData########################################"
    fhandle = open(projpath + '/' + trainfilename, 'r')
    
    rows = csv.reader(fhandle,delimiter=',')
    clickdata = {}
    vec = DictVectorizer()    
    i = 0
    
    
    
        for row in rows:
            print "Row #::" , i
            i = i+1
            if row[0] == 'id':
                headers = row[1:]
                continue
            
            clickdata[row[0]] = {}
            
            h = 1
            for header in headers:
                if header == 'click':
                    trainY.append(row[h])
                else:    
                    clickdata[row[0]][header] = row[h]
                h = h+1

    vcd = vec.fit_transform(clickdata.values()).toarray()
    clickdata = {}
    
#    testdata = importdata(projpath,testfilename)
    
#    fittraindata = traindata(traindata,trainY)

if __name__ == "__main__": main()
    
