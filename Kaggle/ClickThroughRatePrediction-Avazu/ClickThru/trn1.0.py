import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import itertools
from collections import Counter
import math
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import sys

testrun = True

def traindata(a_traindata,a_yvalues):
    print "###############################In TrainData########################################"
    rfc = RandomForestClassifier(n_estimators = 1000)
    rfc.fit(a_traindata,a_yvalues)
    return rfc


def importdata(a_projpath,a_trainfilename,testortrain): 
    print "###############################In ImportData########################################"
    fhandle = open(a_projpath + '/' + a_trainfilename, 'r')
    rows = csv.reader(fhandle,delimiter=',')
    clickdata = {}
    
    for row in rows:
        if row[0] == 'id':
            continue
        
        clickdata[row[0]] = [] 
#        print row
#        print clickdata
        if testortrain == 'test':
            for i in range(1,23):
                clickdata[row[0]].append(row[i])            
        else:
            for i in range(1,24):
                clickdata[row[0]].append(row[i])      
    return clickdata


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
    
    traindata = importdata(projpath,trainfilename,"train")
    testdata = importdata(projpath,testfilename,"test")
    
    fittraindata = traindata(np.array(traindata.values())[:,range(2,23)]
                            ,np.array(traindata.values())[:,0])
    
    print fittraindata.predict(testdata.values()[:,range(1,22)])

#    writepreds(outputpath,traindata)  
    
    
    

if __name__ == "__main__": main()