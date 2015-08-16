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
from sklearn.ensemble import RandomForestClassifier
#import sys
import random

testrun = True

   
def main(): 

    projpath = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/'
    outputpath = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/'

    trainfilename = 'train.csv'
    testfilename = 'test.cs    rows = csv.reader(fhandle,delimiter=',')v' 
    trialfilename = 'trial.csv'
    
    bootstrapsize = 50000
    printflag = False

    print "###############################At ImportData########################################"    

    f = open(projpath + '/' + trainfilename, 'r')   
# Get Headers #
    f.seek(0) 
    headers = f.readline().rstrip().split(",")
    if printflag == True: print headers
    trainY = []
    clickdata = []
 
    
    f.seek(0,2)
    filesize = f.tell() - 300

    
    for i in range(0,bootstrapsize):   
        if printflag == True: print '#####################################'
        offset = random.randrange(filesize)
        f.seek(offset)                  #go to random position
        f.readline()                    # discard - bound to be partial line  
        f.readline()
        random_line = f.readline().rstrip().split(",")      # bingo!
#        print offset        
        if printflag == True: print random_line
        
        tdict = dict(zip(headers, random_line))
        trainY.append(tdict.pop('click'))
        tdict.pop('id')
        clickdata.append(tdict)
 
        if printflag == True: print tdict
        print "i is::" , i
    
    f.close()
    print "###############################At Dict Vectorizer########################################"      
    if printflag == True:  print clickdata
    vec = DictVectorizer()
    vcd = vec.fit_transform(clickdata).toarray()
    clickdata = {}
    
    print "###############################At Classifer########################################"      
    clf = RandomForestClassifier(n_estimators=10,max_leaf_nodes = 2)
    clf.fit(vcd, trainY)
    if printflag == True: print clf.predict(vcd) == trainY
    
    print "###############################Import Test Data########################################"      
    ftest = open(projpath + '/' + testfilename, 'r') 
    rows = csv.reader(ftest,delimiter=',')
    testdata = []
    
    i = 0  
    
    for row in rows:
#        print "Row #::" , i
        i = i+1
        if row[0] == 'id':
            headers = row[1:]
            continue
        else:
            tdict = dict(zip(headers, row))
            tdict.pop('id')
            testdata.append(tdict)
    
    print "###############################Dict Vectorize Test Data########################################"      
    tstvcd = vec.transform(testdata)
#    testdata = importdata(projpath,testfilename)
    
    print "###############################Predict Test Data########################################"      
    testpred = clf.predict(tstvcd)
#    fittraindata = traindata(traindata,trainY)

if __name__ == "__main__": main()
    
