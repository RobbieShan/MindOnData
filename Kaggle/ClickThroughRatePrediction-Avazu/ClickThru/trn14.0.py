import os
import csv
#import matplotlib.pyplot as plt
import numpy as np
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
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
#import sys
import random

testrun = True

trainpath  = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/train/'
trainpath3  = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/train3/'
trainfile  = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/train.csv'
trainfile2  = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/train2.csv'


testpath   = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/test/'
testfile   = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/test.csv'
testfile2   = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/test2.csv'

outputpath = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/testoutput2/'
   
printflag = True
    
cols = ["C1","banner_pos","site_category", "device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21", "hour"]
cols2 = ["click","C1","banner_pos","site_category", "device_type","device_conn_type","C15","C16","C18","C19","C21"]
cols3 = ["C1","banner_pos","site_category", "device_type","device_conn_type","C15","C16","C18","C19","C21"]
cols4 = ["id","C1","banner_pos","site_category", "device_type","device_conn_type","C15","C16","C18","C19","C21"]



c1col = ["C1","click"] # 7 unique values [1001, 100
bpcol = ["banner_pos","click"] # 7 unique values [0, 1, 2, 3, 4, 5, 7]. Almost same spread across chunks
sccol = ["site_category","click"] # 26 unique values ['28905ebd', '75fa27f6', 'a72a0145', 'bcf865d9', '70fb0e29', '72722551', '335d28a8', '74073276', 'c0dd3be3', 'a818d37a', '5378d028', '0569f928', 'dedf689d', '8fd0aea4', 'f66779e6', '42a36e14', '110ab22d', '50e219e0', '9ccfa2ea', 'c706e647', '76b2941d', '6432c423', 'f028772b', 'da34532e', '3e814130', 'e787de0e']
dtcol = ["device_type","click"] # 5 unique values [0, 1, 2, 4, 5]
dctcol = ["device_conn_type","click"] # 4 unique values [0, 2, 3, 5]
c14col = ["C14","click"] # 2626 unique values
c15col = ["C15","click"] # 8 unique values [320, 1024, 300, 728, 480, 216, 120, 768]
c16col = ["C16","click"] # 9 unique values [480, 768, 36, 1024, 320, 50, 20, 90, 250]
c17col = ["C17","click"] # 435 unique values
c18col = ["C18","click"] # 4 unique values [0, 1, 2, 3]
c19col = ["C19","click"] # 68 unique values
c20col = ["C20","click"] # 172 unique values
c21col = ["C21","click"] # 60 unique values
hrcol = ["hour","click"]


vc1col = [1001, 1002, 1005, 1007, 1008, 1010, 1012]
vbpcol = [0, 1, 2, 3, 4, 5, 7]
vsccol = ['28905ebd', '75fa27f6', 'a72a0145', 'bcf865d9', '70fb0e29', '72722551', '335d28a8', '74073276', 'c0dd3be3', 'a818d37a', '5378d028', '0569f928', 'dedf689d', '8fd0aea4', 'f66779e6', '42a36e14', '110ab22d', '50e219e0', '9ccfa2ea', 'c706e647', '76b2941d', '6432c423', 'f028772b', 'da34532e', '3e814130', 'e787de0e']
vdtcol = [0, 1, 2, 4, 5]
vdctcol = [0, 2, 3, 5]
#vc14col = ["C14","click"] # 2626 unique values
vc15col = [320, 1024, 300, 728, 480, 216, 120, 768]
vc16col = [480, 768, 36, 1024, 320, 50, 20, 90, 250]
#vc17col = ["C17","click"] # 435 utrainfile2nique values
vc18col = [0, 1, 2, 3]
vc19col = [33, 34, 35, 38, 39, 41, 43, 45, 47, 161, 163, 167, 169, 171, 175, 289, 290, 291, 295, 297, 299, 303, 417, 419, 423, 425, 427, 431, 545, 547, 551, 553, 555, 559, 673, 675, 677, 679, 681, 683, 687, 801, 803, 809, 811, 813, 815, 935, 937, 939, 943, 1059, 1063, 1065, 1071, 1195,1315, 1319, 1327, 1447, 1451, 1575, 1583, 1711, 1831, 1835, 1839, 1959]
#vc20col = ["C20","click"] # 172 unique values
vc21col = [1, 171, 13, 15, 16, 17, 20, 23, 156, 157, 159, 32, 33, 35, 42, 43, 178, 46, 93, 48, 177, 91, 51, 52, 182, 61, 194, 195, 68, 69, 70, 71, 76, 204, 79, 82, 163, 212, 85, 90, 219, 221, 94, 95, 229, 100, 101, 102, 104, 108, 110, 111, 112, 116, 117, 246, 251, 253, 126, 255]
#vhrcol = ["hour","click"]



def splitfiles():
      
    fr = open(trainfile2,'r')
    frsize = os.path.getsize(trainfile2) - 300    

    #fr.lineno
    
    
    for i in range(0,15):
        fw = open(trainpath3 + str(i),'w+')
        for j in range(0,3000000):
            r = random.randint(0,frsize)
            fr.seek(r)
            line = fr.readline()
            line = fr.readline()
            fw.write(line)
        fw.close()
        
        
#train = pd.read_csv(trainfile, chunksize = 10000000, iterator = True)
#i = 1
#total = 0
#unq = []
#
#for chunk in train:
#    chunk = chunk[c19col]
#    currlist = chunk.groupby(c19col[0])
#    curr = len(currlist)
#    unq = unq + currlist.groups.keys()
#    total = total + curr
#    print "Chunk::", i, "::" , curr , "Cumulative::" , total
#    i = i+1
#print "final unique values:::" ,len(set(unq))


def transfiles(filepath,newfile,cols):
    train = pd.read_csv(filepath, chunksize = 5000000, iterator = True)
    
    f= open(newfile, 'w+')
    firstrun = True
    
    for chunk in train:
        chunk = chunk[cols]
        
        chunk['C1'] =               [ vc1col.index(i) if i in vc1col else len(vc1col) for i in chunk['C1']]
        chunk['banner_pos'] =       [ vbpcol.index(i) if i in vbpcol else len(vbpcol) for i in chunk['banner_pos']]
        chunk['site_category'] =    [ vsccol.index(i) if i in vsccol else len(vsccol) for i in chunk['site_category']]
        chunk['device_type'] =      [ vdtcol.index(i) if i in vdtcol else len(vdtcol) for i in chunk['device_type']]
        chunk['device_conn_type'] = [ vdctcol.index(i) if i in vdctcol else len(vdctcol) for i in chunk['device_conn_type']]
        chunk['C15'] =              [ vc15col.index(i) if i in vc15col else len(vc15col) for i in chunk['C15']]
        chunk['C16'] =              [ vc16col.index(i) if i in vc16col else len(vc16col) for i in chunk['C16']]
        chunk['C18'] =              [ vc18col.index(i) if i in vc18col else len(vc18col) for i in chunk['C18']]
        chunk['C19'] =              [ vc19col.index(i) if i in vc19col else len(vc19col) for i in chunk['C19']]
        chunk['C21'] =              [ vc21col.index(i) if i in vc21col else len(vc21col) for i in chunk['C21']]
        
        if firstrun == True:
            f.writelines(chunk.to_csv(index=False, header=True))
            firstrun = False
        else:
            f.writelines(chunk.to_csv(index=False, header=False))
    
    f.close()

#transfiles(trainfile,trainfile2,cols2)        
#transfiles(testfile,testfile2,cols4)

def main(): 

    print "###############################At Process Train Data by File########################################"
    train = pd.read_csv(trainfile2, chunksize = 3000000, iterator = True)
    
    enc = OneHotEncoder(n_values = np.array([8,8,27,6,5,9,10,5,69,61]))
    
#    enc = OneHotEncoder(n_values=500,categorical_features = "all")
    
#    enc = OneHotEncoder(n_values=70)
    
    loopcounter = 1   
    testpred = np.zeros(4577464) 
    idlist = []
    
    for chunk in train:
        trainY = chunk["click"]

        chunk = chunk[cols3]
#        chunk.drop("C14",axis=1,inplace=True)
#        chunk.drop("C17",axis=1,inplace=True)
#        chunk.drop("C20",axis=1,inplace=True)
#        chunk.drop("hour",axis=1,inplace=True)
    
#        chunk["site_category"] = [hash(i)%100 for i in chunk["site_category"]]
    
#        print chunk    
#        print np.array(chunk)
        vcd = enc.fit_transform(np.array(chunk)).toarray()
        clf = RandomForestClassifier(n_estimators=100,n_jobs=-1)

        clf.fit(vcd,trainY)

        test = pd.read_csv(testfile2, chunksize = 572183, iterator = True , usecols = cols4)
        
        temptestpred = np.array([])
        
        for chunk2 in test:
            if loopcounter == 1:
                idlist.append([i for i in chunk2.id.tolist()])
                
            chunk2 = chunk2[cols3]
            
#            chunk2.drop("C14",axis=1,inplace=True)
#            chunk2.drop("C17",axis=1,inplace=True)
#            chunk2.drop("C20",axis=1,inplace=True)
#            chunk2.drop("hour",axis=1,inplace=True)
            
#            chunk2["site_category"] = [hash(i)%100 for i in chunk2["site_category"]]

            tstvcd = enc.transform(np.array(chunk2)).toarray()
            temp = clf.predict_proba(tstvcd)
#            print len(temp[:,1])
            temptestpred = np.concatenate((temptestpred,temp[:,1]))
               
        testpred = testpred + temptestpred
#        if loopcounter == 200:
#                break
        print "loopcounter is::",loopcounter
        loopcounter = loopcounter + 1
    #    testpred = testpred / len(alltrainfiles)
    
    testpred = testpred / (loopcounter-1)
    
    idlist = sum(idlist,[])
    print "idlist length is.....", len(idlist)
    
    fop = open(outputpath + 'submission.csv', 'w+')
    fop.writelines('id,click\n')
    for i in range(0,len(idlist)):
        fop.writelines(str(idlist[i]) + ',' + str(round(testpred[i],4)) + '\n')
    
    fop.close()
#    print testpred
if __name__ == "__main__": main()
    
