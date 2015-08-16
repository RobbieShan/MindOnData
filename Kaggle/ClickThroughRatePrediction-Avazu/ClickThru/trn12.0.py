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
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
#import sys
import random


############################################################################################

allcols = ["C1","banner_pos","site_category", "device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21", "hour"]
c1col = ["C1","click"]
bpcol = ["banner_pos","click"] # 7 unique values [0, 1, 2, 3, 4, 5, 7]. Almost same spread across chunks
sccol = ["site_category","click"] # 26 unique values ['28905ebd', '75fa27f6', 'a72a0145', 'bcf865d9', '70fb0e29', '72722551', '335d28a8', '74073276', 'c0dd3be3', 'a818d37a', '5378d028', '0569f928', 'dedf689d', '8fd0aea4', 'f66779e6', '42a36e14', '110ab22d', '50e219e0', '9ccfa2ea', 'c706e647', '76b2941d', '6432c423', 'f028772b', 'da34532e', '3e814130', 'e787de0e']
dtcol = ["device_type","click"] # 4 unique values [0, 1, 4, 5]
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


train = pd.read_csv(trainfilepath, chunksize = 1000000, iterator = True)
i = 1
total = 0
unq = []

for chunk in train:
chunk = chunk[c21col]
currlist = chunk.groupby(c21col[0])
curr = len(currlist)
unq = unq + currlist.groups.keys()
total = total + curr
print "Chunk::", i, "::" , curr , "Cumulative::" , total
i = i+1
print "final unique values:::" ,len(set(unq))

############################################################################################

testrun = True
trainfile  = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/train.csv'
trainpath  = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/train/'
testpath   = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/test/'
outputpath = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/testoutput/'
   
#    bootstrapsize = 30000
printflag = True
    

def splitfiles():
      
    fr = open(trainfile,'r')
    frsize = os.path.getsize(trainfile) - 300    

    #fr.lineno
    
    
    for i in range(0,1500):
        fw = open(trainpath + str(i),'w+')
        for j in range(0,30000):
            r = random.randint(0,frsize)
            fr.seek(r)
            line = fr.readline()
            line = fr.readline()
#            print line
            fw.write(line)
        fw.close()
   
def main(): 


#    opfc = 0
    
    print "###############################At Process Train Data by File########################################"
    # Get Train Data Header from Mega File
    fhead = open('/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/train.csv')
    fhead.seek(0) 
    headers = fhead.readline().rstrip().split(",")
#    if printflag == True: print headers
        
    # Go after all the Train Files
    alltrainfiles = os.listdir(trainpath)
#    testpred = np.zeros(40428967)                
    testpred = np.zeros(4577464)   
#    testpred = np.zeros(89999)
    loopcounter = 1                
    
    for atrainfile in alltrainfiles:

        trainY = []
        clickdata = []
        
        for line in open(trainpath + atrainfile, 'r'):
            
            random_line = line.rstrip().split(",")
            tdict = dict(zip(headers, random_line))
            tdict.pop('C14')
            tdict.pop('C17')
            tdict.pop('C20')
            trainY.append(tdict.pop('click'))
            tdict.pop('id')
            clickdata.append(tdict)
     
#                    if printflag == True: print tdict
            
        print "######################At Dict Vectorizer for file:::" , loopcounter, "#####################"
        
#                if printflag == True:  print clickdata
        vec = DictVectorizer()
        vcd = vec.fit_transform(clickdata).toarray()
        clickdata = []
        
#                print "###############################At Tree Classifer########################################"      
#                clf = RandomForestClassifier(n_estimators=5,n_jobs=-1)
#                clf.fit(vcd, trainY)
##                if printflag == True: print clf.predict(vcd) == trainY
##                print sum(int(x) for x in trainY)
##                print sum(clf.predict(vcd) == trainY)
##                print clf.oob_score_
        
        print "############################At PassiveAggressive Classifer###################################"      
        clf = SGDClassifier(penalty = 'l1', n_jobs=-1)
        clf.partial_fit(vcd, trainY,[0,1])
#                if printflag == True: print clf.predict(vcd) == trainY
#                print sum(int(x) for x in trainY)
#                print sum(clf.predict(vcd) == trainY)
#                print clf.oob_score_
        loopcounter = loopcounter + 1
                
    print "###############################Import Test Data########################################"      
    
    testheaders = [headers[0]] + headers[2:]
    
    alltestfiles = os.listdir(testpath)
    
    temptestpred = np.array([])
    idlist = []
    innercounter = 1 
    for atestfile in alltestfiles:
                    
        testclickdata = []
        
        for line in open(testpath + atestfile, 'r'):
            
            random_line = line.rstrip().split(",")
            tdict = dict(zip(testheaders, random_line))
            tdict.pop('C14')
            tdict.pop('C17')
            tdict.pop('C20')
            idlist.append(tdict.pop('id'))
            testclickdata.append(tdict)
            
#                                    if printflag == True: print tdict
        
        print "------------------Dict Vectorize Test Data--------------------------------"      
        tstvcd = vec.transform(testclickdata).toarray()
    #    testdata = importdata(projpath,testfilename)
#                                print tstvcd
        
        print "-------------------Predict Test Data::", innercounter, "-----------------------------" 
        temp = clf.predict_proba(tstvcd)
#                                print "Type is::" , type(temp) , "     Shape is:::" , temp.shape
#                                print temp100
        temptestpred = np.concatenate((temptestpred,temp[:,1]))
#                                print clf.classes_
#                                print temptestpred
#                                print temptestpred[:,1]
        
#                                fop = open(outputpath + 'output_' + atestfile + str(opfc), 'w+')
#                                fop.write(testpred)
#                                opfc = opfc + 1
        
        innercounter = innercounter + 1
    testpred = testpred + temptestpred
    
#    if loopcounter == 300:
#            break
    

#    testpred = testpred / len(alltrainfiles)
#    testpred = testpred / loopcounter
    
    fop = open(outputpath + 'submission.csv', 'w+')
    fop.writelines('id,click\n')
    for i in range(0,len(idlist)):
        fop.writelines(str(idlist[i]) + ',' + str(round(testpred[i],4)) + '\n')
    
    fop.close()
#    print testpred
if __name__ == "__main__": main()
    
