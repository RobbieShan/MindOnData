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
#import sys
import random

testrun = True
trainfile  = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/train.csv'
trainpath  = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/train/'
testpath   = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/test/'
outputpath = '/home/robbie/Hacking/Kaggle/ClickThroughRatePrediction-Avazu/Data/Raw/testoutput2/'
   
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
                    trainY.append(tdict.pop('click'))
                    tdict.pop('id')
                    clickdata.append(tdict)
             
#                    if printflag == True: print tdict
                    
                print "######################At Dict Vectorizer for file:::" , loopcounter, "#####################"
                
#                if printflag == True:  print clickdata
                vec = DictVectorizer()
                vcd = vec.fit_transform(clickdata).toarray()
                clickdata = []
                
                print "###############################At Classifer########################################"      
                clf = RandomForestClassifier(n_estimators=5,n_jobs=-1)
                clf.fit(vcd, trainY)
#                if printflag == True: print clf.predict(vcd) == trainY
#                print sum(int(x) for x in trainY)
#                print sum(clf.predict(vcd) == trainY)
#                print clf.oob_score_
                
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
                
                if loopcounter == 300:
                        break
                
                loopcounter = loopcounter + 1
#    testpred = testpred / len(alltrainfiles)
    testpred = testpred / loopcounter
    
    fop = open(outputpath + 'submission.csv', 'w+')
    fop.writelines('id,click\n')
    for i in range(0,len(idlist)):
        fop.writelines(str(idlist[i]) + ',' + str(round(testpred[i],4)) + '\n')
    
    fop.close()
#    print testpred
if __name__ == "__main__": main()
    
