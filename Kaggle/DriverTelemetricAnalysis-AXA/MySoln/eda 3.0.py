import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import math
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


#################################################################################################################

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
#    print "v1: ", v1, " v2: ", v2
    nr = round(dotproduct(v1, v2),2)
    dr = round(length(v1) * length(v2),2)
    if v1 == [0.0,0.0] or  v2 == [0.0,0.0] or v1==v2 or dr < nr:
        return 0.0
    else:
        return math.acos( nr / dr )
  
#################################################################################################################

def get_all_data(a_projpath , a_folder):
    
    drivers = {}
    allfiles = os.listdir(a_projpath + a_folder)
    drivers[a_folder] = {}
    drivers[a_folder]['DStats'] = np.array([],float)  # Folder Level Descriptive Statistics
    
    for afile in allfiles:
        fhandle = open(a_projpath + a_folder + '/' + afile, 'r')
#        print "File:" + afile
        drivers[a_folder][afile] = {}
        drivers[a_folder][afile]['Data'] = []
        drivers[a_folder][afile]['ACosDel'] = [] 
        drivers[a_folder][afile]['DistDel'] = [] 
        drivers[a_folder][afile]['DStats'] = {}
        
        
        rows = csv.reader(fhandle,delimiter=',')
        lastpoint = [0.0,1.0]
        i = 0
        for arow in rows:
            if arow[0] != 'x':
                curpoint = [float(arow[0]),float(arow[1])]
#                print "Index::" , i, "LastPoint::", lastpoint, "::CurPoint:::" , curpoint
                i = i+1
                drivers[a_folder][afile]['Data'].append(curpoint)
                drivers[a_folder][afile]['ACosDel'].append(angle(lastpoint,curpoint))
                drivers[a_folder][afile]['DistDel'].append(math.hypot((curpoint[0]-lastpoint[0]),(curpoint[1]-lastpoint[1])))
                lastpoint = curpoint
                    
        drivers[a_folder][afile]['ACosDel'] = np.array(drivers[a_folder][afile]['ACosDel'])
        drivers[a_folder][afile]['DistDel'] = np.array(drivers[a_folder][afile]['DistDel'])
        drivers[a_folder][afile]['DStats']['AvgDistDel'] = np.average(drivers[a_folder][afile]['DistDel'])
        drivers[a_folder][afile]['DStats']['AvgACosDel'] = np.average(drivers[a_folder][afile]['ACosDel'])
        drivers[a_folder][afile]['DStats']['SDevDistDel'] = np.std(drivers[a_folder][afile]['DistDel'])
        drivers[a_folder][afile]['DStats']['SDevACosDel'] = np.std(drivers[a_folder][afile]['ACosDel'])    
        drivers[a_folder][afile]['DStats']['TotalDist'] = np.sum(drivers[a_folder][afile]['DistDel'])
        drivers[a_folder][afile]['DStats']['TotalTime'] = len(drivers[a_folder][afile]['Data'])   
        
        drivers[a_folder]['DStats'] = np.concatenate((drivers[a_folder]['DStats']
                                        ,[drivers[a_folder][afile]['DStats']['AvgDistDel']]
                                        ,[drivers[a_folder][afile]['DStats']['AvgACosDel']] 
                                        ,[drivers[a_folder][afile]['DStats']['SDevDistDel']] 
                                        ,[drivers[a_folder][afile]['DStats']['SDevACosDel']] 
                                        ,[drivers[a_folder][afile]['DStats']['TotalDist']]
                                        ,[drivers[a_folder][afile]['DStats']['TotalTime']]
                                        ))
        drivers[a_folder][afile]['Data'] = []
        drivers[a_folder][afile]['ACosDel'] = []
        drivers[a_folder][afile]['DistDel'] = []
                                
#        drivers[a_folder][afile]['DStats'] = {}
        fhandle.close()
#        print drivers[a_folder]
#    print "#############################DStats Before Scaling Below ##################################################"
#    print drivers[a_folder]['DStats']                    
    drivers[a_folder]['DStats'] = drivers[a_folder]['DStats'].reshape(len(allfiles),len(drivers[a_folder][afile]['DStats']))

    
#    print "#############################DStats After Scaling Below ##################################################"
#    print drivers[a_folder]['DStats']
#    print "len(allfiles)" , len(allfiles)
#    print "lendrivers:a_folder::afile::DStats" , len(drivers[a_folder][afile]['DStats'])
#    print drivers[a_folder]['DStats']
#    filelist = [x for x in drivers[a_folder].keys()]
    
 #   drivers[a_folder]['DStats']['triplen'] = sp.mean(x for x in drivers[a_folder][afile]['DStats']['triplen'])
    
    return drivers[a_folder]


#################################################################################################################
def cluster_driver(a_driver):
    
#    print a_driver['DStats']
#    print "#############################DStats Above##################################################"

    X = StandardScaler().fit_transform(a_driver['DStats'])
    
#    print X
#    print "DStats are.....::" , a_driver['DStats']
#    print "X is...........::" , X
#    print "############################Scaled X Above###################################################"
    
    db = DBSCAN(eps=0.6, min_samples=5).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print "###############################################################################"
#    print('Estimated number of clusters: %d' % n_clusters_)
#    print 'Count of Predicts::', len(X)
#    print("Silhouette Coefficient: %0.3f"    % metrics.silhouette_score(X, labels))
#    print "##############################DBSCAN  X Below#################################################"
#    print X    G:/Continuing Education/Research & Presentations/Self - Machine Learning/Kaggle/DriverTelemetricAnalysis-AXA/'
#    try:
    return (metrics.silhouette_samples(X, labels)+1)/2
#    except ValueError:
#        pass

#################################################################################################################

def calc_metrics(a_projpath, a_datafolder):
    allfolders = os.listdir(a_projpath + a_datafolder)
    drivers = {}
#    driverstats = {}
    f = open(a_projpath + 'Predicts.csv', 'w')    
    f.writelines('driver_trip,prob\n')
    
    cnt = 0
    
    for afolder in allfolders:
#        plt.subplot(1,len(allfolders),allfolders.index(afolder)+1)
        print "# is::" , cnt , "::Folder is::", afolder
        cnt = cnt +1
        drivers[afolder] = get_all_data(a_projpath + a_datafolder , afolder)   
#        print "Driver: " + afolder + " ", drivers[afolder] 
        drivers[afolder]['Predicts'] = cluster_driver(drivers[afolder])
#        print drivers[afolder]['Predicts']
        allfiles = os.listdir(a_projpath + a_datafolder + afolder)
        i = 0
        for afile in allfiles:
#            f.writelines(str(afile) + ',' + str(drivers[afolder]['Predicts'][i]) + '\n')
            f.writelines(str(afolder) + '_' + str(afile).split('.')[0] + ',' + str(round(drivers[afolder]['Predicts'][i],2)) + '\n')
#            print "I is ::::::::::" , i, "A File is :::::::::" , afile
#            print str(afile) + ',' + str(drivers[afolder]['Predicts'][i])
            i = i+1        
    f.close()    
#        write_predicts(drivers[[afolder]['Predicts'],drivers[afolder]])
#        plt.show()

#################################################################################################################

def main(): 
    projpath = '/home/robbie/Hacking/Kaggle/'
    datafolder = 'drivers/'
    calc_metrics(projpath , datafolder)

if __name__ == "__main__": main()
