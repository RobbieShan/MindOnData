import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import itertools
import math
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys


plotflag = False
testrun = False

#################################################################################################################

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
#    print "v1: ", v1, " v2: ", v2
    nr = round(dotproduct(v1, v2),2)
    dr = round(length(v1) * length(v2),2)
    if dr <= nr:
        return 0.0
    else:
        return math.acos( nr / dr )

#################################################################################################################

def scatterplot_matrix(data, names, **kwargs):
    """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""
    numvars, numdata = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]:
            axes[x,y].plot(data[x], data[y], **kwargs)

    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction', 
                ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)

    return fig

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
        
        secondlastpoint = [0.0,0.0]
        lastpoint = [0.0,1.0]

        i = 0
        
        for arow in rows:
            
            if arow[0] == 'x':
                continue
            
            if arow[0] == arow[1] == 0:
               i = i+1
               continue
           
            elif i == 1:
                curpoint = [(float(arow[0])),(float(arow[1]))]
#                print "Index::" , i, "LastPoint::", lastpoint, "::CurPoint:::" , curpoint
                v2 = [lastpoint[0]-secondlastpoint[0],lastpoint[1]-secondlastpoint[1]]
                
                lastpoint = [0.0,0.0]    
                v1 = [curpoint[0]-lastpoint[0],curpoint[1]-lastpoint[1]]              
            else:
                curpoint = [(float(arow[0])),(float(arow[1]))]
    #                print "Index::" , i, "LastPoint::", lastpoint, "::CurPoint:::" , curpoint
                v2 = [lastpoint[0]-secondlastpoint[0],lastpoint[1]-secondlastpoint[1]]
                v1 = [curpoint[0]-lastpoint[0],curpoint[1]-lastpoint[1]]
            
            drivers[a_folder][afile]['Data'].append(curpoint)
            drivers[a_folder][afile]['ACosDel'].append(angle(v1,v2))
            drivers[a_folder][afile]['DistDel'].append(math.hypot((curpoint[0]-lastpoint[0]),(curpoint[1]-lastpoint[1])))
    
            secondlastpoint = lastpoint
            lastpoint = curpoint
            i = i+1    
                    
        if plotflag == True:
            plt.subplot(1,1,1)
            plt.plot( [x[0] for x in drivers[a_folder][afile]['Data']], [x[1] for x in drivers[a_folder][afile]['Data']])  
            plt.grid(True)
            plt.title("Driver: Data " + a_folder) 
#            plt.subplot(1,3,2)
#    #        plt.plot(drivers[a_folder][afile]['DStats']['AvgDistDel'],np.random.random()/2,'b^')  
#            plt.plot(drivers[a_folder][afile]['DStats']['AvgDistDel'],drivers[a_folder][afile]['DStats']['AvgACosDel'],'b^')  
#            plt.grid(True)
#            plt.title("Driver: AvgDistDel " + a_folder)
#                    
#            plt.subplot(1,3,3)
#            plt.plot(drivers[a_folder][afile]['DStats']['AvgACosDel'],np.random.random()/2,'b^')
#            plt.grid(True)
#            plt.title("Driver: AvgACosDel " + a_folder)
            
            
        drivers[a_folder][afile]['ACosDel'] = np.array(drivers[a_folder][afile]['ACosDel'])
        drivers[a_folder][afile]['DistDel'] = np.array(drivers[a_folder][afile]['DistDel'])
        
        masACosDel = np.ma.masked_equal(drivers[a_folder][afile]['ACosDel'],0)        
        masDistDel = np.ma.masked_equal(drivers[a_folder][afile]['DistDel'],0)        
        
        drivers[a_folder][afile]['DStats']['AvgDistDel'] = masDistDel.mean()
        drivers[a_folder][afile]['DStats']['AvgACosDel'] = masACosDel.mean()
        drivers[a_folder][afile]['DStats']['SDevDistDel'] = masDistDel.std()
        drivers[a_folder][afile]['DStats']['SDevACosDel'] = masACosDel.std()    
        drivers[a_folder][afile]['DStats']['TotalDist'] = masDistDel.sum()   
        drivers[a_folder][afile]['DStats']['TotalTime'] = len(drivers[a_folder][afile]['Data'])                                          
#        drivers[a_folder][afile]['DStats']['KurtDistDel'] = sp.stats.kurtosis(drivers[a_folder][afile]['DistDel'][masDistDel!=True])   
        drivers[a_folder][afile]['DStats']['SkewDistDel'] = sp.stats.skew(drivers[a_folder][afile]['DistDel'][masDistDel!=True])   
#        drivers[a_folder][afile]['DStats']['KurtACosDel'] = sp.stats.kurtosis(drivers[a_folder][afile]['ACosDel'][masACosDel!=True])   
        drivers[a_folder][afile]['DStats']['SkewACosDel'] = sp.stats.skew(drivers[a_folder][afile]['ACosDel'][masACosDel!=True])   
        
        
        drivers[a_folder]['DStats'] = np.concatenate((drivers[a_folder]['DStats']
                                        ,[drivers[a_folder][afile]['DStats']['AvgDistDel']]
                                        ,[drivers[a_folder][afile]['DStats']['AvgACosDel']] 
                                        ,[drivers[a_folder][afile]['DStats']['SDevDistDel']] 
                                        ,[drivers[a_folder][afile]['DStats']['SDevACosDel']] 
                                        ,[drivers[a_folder][afile]['DStats']['TotalDist']]
                                        ,[drivers[a_folder][afile]['DStats']['TotalTime']]
#                                        ,[drivers[a_folder][afile]['DStats']['KurtDistDel']]
                                        ,[drivers[a_folder][afile]['DStats']['SkewDistDel']]
#                                        ,[drivers[a_folder][afile]['DStats']['KurtACosDel']]
                                        ,[drivers[a_folder][afile]['DStats']['SkewACosDel']]
                                        ))
        
        drivers[a_folder][afile]['Data'] = []
        drivers[a_folder][afile]['ACosDel'] = []
        drivers[a_folder][afile]['DistDel'] = []
        
        drivers[a_folder][afile]['DStats']['AvgDistDel'] = []
        drivers[a_folder][afile]['DStats']['AvgACosDel'] = []
        drivers[a_folder][afile]['DStats']['SDevDistDel'] = []
        drivers[a_folder][afile]['DStats']['SDevACosDel'] = []    
        drivers[a_folder][afile]['DStats']['TotalDist'] = []    
        drivers[a_folder][afile]['DStats']['TotalTime'] = []                                          
#        drivers[a_folder][afile]['DStats']['KurtDistDel'] = []  
        drivers[a_folder][afile]['DStats']['SkewDistDel'] = []  
#        drivers[a_folder][afile]['DStats']['KurtACosDel'] = []   
        drivers[a_folder][afile]['DStats']['SkewACosDel'] = []  
                                                           
                
#        drivers[a_folder][afile]['DStats'] = {}
        fhandle.close()
#        print drivers[a_folder]
#    print "#############################DStats Before Scaling Below ##################################################"
#    print drivers[a_folder]['DStats']                    
    drivers[a_folder]['DStats'] = drivers[a_folder]['DStats'].reshape(len(allfiles),len(drivers[a_folder][afile]['DStats']))
    
#    drivers[a_folder]['Baseline'] = np.mean(drivers[a_folder]['DStats'],axis=0)
    
    drivers[a_folder]['Baseline'] = sp.stats.mode(drivers[a_folder]['DStats'])[0][0]
#    print drivers[a_folder]['Baseline']
    
#    sys.stdout = open('a_projpath' +'output.txt','w') 
#    print drivers[a_folder]

    
    if plotflag == True:
        
        fig = scatterplot_matrix(np.transpose(drivers[a_folder]['DStats'])
                                                , ['AvgDistDel'
                                                , 'AvgACosDel'
                                                , 'SDevDistDel'
                                                , 'SDevACosDel'
                                                ,'TotalDist'
                                                ,'TotalTime'
#                                                ,'KurtDistDel'
                                                ,'SkewDistDel'
                                                ,'KurtACosDel'
#                                                ,'SkewACosDel'
                                                ]
                                                ,linestyle='none', marker='o', color='black', mfc='none')
        fig.suptitle('Simple Scatterplot Matrix')
        plt.show()
    
#    print "####################['AvgDistDel', 'AvgACosDel', 'SDevDistDel', 'SDevACosDel','TotalTime','SkewDistDel','SkewACosDel']#########DStats After Scaling Below ##################################################"
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
#    print "#############################DStats Above#################################ValueError: zero-size array to reduction operation minimum which has no identity#################"

#    sys.stdout = open('a_projpath' +'output.txt','w')
#    print a_driver['DStats']
    
    X = StandardScaler().fit_transform(a_driver['DStats'])
    

    
#    print X
#    print "DStats are.....::" , a_driver['DStats']
#    print "X is...........::" ,['AvgDistDel', 'AvgACosDel', 'SDevDistDel', 'SDevACosDel','TotalTime','SkewDistDel','SkewACosDel'] X
#    print "############################Scaled X Above###################################################"
    
    pca = PCA(n_components=5)
    Xpca = pca.fit(X).transform(X)
    
    if plotflag == True:
        
        fig = scatterplot_matrix(np.transpose(Xpca)
                                                , ['PC1'
                                                , 'PC2'
                                                , 'PC3'
                                                , 'PC4'
#                                                ,'PC5'
                                                ]
                                                ,linestyle='none', marker='o', color='black', mfc='none')
        fig.suptitle('Simple Scatterplot Matrix')
        plt.show()
        

    db = KMeans(n_clusters=1,n_jobs = -1).fit(Xpca)
    
#    db = DBSCAN(eps=0.5).fit(Xpca)
    
#    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
#    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print "###############################################################################"
#    print('Estimated number of clusters: %d' % n_clusters_)
#    print 'Count of Predicts::', len(X)
#    print("Silhouette Coefficient: %0.3f"    % metrics.silhouette_score(Xpca, labels))
  
    
    print "% Variance Explaned: %0.3f" , sum(pca.explained_variance_ratio_)
#    print "##############################DBSCAN  X Below#################################################"
#    print X    G:/Continuing Education/Research & Presentations/Self - Machine Learning/Kaggle/DriverTelemetricAnalysis-AXA/'
#    try:
    
    return (1- (db.transform(Xpca)/max(db.transform(Xpca))))
#    return (metrics.silhouette_samples(X, labels)+1)/2
    
    
#    except ValueError:
#        pass

#################################################################################################################

def chisq_driver(a_driver):
    return sp.stats.chisquare(a_driver['DStats'],a_driver['Baseline'],axis=1)[1]

#################################################################################################################

def jaccard_driver(a_driver):
    
    a_driver['DStats'] = (a_driver['DStats']*100).round()
    a_driver['Baseline'] = (a_driver['Baseline']*100).round()
    a_driver['Predicts'] = []
    
    for i in range (0,len(a_driver['DStats'])):
        a_driver['Predicts'].append(metrics.jaccard_similarity_score(a_driver['DStats'][i],a_driver['Baseline']))
                
    
    return a_driver['Predicts']

#################################################################################################################
metric="mahalanobis"
def pearson_driver(a_driver):
    
#    a_driver['DStats'] = (a_driver['DStats']*100).round()
#    a_driver['Baseline'] = (a_driver['Baseline']*100).round()
    a_driver['Predicts'] = []
    
    for i in range (0,len(a_driver['DStats'])):
        a_driver['Predicts'].append(sp.stats.pearsonr(a_driver['DStats'][i],a_driver['Baseline'])[1])
                
    
    return a_driver['Predicts']
    
#################################################################################################################

def calc_metrics(a_projpath, a_datafolder):
    allfolders = os.listdir(a_projpath + a_datafolder)
    drivers = {}
#    driverstats = {}

    if testrun == True:
            f = open(a_projpath + 'PredictsTest.csv', 'w')    
    else: 
            f = open(a_projpath + 'Predicts.csv', 'w')
    
    f.writelines('driver_trip,prob\n')
            
    cnt = 0
    
    for afolder in allfolders:
#        plt.subplot(1,len(allfolders),allfolders.index(afolder)+1)
        print "# is::" , cnt , "::Folder is::", afolder
        cnt = cnt +1
        drivers[afolder] = get_all_data(a_projpath + a_datafolder , afolder)   
#        print "Driver: " + afolder + " ", drivers[afolder] 
        
#        drivers[afolder]['Predicts'] = chisq_driver(drivers[afolder])

#        drivers[afolder]['Predicts'] = jaccard_driver(drivers[afolder])

#        drivers[afolder]['Predicts'] = pearson_driver(drivers[afolder])
        
        drivers[afolder]['Predicts'] = cluster_driver(drivers[afolder])
        
#        print drivers[afolder]['Predicts']
        
        if plotflag == True:
            plt.hist(drivers[afolder]['Predicts'],range=(0.0,1.0))
        
#        plt.show()
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
    if plotflag == True:
        plt.show()

#################################################################################################################

def main(): 
    projpath = '/home/robbie/Hacking/Kaggle/DriverTelemetricAnalysis-AXA/'
    datafolder = 'drivers/'
    calc_metrics(projpath , datafolder)

if __name__ == "__main__": main()
