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
        drivers[a_folder][afile]['ATanDel'] = [] 
        drivers[a_folder][afile]['DStats'] = {}
        
        
        rows = csv.reader(fhandle,delimiter=',')
        
        secondlastpoint = [0.0,0.0]
        lastpoint = [1.0,0.0]

        i = 0
        
        for arow in rows:
            
            if arow[0] == 'x':
                i = i+1                
                continue
            
#            if arow[0] == arow[1] == 0:
#               i = i+1
#               continue
            else:           
#                if i == 1 or i == 2:
                curpoint = [(round(float(arow[0]),0)),(round(float(arow[1]),0))]
#                i=i+1
    #                print "Index::" , i, "LastPoint::", lastpoint, "::CurPoint:::" , curpoint
    #                v2 = [lastpoint[0]-secondlastpoint[0],lastpoint[1]-secondlastpoint[1]]
    #                lastpoint = [0.0,0.0]    
    #                v1 = [curpoint[0]-lastpoint[0],curpoint[1]-lastpoint[1]]              
                if i%10 == 0:
#                    curpoint = [(round(float(arow[0]),0)),(round(float(arow[1]),0))]
    #                print "Index::" , i, "LastPoint::", lastpoint, "::CurPoint:::" , curpoint
                    v2 = [lastpoint[0]-secondlastpoint[0],lastpoint[1]-secondlastpoint[1]]
                    v1 = [curpoint[0]-lastpoint[0],curpoint[1]-lastpoint[1]]
                                    
                    drivers[a_folder][afile]['ACosDel'].append(angle(v1,v2))
                    drivers[a_folder][afile]['ATanDel'].append(math.atan2(curpoint[1],curpoint[0])-math.atan2(lastpoint[1],lastpoint[0]))
                    drivers[a_folder][afile]['DistDel'].append(math.hypot((curpoint[0]-lastpoint[0]),(curpoint[1]-lastpoint[1])))
                
#                    print "Index::" , i, "SecondLastPoint::", secondlastpoint, "LastPoint::", lastpoint , "::CurPoint:::" , curpoint , "::ACosDel:::" , math.degrees(angle(v1,v2)) , "::DistDel:::" , math.hypot((curpoint[0]-lastpoint[0]),(curpoint[1]-lastpoint[1]))
                    drivers[a_folder][afile]['Data'].append(curpoint)
                secondlastpoint = lastpoint
                lastpoint = curpoint
                i = i+1    
        
#        print drivers[a_folder][afile]['ACosDel']          
#        print drivers[a_folder][afile]['DistDel']
        
        if plotflag == True:
#            plt.subplot(1,1,1)

            temp = drivers[a_folder][afile]['Data']
#            print temp
            plt.plot( [x[0] for x in temp], [x[1] for x in temp])  

#            print type(temp)
#            print len(temp)
#            plt.plot( temp[len(temp)-1][0],temp[len(temp)-1][1], 'b^' )  
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
            plt.show()
            
            
        drivers[a_folder][afile]['ACosDel'] = np.array(drivers[a_folder][afile]['ACosDel'])
        drivers[a_folder][afile]['ATanDel'] = np.array(drivers[a_folder][afile]['ATanDel'])
        drivers[a_folder][afile]['DistDel'] = np.array(drivers[a_folder][afile]['DistDel'])       

        masACosDel = np.ma.masked_equal(drivers[a_folder][afile]['ACosDel'],0)        
        masATanDel = np.ma.masked_equal(drivers[a_folder][afile]['ATanDel'],0)        
        masDistDel = np.ma.masked_equal(drivers[a_folder][afile]['DistDel'],0)                
        
        
        drivers[a_folder][afile]['DStats']['TotalDist'] = masDistDel.sum() if masDistDel.count() != 0 else 0  
        drivers[a_folder][afile]['DStats']['TotalTime'] = len(drivers[a_folder][afile]['Data'])

        drivers[a_folder][afile]['DStats']['AvgSpd'] = np.array(drivers[a_folder][afile]['DStats']['TotalDist'] / drivers[a_folder][afile]['DStats']['TotalTime'])
        drivers[a_folder][afile]['DStats']['NoOfTurns'] = np.array(masACosDel.count())
#        drivers[a_folder][afile]['DStats']['TotAngOfTurns'] = np.array(masACosDel.sum())
        drivers[a_folder][afile]['DStats']['TripEndDist'] = np.array(math.hypot((curpoint[0]-0),(curpoint[1]-0)))
        
        drivers[a_folder][afile]['DStats']['AvgDistDel'] = masDistDel.mean() if masDistDel.count() != 0 else 0
        drivers[a_folder][afile]['DStats']['AvgACosDel'] = masACosDel.mean() if masACosDel.count() != 0 else 0
        drivers[a_folder][afile]['DStats']['AvgATanDel'] = masATanDel.mean() if masATanDel.count() != 0 else 0

        drivers[a_folder][afile]['DStats']['SDevDistDel'] = masDistDel.std() if masDistDel.count() != 0 else 0      
#        drivers[a_folder][afile]['DStats']['SDevACosDel'] = masACosDel.std() if masACosDel.count() != 0 else 0         


#        drivers[a_folder][afile]['DStats']['KurtDistDel'] = sp.stats.kurtosis(drivers[a_folder][afile]['DistDel'][masDistDel!=True])   
#        drivers[a_folder][afile]['DStats']['KurtACosDel'] = sp.stats.kurtosis(drivers[a_folder][afile]['ACosDel'][masACosDel!=True])   

#        drivers[a_folder][afile]['DStats']['SkewDistDel'] = sp.stats.skew(drivers[a_folder][afile]['DistDel'][masDistDel!=True])   
#        drivers[a_folder][afile]['DStats']['SkewACosDel'] = sp.stats.skew(drivers[a_folder][afile]['ACosDel'][masACosDel!=True])   
        
#        print drivers[a_folder][afile]['DStats']     
        drivers[a_folder]['DStats'] = np.concatenate((drivers[a_folder]['DStats']
#                                        ,[afile.split('.')[0]]
                                        ,[drivers[a_folder][afile]['DStats']['AvgSpd']]
                                        ,[drivers[a_folder][afile]['DStats']['NoOfTurns']]
#                                        ,[drivers[a_folder][afile]['DStats']['TotAngOfTurns']]
                                        ,[drivers[a_folder][afile]['DStats']['TripEndDist']]
                                        ,[drivers[a_folder][afile]['DStats']['TotalDist']]
#                                        ,[drivers[a_folder][afile]['DStats']['TotalTime']]
                                        ,[drivers[a_folder][afile]['DStats']['AvgDistDel']]
                                        ,[drivers[a_folder][afile]['DStats']['AvgACosDel']] 
                                        ,[drivers[a_folder][afile]['DStats']['SDevDistDel']] 
#                                        ,[drivers[a_folder][afile]['DStats']['SDevACosDel']] 
#                                        ,[drivers[a_folder][afile]['DStats']['KurtDistDel']]
#                                        ,[drivers[a_folder][afile]['DStats']['KurtACosDel']]
#                                        ,[drivers[a_folder][afile]['DStats']['SkewDistDel']]
#                                        ,[drivers[a_folder][afile]['DStats']['SkewACosDel']]
                                        ))
#        
        del drivers[a_folder][afile]['Data']
        del drivers[a_folder][afile]['ACosDel']
        del drivers[a_folder][afile]['DistDel']
#        
        del drivers[a_folder][afile]['DStats']['AvgSpd']
        del drivers[a_folder][afile]['DStats']['NoOfTurns']
#        del drivers[a_folder][afile]['DStats']['TotAngOfTurns']
        del drivers[a_folder][afile]['DStats']['TripEndDist']      
        del drivers[a_folder][afile]['DStats']['TotalDist']
#        del drivers[a_folder][afile]['DStats']['TotalTime']
        del drivers[a_folder][afile]['DStats']['AvgDistDel']
        del drivers[a_folder][afile]['DStats']['AvgACosDel']
        del drivers[a_folder][afile]['DStats']['SDevDistDel']
#        del drivers[a_folder][afile]['DStats']['SDevACosDel']
#        del drivers[a_folder][afile]['DStats']['KurtDistDel']
#        del drivers[a_folder][afile]['DStats']['KurtACosDel'] 
#        del drivers[a_folder][afile]['DStats']['SkewDistDel']
#        del drivers[a_folder][afile]['DStats']['SkewACosDel']
                                                           
                
#        drivers[a_folder][afile]['DStats'] = {}
        fhandle.close()
#        print drivers[a_folder]
#    print "#############################DStats Before Scaling Below ##################################################"
#    print drivers[a_folder]['DStats']                    
    drivers[a_folder]['DStats'] = drivers[a_folder]['DStats'].reshape(len(allfiles),7)
    
#    drivers[a_folder]['Baseline'] = np.mean(drivers[a_folder]['DStats'],axis=0)
    
    drivers[a_folder]['Baseline'] = sp.stats.mode(drivers[a_folder]['DStats'])[0][0]
#    print drivers[a_folder]['Baseline']
    
#    sys.stdout = open('a_projpath' +'output.txt','w') 
#    print drivers[a_folder]

    
    if plotflag == True:
        
        fig = scatterplot_matrix(np.transpose(drivers[a_folder]['DStats'])
                                                , [
                                                'AvgSpd'
                                                ,'NoOfTurns'
#                                                ,'TotAngOfTurns'
                                                ,'TripEndDist'
#                                                ,'TotalDist'
#                                                ,'TotalTime'
#                                                ,'AvgDistDel'
#                                                ,'AvgACosDel'
                                                ,'SDevDistDel'
                                                ,'SDevACosDel'
#                                                ,'KurtDistDel'
#                                                ,'KurtACosDel'
#                                                ,'SkewDistDel'
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
    
    pca = PCA(n_components=3)
    Xpca = pca.fit(X).transform(X)
    
    if plotflag == True:
        
        fig = scatterplot_matrix(np.transpose(Xpca)
                                                , ['PC1'
                                                , 'PC2'
                                                , 'PC3'
#                                                , 'PC4'
#                                                ,'PC5'
                                                ]
                                                ,linestyle='none', marker='o', color='black', mfc='none')
        fig.suptitle('Simple Scatterplot Matrix')
        plt.show()
        

    db = KMeans(n_clusters=3,n_jobs = -1).fit(Xpca)
    minDist = db.transform(Xpca).min(axis=1) # Get  distance for each point from the nearest cluster

#    zMinDist = sp.stats.mstats.zscore(minDist) # Convert to z-score i.e. Normalize the distances
    zMinDist = sp.stats.mstats.zscore(minDist) # Convert to z-score i.e. Normalize the distances
    
    
#    plt.figure()
#    plt.hist(zMinDist,range=(-3,3))

#    print zMinDist.round(2)
#    print "###############################################################################"
#    print sp.stats.kstest(zMinDist,"expon" ,(0,.715))   
#    print sp.stats.kstest(zMinDist,"expon" ,(0,.72))
    
#    tef = sp.stats.norm.fit(zMinDist)
#    tef = sp.stats.expon.fit(zMinDist)
    
#    plt.hist(sp.stats.expon.rvs(0,tef[1],size=200))
#    print tef
#    print sp.stats.kstest(zMinDist,"expon" ,(0,.72))
#    print sp.stats.kstest(zMinDist,"expon" ,(0,.75))
#    print sp.stats.kstest(zMinDist,"expon" ,(0,76))

#    print "###############################################################################"

#    probZMinDist = sp.stats.expon.pdf(zMinDist.round(2),loc=tef[0],scale=tef[1]) # Find the probability distribution this belongs to and get the probability
#    probZMinDist = sp.stats.expon.pdf(zMinDist,scale=1/tef[1]) # Find the probability distribution this belongs to and get the probability
    probZMinDist = 2*sp.stats.norm.pdf(zMinDist) # Find the probability distribution this belongs to and get the probability
    
#    plt.figure()
#    plt.hist(probZMinDist)   
#    print probZMinDist.round(2)
    
    
#    XpcaMean = Xpca.mean(axis=0)
#    XpcaDistFromMean = np.array([],float)
    
#    print "Xpca::", Xpca    
#    print "XpcaMean::", XpcaMean
#    print "XpcaDistFromMean::", XpcaDistFromMean
    
#    for i in range(0,len(Xpca)):
#        XpcaDistFromMean = np.append(XpcaDistFromMean, np.linalg.norm(Xpca[i]-XpcaMean))
    
#    temp = (1-sp.stats.norm.cdf(sp.stats.mstats.zscore(XpcaDistFromMean)))
#    temp = temp.round(2)
#    
#    print temp.round(2)
#    plt.figure()
#    plt.hist(temp,range=(0,1))
#    plt.plot(temp,'b^')
#    plt.show()
    

    
#    db = DBSCAN(eps=0.7).fit(Xpca)
    
#    db = AgglomerativeClustering(n_clusters=5).fit(Xpca)
        
#    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#    core_samples_mask[db.core_sample_indices_] = True
    
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
#    print Counter(labels)
#    print labels
    
#    print "###############################################################################"
    print('Estimated number of clusters: %d' % n_clusters_)
#    print 'Count of Predicts::', len(X)
#    print("Silhouette Coefficient: %0.3f"    % metrics.silhouette_score(Xpca, labels))
    print "% Variance Explaned: %0.3f" , sum(pca.explained_variance_ratio_)
#    print "##############################DBSCAN  X Below#################################################"
    
    return probZMinDist ##KMeans ZScores
    
#    print X    G:/Continuing Education/Research & Presentations/Self - Machine Learning/Kaggle/DriverTelemetricAnalysis-AXA/'
#    try:
    
#    return (1- (db.transform(Xpca)/max(db.transform(Xpca))))
#    print (metrics.silhouette_samples(Xpca, labels)+1)/2
    
#    t = []
#    for i in labels:
#        if i >=0: t.append(1)
#        else: t.append(0)
#    
#    return t
#    return (metrics.silhouette_samples(Xpca, labels,metric='canberra')+1)/2
#    return (metrics.silhouette_samples(Xpca, labels)+1)/2
#    
#    temp2 = np.array([],float)
#    for i in temp:
#        if i < 0.1: 
#            temp2=np.append(temp2,0)
#        else: 
#            temp2=np.append(temp2,1)
#    
##    print temp2
#    return (temp2)
    
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
    
    f2 = open(a_projpath + 'PredictsSimple.csv', 'w')
    
    f.writelines('driver_trip,prob\n')
    f2.writelines('driver_trip,prob\n')
        
    cnt = 0
    
    
    for afolder in allfolders:
#        plt.subplot(1,len(allfolders),allfolders.index(afolder)+1)
        print "###############################################################################"
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
            plt.figure()
            plt.hist(drivers[afolder]['Predicts'],range=(0.0,1.0))
            plt.show()
        
#        plt.show()
        allfiles = os.listdir(a_projpath + a_datafolder + afolder)    
        i = 0
        for afile in allfiles:
#            f.writelines(str(afile) + ',' + str(drivers[afolder]['Predicts'][i]) + '\n')
            temp = round(drivers[afolder]['Predicts'][i],2)
            f.writelines(str(afolder) + '_' + str(afile).split('.')[0] + ',' + str(temp) + '\n')
            
            if temp  <= 0.1:
                f2.writelines(str(afolder) + '_' + str(afile).split('.')[0] + ',' + str(0.0) + '\n')
            else:
                f2.writelines(str(afolder) + '_' + str(afile).split('.')[0] + ',' + str(1.0) + '\n')
#            print "I is ::::::::::" , i, "A File is :::::::::" , afile
#            print str(afile) + ',' + str(drivers[afolder]['Predicts'][i])
            i = i+1        
    f.close()    
    f2.close()
#        write_predicts(drivers[[afolder]['Predicts'],drivers[afolder]])
#    if plotflag == True:
#        plt.show()

#################################################################################################################

def main(): 
    projpath = '/home/robbie/Hacking/Kaggle/DriverTelemetricAnalysis-AXA/'
    datafolder = 'drivers/'
    calc_metrics(projpath , datafolder)

if __name__ == "__main__": main()
