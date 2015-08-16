import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import itertools
import math
from sklearn.cluster import KMeans
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
        drivers[a_folder][afile]['DStats']['KurtDistDel'] = sp.stats.kurtosis(drivers[a_folder][afile]['DistDel'])   
        drivers[a_folder][afile]['DStats']['SkewDistDel'] = sp.stats.skew(drivers[a_folder][afile]['DistDel'])   
        drivers[a_folder][afile]['DStats']['KurtACosDel'] = sp.stats.kurtosis(drivers[a_folder][afile]['ACosDel'])   
        drivers[a_folder][afile]['DStats']['SkewACosDel'] = sp.stats.skew(drivers[a_folder][afile]['ACosDel'])   
        
        drivers[a_folder]['DStats'] = np.concatenate((drivers[a_folder]['DStats']
                                        ,[drivers[a_folder][afile]['DStats']['AvgDistDel']]
                                        ,[drivers[a_folder][afile]['DStats']['AvgACosDel']] 
                                        ,[drivers[a_folder][afile]['DStats']['SDevDistDel']] 
                                        ,[drivers[a_folder][afile]['DStats']['SDevACosDel']] 
                                        ,[drivers[a_folder][afile]['DStats']['TotalDist']]
                                        ,[drivers[a_folder][afile]['DStats']['TotalTime']]
                                        ,[drivers[a_folder][afile]['DStats']['KurtDistDel']]
                                        ,[drivers[a_folder][afile]['DStats']['SkewDistDel']]
                                        ,[drivers[a_folder][afile]['DStats']['KurtACosDel']]
                                        ,[drivers[a_folder][afile]['DStats']['SkewACosDel']]
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
    
    fig = scatterplot_matrix(np.transpose(drivers[a_folder]['DStats'])
                                            , ['AvgDistDel', 'AvgACosDel', 'SDevDistDel', 'SDevACosDel','TotalDist','TotalTime','KurtDistDel','SkewDistDel','KurtACosDel','SkewACosDel']
                                            ,linestyle='none', marker='o', color='black', mfc='none')
    fig.suptitle('Simple Scatterplot Matrix')
    plt.show()
    
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
#    print "#############################DStats Above#################################ValueError: zero-size array to reduction operation minimum which has no identity#################"

    X = StandardScaler().fit_transform(a_driver['DStats'])
    
#    print X
#    print "DStats are.....::" , a_driver['DStats']
#    print "X is...........::" , X
#    print "############################Scaled X Above###################################################"
    
    db = KMeans(n_clusters=10).fit(X)
#    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
#    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

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
    datafolder = 'drivers2/'
    calc_metrics(projpath , datafolder)

if __name__ == "__main__": main()
