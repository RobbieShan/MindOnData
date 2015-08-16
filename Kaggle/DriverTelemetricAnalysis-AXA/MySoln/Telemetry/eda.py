import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import math

#################################################################################################################

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
#    print "v1: ", v1, " v2: ", v2
    if v1 == [0.0,0.0] or  v2 == [0.0,0.0] or v1==v2:
        return 0.0
    else:
        return math.acos(round(dotproduct(v1, v2),2) / round(length(v1) * length(v2),2))
  
#################################################################################################################

def get_all_data(a_projpath , a_folder):
    
    drivers = {}
    allfiles = os.listdir(a_projpath + a_folder)
    drivers[a_folder] = {}
    drivers[a_folder]['DStats'] = {} # Folder Level Descriptive Statistics
    
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
        
        for arow in rows:
            if arow[0] != 'x':
                curpoint = [float(arow[0]),float(arow[1])]
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
        
        plt.subplot(1,3,1)
        plt.plot( [x[0] for x in drivers[a_folder][afile]['Data']], [x[1] for x in drivers[a_folder][afile]['Data']])  
        plt.grid(True)
        plt.title("Driver: Data " + a_folder) 
      
        plt.subplot(1,3,2)
#        plt.plot(drivers[a_folder][afile]['DStats']['AvgDistDel'],np.random.random()/2,'b^')  
        plt.plot(drivers[a_folder][afile]['DStats']['AvgDistDel'],drivers[a_folder][afile]['DStats']['AvgACosDel'],'b^')  
        plt.grid(True)
        plt.title("Driver: AvgDistDel " + a_folder)
                
        plt.subplot(1,3,3)
        plt.plot(drivers[a_folder][afile]['DStats']['AvgACosDel'],np.random.random()/2,'b^')
        plt.grid(True)
        plt.title("Driver: AvgACosDel " + a_folder)
            
#        drivers[a_folder][afile]['DStats']['mean'] = sp.mean([x[0] for x in drivers[a_folder][afile]['Data']])
#        drivers[a_folder][afile]['DStats']['triplen'] = len([x[0] for x in drivers[a_folder][afile]['Data']])
    
    filelist = [x for x in drivers[a_folder].keys()]
    
 #   drivers[a_folder]['DStats']['triplen'] = sp.mean(x for x in drivers[a_folder][afile]['DStats']['triplen'])
    
    return drivers[a_folder]

#################################################################################################################

def calc_metrics(a_projpath):
    allfolders = os.listdir(a_projpath)
    drivers = {}
#    driverstats = {}
    
    for afolder in allfolders:
#        plt.subplot(1,len(allfolders),allfolders.index(afolder)+1)
        drivers[afolder] = get_all_data(a_projpath , afolder)
   
#        print "Driver: " + afolder + " ", drivers[afolder]    
        plt.show()        

#################################################################################################################
   
def main(): 
    projpath = 'G:/Continuing Education/Research & Presentations/Self - Machine Learning/Kaggle/DriverTelemetricAnalysis-AXA/'
    datafolder = 'tempdrivers/'
    calc_metrics(projpath + datafolder)

if __name__ == "__main__": main()
