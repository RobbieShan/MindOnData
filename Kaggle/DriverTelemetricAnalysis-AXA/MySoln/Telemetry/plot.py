# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 18:07:44 2014

@author: robbie
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import math


projpath = '/home/robbie/Hacking/Kaggle/DriverTelemetricAnalysis-AXA/'
datafolder = 'drivers3/'
afolder = '2460'

allfiles = os.listdir(projpath + datafolder + afolder + '/')

drivers = {}
drivers[afolder] = {}

for afile in allfiles:
    fhandle = open(projpath + datafolder + afolder + '/' + afile, 'r')
    drivers[afolder][afile] = {}
    drivers[afolder][afile]['Data'] = []
    rows = csv.reader(fhandle,delimiter=',')
    lastpoint = [0.0,1.0]
    for arow in rows:
        if arow[0] != 'x':
            curpoint = [float(arow[0]),float(arow[1])]
            drivers[afolder][afile]['Data'].append(curpoint)
    fhandle.close()
#drivers = {}

#drivers[a_folder] = {}
#drivers[a_folder]['DStats'] = {} # Folder Level Descriptive Statistics

for afile in allfiles:
#    plt.subplot(1,1,1)
    plt.figure(1)
    plt.plot( [x[0] for x in drivers[afolder][afile]['Data']], [x[1] for x in drivers[afolder][afile]['Data']], label=afile)
    plt.grid(True)
    plt.legend()
    plt.title("Driver: Data " + afolder)

plt.show()



#Total Distance Travelled
#Ending Distance from Origin i.e. Distance of the Final Point from the Origin
#Total Distance / Total Time (Should I include Zeros for the Time? -- I am thinking Yes) -- This is Distance per Second
#Total Deviation .. I am thiking of ignoring the first deviation because I am anchoring it against the +ve X Axis.. Need to check with a few points if I am doing this right
#Total Deviation / Total Time (Again, thinking of including Zeros) --- This is Deviation per Second

#Cluster a few folders and look at # of trips that fall into clusters -- Pick that as the # of clusters to be found