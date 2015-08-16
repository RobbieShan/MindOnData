import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import itertools
import math
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys


def main(): 
    filepathi = '/home/robbie/Hacking/Kaggle/DriverTelemetricAnalysis-AXA/Predicts.csv'
    filepatho = '/home/robbie/Hacking/Kaggle/DriverTelemetricAnalysis-AXA/predicts3.csv'
    
    fhandlei = open(filepathi, 'r')
    fhandleo = open(filepatho, 'w')
    
    rows = csv.reader(fhandlei,delimiter=',')
    
    for row in rows:
        if  row[0] == 'driver_trip':
            fhandleo.writelines(row[0] + ',' + row[1] + '\n')
        
        else:
            if float(row[1]) > 0.7:
                fhandleo.writelines(row[0] + ',' + '1' + '\n')
            else:
                fhandleo.writelines(row[0] + ',' + '0' + '\n')
    
    fhandlei.close()
    fhandleo.close()

if __name__ == "__main__": main()