import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import random
 
trial = np.genfromtxt('C:/Users/rshanbh/Desktop/Kaggle/ClickThru/Raw/trial.csv'
                        , delimiter = ','
                        , names=True
                        ,usecols =range(2,23)
                        ,dtype=[('id', 'S20')
                                , ('hour', '<i8')
                                , ('C1', '<i4')
                                , ('banner_pos', '<i2')
                                , ('site_id', 'S8')
                                , ('site_domain', 'S8')
                                , ('site_category', 'S8')
                                , ('app_id', 'S8')
                                , ('app_domain', 'S8')
                                , ('app_category', 'S8')
                                , ('device_id', 'S8')
                                , ('device_ip', 'S8')
                                , ('device_model', 'S8')
                                , ('device_type', 'S8')
                                , ('device_conn_type', 'S8')
                                , ('C14', 'S8')
                                , ('C15', 'S8')
                                , ('C16', 'S8')
                                , ('C17', 'S8')
                                , ('C18', 'S8')
                                , ('C19', 'S8')
                                , ('C20', 'S8')
                                , ('C21', 'S8')]
                                )
Y = [random.randint(0,1) for p in range(0,99)]
 
#trial = np.unique(trial,return_inverse=True)[1].reshape(len(trial),len(trial[0]))
 
trial['site_id'] = np.unique(trial['site_id'],return_inverse=True)[1]
trial['site_domain'] = np.unique(trial['site_domain'],return_inverse=True)[1]
trial['site_category'] = np.unique(trial['site_category'],return_inverse=True)[1]
trial['app_id'] = np.unique(trial['app_id'],return_inverse=True)[1]
trial['app_domain'] = np.unique(trial['app_domain'],return_inverse=True)[1]
trial['app_category'] = np.unique(trial['app_category'],return_inverse=True)[1]
trial['device_id'] = np.unique(trial['device_id'],return_inverse=True)[1]
trial['device_ip'] = np.unique(trial['device_ip'],return_inverse=True)[1]
trial['device_model'] = np.unique(trial['device_model'],return_inverse=True)[1]
trial['device_type'] = np.unique(trial['device_type'],return_inverse=True)[1]
trial['device_conn_type'] = np.unique(trial['device_conn_type'],return_inverse=True)[1]
trial['C14'] = np.unique(trial['C14'],return_inverse=True)[1]
trial['C15'] = np.unique(trial['C15'],return_inverse=True)[1]
trial['C16'] = np.unique(trial['C16'],return_inverse=True)[1]
trial['C17'] = np.unique(trial['C17'],return_inverse=True)[1]
trial['C18'] = np.unique(trial['C18'],return_inverse=True)[1]
trial['C19'] = np.unique(trial['C19'],return_inverse=True)[1]
trial['C20'] = np.unique(trial['C20'],return_inverse=True)[1]
trial['C21'] = np.unique(trial['C21'],return_inverse=True)[1]
 
trial2 = trial.reshape(-2,1)
trial = []
 
newdtype = np.dtype([('id', '<i8')
                    , ('hour', '<i8')
                    , ('C1', '<i8')
                    , ('banner_pos', '<i8')
                    , ('site_id', '<i8')
                    , ('site_domain', '<i8')
                    , ('site_category', '<i8')
                    , ('app_id', '<i8')
                    , ('app_domain', '<i8')
                    , ('app_category', '<i8')
                    , ('device_id', '<i8')
                    , ('device_ip', '<i8')
                    , ('device_model', '<i8')
                    , ('device_type', '<i8')
                    , ('device_conn_type', '<i8')
                    , ('C14', '<i8')
                    , ('C15', '<i8')
                    , ('C16', '<i8')
                    , ('C17', '<i8')
                    , ('C18', '<i8')
                    , ('C19', '<i8')
                    , ('C20', '<i8')
                    , ('C21', '<i8')]
                    )
 
trial2 = trial2.astype(newdtype)
trial2 = trial2.view('<i8')
trial2
 
#for i in range(0,len(trial2)):
# trial2[i] = list(trial2[i])
 
 
enc = OneHotEncoder()
enc.fit(trial2)
X = np.array(enc.transform(trial2).toarray())
 
trial2 = []
 
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)
clf.predict(X) == Y
 
#>>> enc.feature_indices_
#array([ 0, 1, 2, 1015, 1023, 1045, 1064, 1071, 1093, 1102, 1107,
# 1127, 1224, 1297, 1300, 1303, 1360, 1362, 1365, 1408, 1412, 1431,
# 1457, 1474])
#
#enc = OneHotEncoder()
#X = [[0, 0, 10], [1, 1, 0], [0, 2, 1], [1, 0, 2],[1, 0, 3]]
#enc.fit(X)
#enc.n_values_
#enc.feature_indices_
#
#enc.transform([[0, 1, 3]]).toarray()