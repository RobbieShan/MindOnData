import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
 
# plt.plot(a[0],a[1],'b^')
# plt.plot(b[0],b[1],'b^')
# plt.show()
 
##############################################################################
a = [np.random.normal(0,1,100),np.random.normal(0,1,100),np.random.normal(0,1,100)]
b = [np.random.normal(4,1,100),np.random.normal(4,1,100),np.random.normal(4,1,100)]
 
aa = np.array([[x,y,z] for (x,y,z) in zip(a[0],a[1],a[2])],float)
bb = np.array([[x,y,z] for (x,y,z) in zip(b[0],b[1],b[2])],float)
 
X = np.concatenate((aa,bb))
X = StandardScaler().fit_transform(X)
 
# Compute DBSCAN
 
db = DBSCAN(eps=0.9, min_samples=50).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
 
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
 
print('Estimated number of clusters: %d' % n_clusters_)
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#print("Adjusted Rand Index: %0.3f"
# % metrics.adjusted_rand_score(labels_true, labels))
#print("Adjusted Mutual Information: %0.3f"
# % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
% metrics.silhouette_score(X, labels))
print '######################################################'
 
#for i in range(0,len(X)):
print "Individual Sihouette Scores: ", metrics.silhouette_samples(X, labels)
 
##############################################################################
 
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()