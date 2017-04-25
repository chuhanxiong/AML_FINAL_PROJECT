import numpy as np
from pystruct.models import GraphCRF
from pystruct.learners import OneSlackSSVM
from pystruct.inference import get_installed
from util import *
from features import simpleUndirected
# debugging
# from sklearn.datasets import load_iris
# from sklearn.cross_validation import train_test_split
# inference_method = get_installed(["qpbo", "ad3", "lp"])[0]

# iris = load_iris()
# X, y = iris.data, iris.target
 
# X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
# Y = y.reshape(-1, 1)
# print len(X_)
# print Y.shape
# print X_[0]
# crf = GraphCRF(n_states=3, n_features=4, inference_method=inference_method)
# model = OneSlackSSVM(model=crf, max_iter=100, C=100, check_constraints=False)
# model.fit(X_, Y)
####################################################################################

inference_method = get_installed(["qpbo", "ad3", "lp"])[0]

data = get_dF_F1()
print 'data shape', data.shape
data = data[:,:50]
(n, p) = data.shape

crf = GraphCRF(n_states=2, n_features=n, inference_method=inference_method)
print 'get crf'

model = OneSlackSSVM(model=crf, max_iter=100, C=100, check_constraints=False)
print 'get model'

features, labels = simpleUndirected(data)
print 'get features, labels'
edges = getEdges(data)
print 'get edges'

print 'labels shape', len(labels), labels[0].shape
print 'features shape', len(features), features[0].shape
print 'edges shape', edges.shape

Y = labels
X = zip(features, [edges]*len(features))

labels[0] = np.zeros(n).astype('int64') + 1
print labels[0]
print labels[1]

print 'fitting the model'
model.fit(X, Y)