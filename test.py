import numpy as np
from pystruct.models import GraphCRF
from pystruct.learners import OneSlackSSVM
from pystruct.inference import get_installed
from util import *
from features import simpleUndirected
# debugging
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
inference_method = get_installed(["qpbo", "ad3", "lp"])[0]

 
# triangle
x_1 = np.array([[0, 1], [1, 0], [.4, .6]])
g_1 = np.array([[0, 1], [1, 2], [0, 2]])
# expected result
y_1 = np.array([1, 0, 1])
x = (x_1, g_1)
y = y_1
crf = GraphCRF(n_states=2, n_features=2, inference_method=inference_method)
model = OneSlackSSVM(model=crf, max_iter=100, C=100, check_constraints=False)
model.fit([x],[y])
print model.w.shape
print model.w
####################################################################################

# inference_method = get_installed(["qpbo", "ad3", "lp"])[0]

# data = get_dF_F1()
# print 'data shape', data.shape
# data = data[:,:50]
# (n, p) = data.shape

# crf = GraphCRF(n_states=2, n_features=n, inference_method=inference_method)
# print 'get crf'

# model = OneSlackSSVM(model=crf, max_iter=100, C=100, check_constraints=False)
# print 'get model'

# features, labels = simpleUndirected(data)
# print 'get features, labels'
# edges = getEdges(data)
# print 'get edges'

# print 'labels shape', len(labels), labels[0].shape
# print 'features shape', len(features), features[0].shape
# print 'edges shape', edges.shape

# Y = labels
# X = zip(features, [edges]*len(features))
# (feature, edge) = X[0]
# print 'tuple feature shape', feature.shape
# print 'tuple edge shape', edge.shape

# labels[0] = np.zeros(n).astype('int64') + 1
# print labels[0]
# print labels[1]

# print 'fitting the model'
# model.fit(X, Y)
# weights = model.w
# print 'weights shape', weights.shape
# print 'size_feature_joint', crf.size_joint_feature
# print weights
