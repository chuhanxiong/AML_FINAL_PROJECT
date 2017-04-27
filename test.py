import time
import numpy as np
from pystruct.models import GraphCRF
from pystruct.learners import OneSlackSSVM
from pystruct.inference import get_installed
from util import *
from features import simpleUndirected
DEBUG = False
START_TIME = time.time()
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
print 'Original data shape', data.shape
data = data[:, :50]
data = binarize(data)
(n, p) = data.shape
print("We will use data of shape: {}".format((n, p)))

crf = GraphCRF(n_states=2, n_features=n, inference_method=inference_method)
print 'get crf'

if DEBUG:
    print 'DEBUG'
    model = OneSlackSSVM(model=crf, max_iter=100, C=100, verbose=1, check_constraints=False)
else:    
    model = OneSlackSSVM(model=crf, max_iter=10, C=100, check_constraints=False)
print 'get model'

features, labels = simpleUndirected(data)
print 'get features, labels'
edges = getEdges(data)
print 'get edges'

print 'len(labels): {0}, labels[0].shape: {1}'.format(len(labels), labels[0].shape)
print 'len(features): {0}, features[0].shape: {1}'.format(len(features), features[0].shape)
print 'edges shape: ', edges.shape
print("First 10 edges: {}".format(edges[:10]))

Y = labels
X = zip(features, [edges]*len(features))

print 'fitting the model, runtime so far = {0:.2f}'.format(time.time() - START_TIME)
model.fit(X, Y)
print("Total run time: {0:.2f} seconds".format(time.time() - START_TIME))
print 'weights'
print model.w