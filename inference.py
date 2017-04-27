import numpy as np
from pystruct.models import EdgeFeatureGraphCRF
from pystruct.learners import OneSlackSSVM
from pystruct.inference import get_installed
from util import *
from features import simpleUndirected

def getEdgeFeatures(features, edges_shape):    
    edge_features = [None] * len(features)
    e_f_idex = 0
    for f in features:
        efs = np.zeros(shape=edges_shape, dtype='int8')
        idx = 0
        for i in range(0, n):
            for j in range(i+1, n):
                efs[idx][0] = f[i, j]
                efs[idx][1] = f[i, j]
                idx += 1
            efs[idx][0] = f[i, i]
            efs[idx][1] = f[i, i]
            idx += 1
        edge_features[e_f_idex] = efs
        e_f_idex += 1
    return edge_features

inference_method = get_installed(["qpbo", "ad3", "lp"])[0]

data = get_dF_F1()
print 'Original data shape', data.shape
test_data = data[:,50:200]
test_data = binarize(test_data)

train_data = data[:, :50]
train_data = binarize(train_data)
(n, p) = train_data.shape
print("We will use train_data of shape: {}".format((n, p)))

crf = EdgeFeatureGraphCRF(n_states=2, n_edge_features=2, inference_method=inference_method)
print 'get crf'

model = OneSlackSSVM(model=crf, max_iter=10, C=1, verbose=1, check_constraints=False)
print 'get model'

features, labels = simpleUndirected(train_data)
print 'get features, labels'
edges = getEdges(train_data)
print 'get edges'
edge_features = getEdgeFeatures(features, edges.shape)
print 'get edge_features'

train_Y = labels
train_X = list(zip(features, [edges]*len(features), edge_features))

# print 'fitting the model'
# model.fit(train_X, train_Y)
# w = model.w

print 'getting weights'
with open('weights.txt') as f:
    content = f.readlines()
w = []
for line in content:
    tokens = line.strip().split()
    w += tokens
w = np.asarray(w, dtype='float32')

print 'initializing crf'
crf.initialize(train_X, train_Y)

print 'building sessions'
A = range(67)
B = range(67, 120)
C = range(120, n)

sessionA = test_data[:,:50]
sessionA[B+C,:] = 0
print 'sessionA shape', sessionA.shape
print 'non-zero entries', np.sum(sessionA)
A_features, A_Y = simpleUndirected(sessionA)
A_edge_features = getEdgeFeatures(A_features, edges.shape)
A_X = list(zip(A_features, [edges]*len(A_features), A_edge_features))

sessionB = test_data[:,50:100]
sessionB[A+C,:] = 0
print 'sessionB shape', sessionB.shape
print 'non-zero entries', np.sum(sessionB)
B_features, B_Y = simpleUndirected(sessionB)
B_edge_features = getEdgeFeatures(B_features, edges.shape)
B_X = list(zip(B_features, [edges]*len(B_features), B_edge_features))

sessionC = test_data[:,100:]
sessionC[A+B,:] = 0
print 'sessionC shape', sessionC.shape
print 'non-zero entries', np.sum(sessionC)
C_features, C_Y = simpleUndirected(sessionC)
C_edge_features = getEdgeFeatures(C_features, edges.shape)
C_X = list(zip(C_features, [edges]*len(C_features), C_edge_features))


print 'doing the inference'

print 'test sessionA'
A_correct_rate_list = []
for x, y in zip(A_X, A_Y):
    y_hat = crf.inference(x, w)    
    correct_rate = np.zeros(shape=(y_hat.shape))
    correct_rate[y_hat==y] = 1    
    A_correct_rate_list.append(np.sum(correct_rate)/(1.0*len(correct_rate)))
    print 'correct_rate', A_correct_rate_list[-1]
print 'A_correct_rate_mean:', np.mean(A_correct_rate_list)

print 'test sessionB'
B_correct_rate_list = []
for x, y in zip(B_X, B_Y):
    y_hat = crf.inference(x, w)    
    correct_rate = np.zeros(shape=(y_hat.shape))
    correct_rate[y_hat==y] = 1    
    B_correct_rate_list.append(np.sum(correct_rate)/(1.0*len(correct_rate)))
    print 'correct_rate', B_correct_rate_list[-1]
print 'B_correct_rate_mean:', np.mean(B_correct_rate_list)

print 'test sessionC'
C_correct_rate_list = []
for x, y in zip(C_X, C_Y):
    y_hat = crf.inference(x, w)
    correct_rate = np.zeros(shape=(y_hat.shape))
    correct_rate[y_hat==y] = 1    
    C_correct_rate_list.append(np.sum(correct_rate)/(1.0*len(correct_rate)))
    print 'correct_rate', C_correct_rate_list[-1]
print 'C_correct_rate_mean:', np.mean(C_correct_rate_list) 