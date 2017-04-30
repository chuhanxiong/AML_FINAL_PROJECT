import numpy as np
from numpy.testing import assert_array_equal
from pystruct.models import EdgeFeatureGraphCRF
from pystruct.learners import OneSlackSSVM
from pystruct.inference import get_installed
from util import *
from features import simpleUndirected
import matplotlib.pyplot as plt
import random
import pandas as pd

inference_method = get_installed(["qpbo", "ad3", "lp"])[0]

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

def test_connectivites_for_neuron(neuron_idx, edges, labels, session, w):
    print 'test_connectivites_for_neuron', neuron_idx
    brothers = find_neuron_connectivities(neuron_idx,labels)
    if brothers == []:
        print 'no connectivities found for neuron', neuron_idx
        return -1
    else:
        print 'brothers for', neuron_idx, 'are', brothers        
        features, labels = simpleUndirected(session)
        edge_features = getEdgeFeatures(features, edges.shape)
        X = list(zip(features, [edges]*len(features), edge_features))
      
        (p, n, n) = features.shape
        for t in range(p):
            temp = features[t,:,:]
            temp[brothers,:] = temp[neuron_idx,:]        
            
        modified_edge_features = getEdgeFeatures(features, edges.shape)
        X_prime = list(zip(features, [edges]*len(features), modified_edge_features))
        
        correct_rate_list = []
        for t in range(p):
            x = X[t]
            y = crf.inference(x, w)
            x_prime = X_prime[t]
            y_hat = crf.inference(x_prime, w)
            correct_rate = np.zeros(shape=(y_hat.shape))
            correct_rate[y_hat==y] = 1
            correct_rate_list.append(np.sum(correct_rate)/(1.0*len(correct_rate)))
            print 'correct_rate at time', t, 'is', correct_rate_list[-1]
        avg = np.mean(correct_rate_list)
        print 'neuron', neuron_idx, 'correct_rate_mean:', avg
        return avg

######################################### get train_data, and test_data from raw Data #########################################
time_frame = 50000
print 'time_frame is', time_frame
data = get_dF_F1()
print 'Original data shape', data.shape

test_start_idx = random.randint(time_frame, data.shape[1])
test_data = data[:,test_start_idx:test_start_idx+time_frame]
test_data = binarize(test_data)

train_data = data[:, :time_frame]
train_data = binarize(train_data)


######################################### get train_data, and test_data from simulatedData #########################################
# X, shuf_A_true, unshuffle = simulatedData(n=18, T=1000)
# A_true = shuf_A_true[unshuffle][:, unshuffle]
# A_true[A_true != 0] = 1
# X_data = binarize(X[unshuffle])
# train_size = 200
# test_size = 200
# X_train = sessionize(X_data[:, :train_size], num_sessions=2)[0]
# X_train = X_data
# X_test = X_data[:, -test_size:]

# train_data = X_train
# test_data = X_test

(n, p) = train_data.shape
print("We will use train_data of shape: {}".format((n, p)))

crf = EdgeFeatureGraphCRF(n_states=2, n_edge_features=2, inference_method=inference_method)
print 'get crf'

model = OneSlackSSVM(model=crf, max_iter=10, C=1, verbose=1, check_constraints=False)
print 'get model'

features, labels = simpleUndirected(train_data)
print 'get features, labels'

# get adjacencyMatrix
adjacencyMatrix = getAdjacencyMatrix(labels)
# df = pd.DataFrame(adjacencyMatrix)
# df.to_csv('adjacencyMatrix.csv', index=False, header=False)

# list1 contains points from the adjacency matrix
list1x = []
list1y = []
s1 = set()
print adjacencyMatrix.shape
for i in range(adjacencyMatrix.shape[0]):
    for j in range(adjacencyMatrix.shape[1]):
        if adjacencyMatrix[i, j] == 1:
            list1x.append(i)
            list1y.append(j)
            s1.add((i, j))

plt.scatter(list1x, list1y, c='b')
plt.show()

# list2 contains points from A_true matrix, i.e., the adjacency matrix from the simulatedData
list2x = []
list2y = []
s2 = set()
print A_true.shape
for i in range(A_true.shape[0]):
    for j in range(A_true.shape[1]):
        if A_true[i, j] == 1 and i != j:
            list2x.append(i)
            list2y.append(j)
            s2.add((i, j))

plt.scatter(list2x, list2y, c='b')
plt.show()

# list3 contains the intersection of the adjacency matrix and the A_ture matrix
s3 = s1.intersection(s2)
list3x = []
list3y = []
for x,y in s3:
    list3x.append(x)
    list3y.append(y)


plt.scatter(list3x, list3y, c='g')
plt.show()

print 'overlap rate for s2', 1.0*len(s3)/len(s2) # recall
print 'overlap rate for s1', 1.0*len(s3)/len(s1) # precision

edges = getEdges(train_data)
print 'get edges'
edge_features = getEdgeFeatures(features, edges.shape)
print 'get edge_features'

train_Y = labels
train_X = list(zip(features, [edges]*len(features), edge_features))

print 'fitting the model'
model.fit(train_X, train_Y)
w = model.w

# print 'getting weights'
# with open('weights.txt') as f:
#     content = f.readlines()
# w = []
# for line in content:
#     tokens = line.strip().split()
#     w += tokens
# w = np.asarray(w, dtype='float32')

print 'initializing crf'
crf.initialize(train_X, train_Y)

print 'test neurons connectivities'
neurons = [18]
valid_neurons = []
avgs = []
for neuron_idx in neurons:
    res = test_connectivites_for_neuron(neuron_idx, edges, labels, test_data, w)
    if res != -1:
        valid_neurons.append(neuron_idx)
        avgs.append(res)

plt.plot(valid_neurons, avgs)
plt.xlabel('neurons')
plt.ylabel('mean correct rate')
plt.title('Test neurons connectivities')
plt.show()