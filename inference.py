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

inference_method = get_installed(["qpbo", "ad3", "lp"])[0]

# time_frame = 100
# print 'time_frame is', time_frame
# data = get_dF_F1()
# print 'Original data shape', data.shape
# test_start_idx = random.randint(time_frame, data.shape[1])
# test_data = data[:,test_start_idx:test_start_idx+time_frame]
# test_data = binarize(test_data)

# train_data = data[:, :time_frame]
# train_data = binarize(train_data)




X, shuf_A_true, unshuffle = simulatedData(n=18, T=1000)
A_true = shuf_A_true[unshuffle][:, unshuffle]
A_true[A_true != 0] = 1
X_data = binarize(X[unshuffle])
print X_data
train_size = 200
test_size = 200
# X_train = sessionize(X_data[:, :train_size], num_sessions=2)[0]
X_train = X_data
X_test = X_data[:, -test_size:]

train_data = X_train
test_data = X_test



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
adjMatrixBothOnes = np.dot(labels.T, labels)
            
# df = pd.DataFrame(adjacencyMatrix)
# df.to_csv('adjacencyMatrix.csv', index=False, header=False)
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

# plt.scatter(list1x, list1y, c='r')
# plt.show()

list2x = []
list2y = []
s2 = set()
print A_true.shape
for i in range(A_true.shape[0]):
    for j in range(A_true.shape[1]):
        if A_true[i, j] == 1:
            list2x.append(i)
            list2y.append(j)
            s2.add((i, j))
n = A_true.shape[0]
assert adjacencyMatrix.shape[0] == n
TP = np.count_nonzero(A_true[~np.eye(n, dtype=bool)] == adjacencyMatrix[~np.eye(n, dtype=bool)])
TP = np.float(TP)
print("Precision: {0:.2f}%".format(TP / np.count_nonzero(adjacencyMatrix[~np.eye(n, dtype=bool)])))
print("Recall: {0:.2f}%".format(TP / np.count_nonzero(A_true[~np.eye(n, dtype=bool)])))
# plt.scatter(list2x, list2y, c='b')
# plt.show()

s3 = s1.intersection(s2)
list3x = []
list3y = []
for x,y in s3:
    list3x.append(x)
    list3y.append(y)


# plt.scatter(list3x, list3y, c='g')
# plt.show()

print 'overlap rate for s2', 1.0*len(s3)/len(s2) #60%
print 'overlap rate for s1', 1.0*len(s3)/len(s1) #40%
# edges = getEdges(train_data)
# print 'get edges'
# edge_features = getEdgeFeatures(features, edges.shape)
# print 'get edge_features'

# train_Y = labels
# train_X = list(zip(features, [edges]*len(features), edge_features))

# print 'fitting the model'
# model.fit(train_X, train_Y)
# w = model.w

# print 'getting weights'
# with open('weights.txt') as f:
#     content = f.readlines()
# w = []
# for line in content:
#     tokens = line.strip().split()
#     w += tokens
# w = np.asarray(w, dtype='float32')

# print 'initializing crf'
# crf.initialize(train_X, train_Y)

# print 'test neurons connectivities'
# neurons = [18]
# valid_neurons = []
# avgs = []
# for neuron_idx in neurons:
#     res = test_connectivites_for_neuron(neuron_idx, edges, labels, test_data, w)
#     if res != -1:
#         valid_neurons.append(neuron_idx)
#         avgs.append(res)

# plt.plot(valid_neurons, avgs)
# plt.xlabel('neurons')
# plt.ylabel('mean correct rate')
# plt.title('Test neurons connectivities')
# plt.show()




# print 'building sessions'
# A = range(53)
# B = range(53, 105)
# C = range(105, n)

# sessionA = test_data[:,:50]
# sessionA[B+C,:] = 0
# print 'sessionA shape', sessionA.shape
# print 'non-zero entries', np.sum(sessionA)
# A_features, A_Y = simpleUndirected(sessionA)
# A_edge_features = getEdgeFeatures(A_features, edges.shape)
# A_X = list(zip(A_features, [edges]*len(A_features), A_edge_features))

# sessionB = test_data[:,50:100]
# sessionB[A+C,:] = 0
# print 'sessionB shape', sessionB.shape
# print 'non-zero entries', np.sum(sessionB)
# B_features, B_Y = simpleUndirected(sessionB)
# B_edge_features = getEdgeFeatures(B_features, edges.shape)
# B_X = list(zip(B_features, [edges]*len(B_features), B_edge_features))

# sessionC = test_data[:,100:]
# sessionC[A+B,:] = 0
# print 'sessionC shape', sessionC.shape
# print 'non-zero entries', np.sum(sessionC)
# C_features, C_Y = simpleUndirected(sessionC)
# C_edge_features = getEdgeFeatures(C_features, edges.shape)
# C_X = list(zip(C_features, [edges]*len(C_features), C_edge_features))


# print 'doing the inference'

# print 'test sessionA'
# A_correct_rate_list = []
# for x, y in zip(A_X, A_Y):
#     y_hat = crf.inference(x, w)
#     correct_rate = np.zeros(shape=(y_hat.shape))
#     correct_rate[y_hat==y] = 1
#     A_correct_rate_list.append(np.sum(correct_rate)/(1.0*len(correct_rate)))
#     print 'correct_rate', A_correct_rate_list[-1]
# print 'A_correct_rate_mean:', np.mean(A_correct_rate_list)

# print 'test sessionB'
# B_correct_rate_list = []
# for x, y in zip(B_X, B_Y):
#     y_hat = crf.inference(x, w)
#     correct_rate = np.zeros(shape=(y_hat.shape))
#     correct_rate[y_hat==y] = 1
#     B_correct_rate_list.append(np.sum(correct_rate)/(1.0*len(correct_rate)))
#     print 'correct_rate', B_correct_rate_list[-1]
# print 'B_correct_rate_mean:', np.mean(B_correct_rate_list)

# print 'test sessionC'
# C_correct_rate_list = []
# for x, y in zip(C_X, C_Y):
#     y_hat = crf.inference(x, w)
#     correct_rate = np.zeros(shape=(y_hat.shape))
#     correct_rate[y_hat==y] = 1
#     C_correct_rate_list.append(np.sum(correct_rate)/(1.0*len(correct_rate)))
#     print 'correct_rate', C_correct_rate_list[-1]
# print 'C_correct_rate_mean:', np.mean(C_correct_rate_list)