#!/usr/bin/env python2.7

import numpy as np
# p = no of timesteps
# n = no of neurons


# Capture full 2-way directed correlations between all pairs of neurons
def fourWayDirected(data):
    (n, p) = data.shape
    # t = 20
    out = np.array(np.zeros(shape=(4, n, n)))
    for t in range(p):
        for j in range(n):
            for k in range(n):
                if j == k:
                    # Dep of neuron on itself at previous timestep
                    if t == 0:
                        if data[k][t] == 1:
                            out[0][j][k] = 1
                            out[1][j][k] = 1
                            out[2][j][k] = 1
                            out[3][j][k] = 1
                    else:
                        if data[j][t - 1] == 1 and data[k][t] == 1:
                            out[0][j][k] = 1
                        if data[j][t - 1] == 1 and data[k][t] == 0:
                            out[1][j][k] = 1
                        if data[j][t - 1] == 0 and data[k][t] == 1:
                            out[2][j][k] = 1
                        if data[j][t - 1] == 1 and data[k][t] == 1:
                            out[3][j][k] = 1
                else:
                    # Dep of neuron on other neuron at same timestep
                    if data[j][t] == 1 and data[k][t] == 1:
                        out[0][j][k] = 1
                    if data[j][t] == 1 and data[k][t] == 0:
                        out[1][j][k] = 1
                    if data[j][t] == 0 and data[k][t] == 1:
                        out[2][j][k] = 1
                    if data[j][t] == 1 and data[k][t] == 1:
                        out[3][j][k] = 1
    # print out[1, :, :]
    return out

# Capture only undirected correlations between all pairs of neurons
# Output is ~XOR of two inputs (true iff two neurons both firing or both not firing)
# Single feature per neuron pair per timestep
# Columns of output correspond to feature vectors
def simpleUndirected(data):
    (n, p) = data.shape
    # t = 20
    # out = np.array(np.zeros(shape=(4, n, n)))
    # for each sample (neuron at a timestep), one feature per neuron
    features = np.zeros((n * p, n))
    for t in range(p):
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Dep of neuron on itself at previous timestep
                    if t == 0:
                        # No data == no correlation
                        features[i * t, j] = 0
                    else:
                        features[i * t, j] = ~(data[i, t] ^ data[i, t - 1])
                else:
                    # Dep of neuron on other neuron at same timestep
                    features[i * t, j] = ~(data[i, t] ^ data[j, t])
    # print out[1, :, :]
    return features
