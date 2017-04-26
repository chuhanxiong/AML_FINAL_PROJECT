#!/usr/bin/env python2.7

import numpy as np
from numpy.testing import assert_array_equal
# p = no of timesteps
# n = no of neurons


# Capture full 2-way directed correlations between all pairs of neurons
def fourWayDirected(data):
    (n, p) = data.shape
    # t = 20
    out = np.array(np.zeros(shape=(4, n, n)), dtype=np.int8)
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
    return out


def simpleUndirected(data):
    """Capture only undirected correlations between all pairs of neurons.

    Output is ~XOR of two inputs (true iff two neurons both firing or both not
    firing). n features per neuron. One n by n feature matrix per timestep.

    Args:
        data (np.array): (n, p) sized, for n neurons, p timesteps.

    Returns:
        [np.array]: p length list of nxn np.array feature matrix
        [np.array]: p length list of n length np.array label vectors
    """
    (n, p) = data.shape

    # for each sample (neuron at a timestep), one feature per neuron
    features = []
    feature = np.zeros((n, n), dtype=np.int8)
    labels = []
    label = np.zeros(n, dtype=np.int8)
    for t in range(p):
        for i in range(n):
            label[i] = data[i, t]
            for j in range(n):
                if i == j:
                    # Dep of neuron on itself at previous timestep
                    if t == 0:
                        # No data == no correlation
                        feature[i, i] = 0
                    else:
                        feature[i, i] = ~(data[i, t] ^ data[i, t - 1])
                else:
                    # Dep of neuron on other neuron at same timestep
                    feature[i, j] = ~(data[i, t] ^ data[j, t])
        if len(features) > 0:
            assert_array_equal(features[-1],feature)
        features.append(feature)
        if len(labels) > 0:
            assert_array_equal(labels[-1],label)
        labels.append(label)
    return features, labels
