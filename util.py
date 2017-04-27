#!/usr/bin/env python2.7

import numpy as np
import scipy.io as sio


def get_dF_F1():
    """Get raw dfF1 calcium imaging data.

    Returns:
        np.array: shape=(n, p), dtype=float
    """
    dataset = sio.loadmat('Jan25_all_data_exp2.mat')
    data = dataset['dF_F1']
    return data


def binarize(data):
    """Return data binarized: values >= mean + std ==> 1.

    Args:
        data (np.array): shape=(n, p), dtype=float

    Returns:
        np.array: shape=data.shape, dtype=int64
    """
    threshold = np.std(data) + np.mean(data)
    binary = np.zeros_like(data, dtype=np.int8)
    binary[data >= threshold] = 1
    return binary


def getEdges(data, self_edges=True):
    n = data.shape[0]
    edges = np.zeros(shape=((n*(n+1))/2, 2), dtype=np.int16)
    idx = 0

    for i in range(0, n):
        if self_edges:
            edges[idx][0] = i
            edges[idx][1] = i
            idx += 1
        for j in range(i+1, n):
            edges[idx][0] = i
            edges[idx][1] = j
            idx += 1
    return edges

def find_neuron_connectivities(neuron_idx, labels):
    p, n = labels.shape
    group_one = dict()
    group_zero = dict()
    for t in range(p):
        for i in range(n):
            if labels[t, i] == 1:
                if t not in group_one:
                    group_one[t] = [i]
                else:
                    group_one[t].append(i)

            else:
                if t not in group_zero:
                    group_zero[t] = [i]
                else:
                    group_zero[t].append(i)

    scores = np.zeros((n,))
    for t in range(p):
        if neuron_idx not in group_one[t]:
            # in group_zero
            scores[group_zero[t]] += 1
        else:
            scores[group_one[t]] += 1
    scores[neuron_idx] = -1
    if np.amax(scores) == 0:
        return []    
    return np.argwhere(scores == np.amax(scores)).flatten().tolist()