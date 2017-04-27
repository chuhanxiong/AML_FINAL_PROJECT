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
