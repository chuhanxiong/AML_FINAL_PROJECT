#!/usr/bin/env python2.7

import numpy as np
import scipy.io as sio


def get_dF_F1():
    dataset = sio.loadmat('Jan25_all_data_exp2.mat')
    data = dataset['dF_F1']
    return data.astype('int64')


def binarize(data,):
    d = np.std(data) + np.mean(data)
    return ((data >= d) * np.ones(shape=data.shape)).T

def getEdges(data):
    n = data.shape[0]
    edges = np.zeros(shape=((n*(n+1))/2, 2)).astype('int64')
    idx = 0
    
    for i in range(0, n):
        for j in range(i+1, n):
            edges[idx][0] = i
            edges[idx][1] = j
            idx += 1
        edges[idx][0] = i
        edges[idx][1] = i
        idx += 1
    return edges

