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


def sessionize(data, num_sessions=3, neuron_split=None, time_split=None, overlap=0.1):
    """Produce sessionized version of data.

    Creates copy of data divided into sessions, splitting neurons and timesteps
    evenly, with overlap extra neurons in each session. Entries outside of
    sessions are set to zero.

    Args:
        data (ndarray), shape=(n,p): Data to wipe into sessions..
        num_sessions (int): Number of sessions. Sessions are created with
            floor(n/num_sessions * (1 + overlap)) neurons and
            floor(n/num_sessions) timesteps, with earlier sessions incremented
            first for remainders.
        neuron_split (iterable of index sets, optional): Will pick neurons
            following the indexes in each element. If provided, num_sessions is
            ignored.
        time_split (iterable of index sets, optional): Will pick timesteps
            following the indexes in each element. If neuron_split is not
            provided, this is ignored.
        overlap (float, optional): How much to extend each session by including
            neighboring neurons. If neuron_split is provided, this has no
            effect.

    Returns:
        (ndarray), shape=(n, p): data with entries outside sessions zeroed.
        (ndarray), shape=(num_sessions,): Entries are neurons session slices.
        (ndarray), shape=(num_sessions,): Entries are timeframe session slices.

    """
    (n, p) = data.shape
    if neuron_split is None:
        split = []
        start_idx = 0
        for k in range(num_sessions):
            size = n / int(num_sessions) + (k < (n % num_sessions))
            extra = int(size * overlap)
            end_idx = start_idx + size
            next_start_idx = end_idx
            if k < (num_sessions - 1):
                end_idx = min(n, end_idx + np.int(extra / 2))
            if k > 0:
                start_idx = max(0, start_idx - np.ceil(extra / 2))
            split.append(slice(int(start_idx), int(end_idx)))
            start_idx = next_start_idx
        neuron_split = np.array(split)

        split = []
        start_idx = 0
        for k in range(num_sessions):
            size = p / int(num_sessions) + (k < (p % num_sessions))
            end_idx = start_idx + size
            split.append(slice(start_idx, end_idx))
            start_idx = end_idx
        time_split = np.array(split)
    else:
        # Set neuron_split
        # TODO: Implement specifying session indexes?
        raise Exception("Specifying session indexes not implemented.")

    sessioned = np.zeros_like(data)
    for k in range(num_sessions):
        sessioned[neuron_split[k]][:, time_split[k]] = data[neuron_split[k]][:, time_split[k]]
    return sessioned, neuron_split, time_split


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
