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


def simulatedData():
    """Produce simulated data, following Turaga et al.

    Returns:
        ndarray, size=(num_neurons, timesteps): X
        ndarray, size=(num_neurons, num_neurons): A_True
        list of indexes: A_True[unshuffle] will recover original arrangement.
    """
    num_neurons = 60
    timesteps = 10000
    spike_dur = 500
    spike_strength = 1e-6

    def ABlock(n, sigma=0.2, connP=0.5, cap=0.2):
        A = np.random.normal(scale=0.2, size=(n, n))
        A[np.random.rand(n, n) >= connP] = 0
        A[A > cap] = cap
        A[A < -cap] = -cap
        return A

    true_A = np.zeros((num_neurons, num_neurons))
    sl = [slice(0, 20), slice(20, 40), slice(40, 60)]
    true_A[sl[0], sl[0]] = ABlock(num_neurons / 3)
    true_A[sl[1], sl[0]] = np.abs(ABlock(num_neurons / 3))
    true_A[sl[2], sl[0]] = ABlock(num_neurons / 3, connP=0.25)
    # true_A[sl[0], sl[1]] = np.zeros((num_neurons/3, num_neurons/3))
    true_A[sl[1], sl[1]] = ABlock(num_neurons / 3)
    true_A[sl[2], sl[1]] = np.abs(ABlock(num_neurons / 3, sigma=0.05, connP=0.4, cap=0.1))
    true_A[sl[0], sl[2]] = -np.abs(ABlock(num_neurons / 3))
    true_A[sl[1], sl[2]] = -np.abs(np.abs(ABlock(num_neurons / 3)))
    true_A[sl[2], sl[2]] = ABlock(num_neurons / 3)

    shuffle = np.random.permutation(num_neurons)
    unshuffle = [np.nonzero(shuffle == x) for x in range(len(shuffle))]
    shuf_A = true_A[shuffle]

    # Stimulus only to cell group 1
    num_stim_neurons = 20
    B = np.zeros((num_neurons, num_stim_neurons))
    B[:num_stim_neurons] = 1
    shuf_B = B[shuffle]

    # stimulus
    u = np.zeros((num_stim_neurons, timesteps))
    u[:] = ([spike_strength] * spike_dur + [0] * spike_dur) * (timesteps / (spike_dur * 2))

    # neural noise
    mu = np.zeros(num_neurons)
    Q = np.eye(num_neurons)

    X = np.zeros((num_neurons, timesteps))
    X[:, 0] = np.dot(shuf_B, u[:, 0]) + np.random.multivariate_normal(mean=mu, cov=Q)
    for i in range(1, timesteps):
        X[:, i] = (np.dot(shuf_A, X[:, i - 1]) + np.dot(shuf_B, u[:, i]) +
                  np.random.multivariate_normal(mean=mu, cov=Q))

    return X, shuf_A, unshuffle


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
    edges = np.zeros(shape=((n * (n + 1)) / 2, 2), dtype=np.int16)
    idx = 0

    for i in range(0, n):
        if self_edges:
            edges[idx][0] = i
            edges[idx][1] = i
            idx += 1
        for j in range(i + 1, n):
            edges[idx][0] = i
            edges[idx][1] = j
            idx += 1
    return edges
