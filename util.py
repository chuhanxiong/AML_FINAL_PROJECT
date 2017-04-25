#!/usr/bin/env python2.7

import numpy as np
import scipy.io as sio


def get_dF_F1():
    dataset = sio.loadmat('Jan25_all_data_exp2.mat')
    data = dataset['dF_F1']
    return data


def binarize(data,):
    d = np.std(data) + np.mean(data)
    return ((data >= d) * np.ones(shape=data.shape)).T
