#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 01:06:38 2017

@author: jonathanshor
"""

import numpy as np
import pystruct.models as models

n_samples=50; n_nodes=100; n_states=7; n_features=10
crf = models.GraphCRF(inference_method='max-product', n_states=n_states,
                   n_features=n_features)
