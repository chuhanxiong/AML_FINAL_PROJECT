import numpy as np
from pystruct.models import GraphCRF
from pystruct.learners import OneSlackSSVM
from pystruct.inference import get_installed
from util import *
from features import simpleUndirected
# debugging
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
inference_method = get_installed(["qpbo", "ad3", "lp"])[0]

 
# triangle
x_1 = np.array([[0, 1], [1, 0], [.4, .6]])
g_1 = np.array([[0, 1], [1, 2], [0, 2]])
# expected result
y_1 = np.array([1, 0, 1])
x = (x_1, g_1)
y = y_1
crf = GraphCRF(n_states=2, n_features=2, inference_method=inference_method)
model = OneSlackSSVM(model=crf, max_iter=100, C=100, check_constraints=False)
model.fit([x],[y])
print model.w.shape
print model.w