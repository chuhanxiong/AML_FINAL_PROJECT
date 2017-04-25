# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import scipy.io

mat=scipy.io.loadmat('file.mat')

data=mat['dF_F1']

import numpy as np

#a=np.ones((2,600))

import sklearn_crfsuite

#test of what I can feed to crf

a=np.chararray((2,600))

for i in xrange(a.shape[0]):
    for j in xrange(a.shape[1]):
        a[i][j]=True
        
        
y=np.chararray(2)
y[0]=True
y[1]=False

dict1={}

for i in xrange(a.shape[0]):
    dict1[str(i)]=a[i,:]
    
crf=sklearn_crfsuite.CRF()
crf.fit(dict1,y)
ans=crf.predict(dict1)