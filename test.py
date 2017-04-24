import numpy as np
import scipy.io as sio
import scipy.sparse as ss


# n = no of timesteps
# m = no of neurons 
dataset = sio.loadmat('Jan25_all_data_exp2.mat')

#for a particular timestep
data = dataset['dF_F1']
n = dataset['dF_F1'].shape[1]
m = dataset['dF_F1'].shape[0]

d = np.std(data)+np.mean(data)

data = ((data>=d)*np.ones(shape = dataset['dF_F1'].shape )).T
t = 20
#out = [[[0 for i in range(m)] for  i in range(m)] for i in range(4)]
out = np.array(np.zeros(shape = (4,m,m)))
for j in range(m):
    for k in range(m):
        if j==k:
            if t == 0:
                if data[k][t]==1:
                    out[0][j][k] = 1
                    out[1][j][k] = 1
                    out[2][j][k] = 1
                    out[3][j][k] = 1
            else:
                if data[j][t-1]==1 and data[k][t]==1:
                    out[0][j][k] = 1
                if data[j][t-1]==1 and data[k][t]==0:
                    out[1][j][k] = 1
                if data[j][t-1]==0 and data[k][t]==1:
                    out[2][j][k] = 1
                if data[j][t-1]==1 and data[k][t]==1:
                    out[3][j][k] = 1
        else:
            if data[j][t]==1 and data[k][t]==1:
                out[0][j][k] = 1
            if data[j][t]==1 and data[k][t]==0:
                out[1][j][k] = 1
            if data[j][t]==0 and data[k][t]==1:
                out[2][j][k] = 1
            if data[j][t]==1 and data[k][t]==1:
                out[3][j][k] = 1
print out[1,:,:]