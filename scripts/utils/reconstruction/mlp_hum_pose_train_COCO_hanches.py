from __future__ import print_function
import numpy as np
import math
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop, adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json
#from keras.utils.visualize_util import plot

from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2, activity_l2

import matplotlib.pyplot as plt

import scipy.io

import normalization as nor


def DataGenerator(data_set, K, alphaX, alphaY, alphaZ):
    
    N = data_set.shape[0]
    M = data_set.shape[1]
        
    Xtot = np.zeros([K*N, 2*M/3])
    Ytot = np.zeros([K*N, M/3])
    
    temp1 = sorted(list(range(0, M, 3))+list(range(2, M, 3)))
    
    X = data_set[:, temp1]
    Y = data_set[:, 1::3]
    
    Xtot[0:N,:] = X
    Ytot[0:N,:] = Y

    rmx = np.zeros((3, 3))    
    rmy = np.zeros((3, 3))
    rmz = np.zeros((3, 3))
    
    for k in range(1,K):
        
        
        rx = np.random.uniform(-alphaX, alphaX)
        ry = np.random.uniform(-alphaY, alphaY)
        rz = np.random.uniform(-alphaZ, alphaZ)
        
        rmx[0][0] = 1
        rmx[1][1] = math.cos(math.radians(rx))
        rmx[1][2] = -math.sin(math.radians(rx))
        rmx[2][1] = math.sin(math.radians(rx))
        rmx[2][2] = math.cos(math.radians(rx))
        
        rmy[0][0] = math.cos(math.radians(ry))
        rmy[0][2] = math.sin(math.radians(ry))
        rmy[1][1] = 1
        rmy[2][0] = -math.sin(math.radians(ry))
        rmy[2][2] = math.cos(math.radians(ry))
        
        rmz[0][0] = math.cos(math.radians(rz))
        rmz[0][1] = -math.sin(math.radians(rz))
        rmz[1][0] = math.sin(math.radians(rz))
        rmz[1][1] = math.cos(math.radians(rz))
        rmz[2][2] = 1
        
        rm = np.dot(np.dot(rmx, rmy), rmz)
        
        data_set_tmp = data_set.reshape(M/3*N, 3)
        data_set_tmp = data_set_tmp.dot(rm)
        data_set_tmp = data_set_tmp.reshape(N, M)
        
        Xtot[k*N:(k+1)*N,:] = data_set_tmp[:, temp1]
        Ytot[k*N:(k+1)*N,:] = data_set_tmp[:, 1::3]
        
    Xtot, Ytot, temp5 = nor.normalizedataXY(Xtot,Ytot)    
    
    return Xtot, Ytot

# on enleve les jambes
pts_kept = np.array([0,1,2,3,4,5,6,7,8,11,14,15,16,17])
subset = 3*pts_kept
tmp1 = sorted(list(subset)+list(subset+1)+list(subset+2))

# the data
train_set = np.zeros((1,len(tmp1)))
for l in range(1,3):
    for i in range(1,26):
        mat = scipy.io.loadmat('mat/testi'+str(i)+'l'+str(l)+'_COCO.mat')
        data1 = np.array(mat['allValuesCOCO'])
        data1[:, :, [1, 2]] = data1[:, :, [2, 1]]
        data1[:,:,0] = -data1[:,:,0]
        data2 = np.zeros((data1.shape[0],len(tmp1)))
        ind = 0
        for j in tmp1:
            data2[:,ind] = data1[:,j/3,j%3]
            ind = ind+1
        train_set = np.concatenate((train_set,data2))
        
train_set = train_set[1:train_set.shape[0],:]




val_set = np.zeros((1,len(tmp1)))
for l in range(3,4):
    for i in range(1,26):
        mat = scipy.io.loadmat('mat/testi'+str(i)+'l'+str(l)+'_COCO.mat')
        data1 = np.array(mat['allValuesCOCO'])
        data1[:, :, [1, 2]] = data1[:, :, [2, 1]]
        data1[:,:,0] = -data1[:,:,0]
        data2 = np.zeros((data1.shape[0],len(tmp1)))
        ind = 0
        for j in tmp1:
            data2[:,ind] = data1[:,j/3,j%3]
            ind = ind+1
        val_set = np.concatenate((val_set,data2))
        
val_set = val_set[1:val_set.shape[0],:]
            
X_train, Y_train = DataGenerator(train_set, 30, 10, 20, 45)

X_val, Y_val = DataGenerator(val_set, 30, 10, 20, 45)



X_train = X_train.astype('float32')
X_val = X_val.astype('float32')

Y_train = Y_train.astype('float32')
Y_val = Y_val.astype('float32')


print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'val samples')


n_pts = pts_kept.shape[0]
n_hid_layers = 2
add_batchnorm = False
Dropout_value = 0
act_type = 'tanh'


model = Sequential()

# Input layer
model.add(Dense(2*n_pts, input_dim=2*n_pts))
if add_batchnorm:
    model.add(BatchNormalization())
model.add(Activation(act_type))
if Dropout_value>0:
    model.add(Dropout(Dropout_value))

# Hidden layers
for n in range(n_hid_layers):
    model.add(Dense(2*n_pts))
    if add_batchnorm:
        model.add(BatchNormalization())
    model.add(Activation(act_type))
    if Dropout_value>0:
        model.add(Dropout(Dropout_value))

# Output layer
model.add(Dense(n_pts))

#model.add(Activation(act_type))
#model.add(Dense(n_pts))


#model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.005, momentum=0.0, decay=1e-6, nesterov=False))
model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.00001, rho=0.9, epsilon=1e-06))
#model.compile(loss='mean_squared_error', optimizer='RMSprop')
##model = model_from_json(open('mocap_mlp.json').read())
##model.load_weights('mocap_val_best.h5')
checkpointer = ModelCheckpoint(filepath="models/mocap_COCOhanches_val_best.h5", verbose=1, monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=15)

json_string = model.to_json()



history = model.fit(X_train, Y_train, batch_size=300, nb_epoch = 20, verbose=2, validation_data=(X_val, Y_val), callbacks=[early_stopping, checkpointer])

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

open('models/mocap_COCOhanches_mlp.json', 'w').write(json_string)       
model.save_weights('models/mocap_COCOhanches_mlp.h5', overwrite=True)
