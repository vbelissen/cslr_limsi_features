from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility


from keras.models import model_from_json
#from keras.utils.visualize_util import plot
from scipy.spatial import procrustes
import scipy.io
import normalization as nor




model = model_from_json(open('models/mocap_COCOhanches_mlp.json').read())
model.load_weights('models/mocap_COCOhanches_val_best.h5')



# remove legs
subset = 3*np.array([0,1,2,3,4,5,6,7,8,11,14,15,16,17])
tmp1 = sorted(list(subset)+list(subset+1)+list(subset+2))

test_set = np.zeros((1,len(tmp1)))
for l in range(4,5):
    for i in range(1,26):
        if i!=20: 
            mat = scipy.io.loadmat('mat/testi'+str(i)+'l'+str(l)+'_COCO.mat')
            data1 = np.array(mat['allValuesCOCO'])
            data1[:, :, [1, 2]] = data1[:, :, [2, 1]]
            data1[:,:,0] = -data1[:,:,0]
            data2 = np.zeros((data1.shape[0],len(tmp1)))
            ind = 0
            for j in tmp1:
                data2[:,ind] = data1[:,j/3,j%3]
                ind = ind+1
            test_set = np.concatenate((test_set,data2))
        
test_set = test_set[1:test_set.shape[0],:]





X_test, Y_test, temp5 = nor.normalizedata(test_set)
output = model.predict(X_test, batch_size=X_test.shape[0], verbose=2)
##
mlp_pred = np.zeros(test_set.shape)
mlp_pred[:, 0::3] = X_test[:, 0::2] 
mlp_pred[:, 2::3] = X_test[:, 1::2]
mlp_pred[:, 1::3] = output 
test_set[:, 1::3] = test_set[:, 1::3] - np.outer(test_set[:, 1::3].mean(axis = 1), np.ones((1, (test_set.shape[1])/3)))
test_set[:, 0::3] = test_set[:, 0::3] - np.outer(test_set[:, 0::3].mean(axis = 1), np.ones((1, (test_set.shape[1])/3)))
test_set[:, 2::3] = test_set[:, 2::3] - np.outer(test_set[:, 2::3].mean(axis = 1), np.ones((1, (test_set.shape[1])/3)))


err = model.evaluate(X_test, Y_test, batch_size=X_test.shape[0], verbose=2)
err = np.zeros((1, test_set.shape[0]))


for k in range(test_set.shape[0]):
    a = test_set[k].reshape(test_set.shape[1]/3, 3)
    b = mlp_pred[k].reshape(test_set.shape[1]/3, 3)
    mtx1, mtx2, err[0, k] = procrustes(a, b)