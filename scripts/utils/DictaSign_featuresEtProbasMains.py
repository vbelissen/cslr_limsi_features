import matplotlib 
#matplotlib.use('Agg')  # to solve the backend problem
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg

from sklearn.mixture import GMM

import numpy as np
import caffe

import Image

import sys
import os.path


caffe.set_mode_gpu()
caffe.set_device(0)







""" Load net """
hand_net = caffe.Net('./mod_submit-net.prototxt',
                     './1miohands-modelzoo-v2/1miohands-v2.caffemodel',
                     caffe.TEST)





""" Load in the dict for label and handShape """
with open('./handShapeDict.txt') as handDictFile:
    handDict = dict(it.strip().split('\t') for it in handDictFile.readlines())  # create dict in a line
handDict['0'] = 'None'


session = sys.argv[1]
task = sys.argv[2]
AB = sys.argv[3]
n = int(sys.argv[4])
m = 61

Tab_probs_D = np.zeros((n,m))
Tab_predict_D = np.zeros(n)
Tab_features_D = np.zeros((n,1024))


for j in range(n):
    
    """ Get all images """
    imgs = []
    #for it in range(n):
    #    # imgs.append(caffe.io.load_image('./ph2014-dev-set-handshape-annotations/' + it[0]))
    
    file_path = '/people/belissen/Videos/DictaSign/convert/img_hands/DictaSign_lsf_S'+session+'_T'+task+'_'+AB+'_front/' + str(j+1).zfill(5) + '_D.png'
    if os.path.exists(file_path):
        imgs.append(caffe.io.load_image(file_path))
        imgs = np.asarray(imgs)
        
        #print(str(100*j/n)+' %')
        
        
        
        
        
        """ Set img transformer """
        transformer = caffe.io.Transformer({'data': hand_net.blobs['data'].data.shape})
        transformer.set_mean('data', np.array([np.mean(imgs[:, :, :, 0])*255,  # use 0~255 range
                                               np.mean(imgs[:, :, :, 1])*255, 
                                               np.mean(imgs[:, :, :, 2])*255]))  
        transformer.set_transpose('data', (2,0,1))  # change dimension order
        transformer.set_channel_swap('data', (2,1,0))  # change colomns in colar dimenson (1st dim)
        transformer.set_raw_scale('data', 255.0)  # amplify to 0~255 range
        
        
        
        
        
        """ Form a batch of inputs """
        imgBatch = []
        #for it in range(n):
        imgBatch.append(transformer.preprocess('data', imgs[0, :, :, :]))
        
        """ Forward """
        imgBatch = np.asanyarray(imgBatch)  # have to batchlize; otherwise will regard channel as batch dimension
        # out = hand_net.forward_all(data = imgBatch)  # use forward all to process a batch of data
        
        hand_net.blobs['data'].reshape(*imgBatch.shape)
        hand_net.blobs['data'].data[...] = imgBatch  # to get all features
        out = hand_net.forward()
        
        features = hand_net.blobs['pool5/7x7_s1'].data[0,:,0,0].copy()
        Tab_features_D[j,:] = features
    
        
        """ Get labels """
    
        predict = out['loss3/loss3'].argmax(axis = 1)
        prob = [probs[predict[it]] for it, probs in enumerate(out['loss3/loss3'])]
        
        Tab_probs_D[j,:] = probs[:]
        Tab_predict_D[j] = predict

Tab_predict_D = Tab_predict_D.astype(int)


Tab_probs_G = np.zeros((n,m))
Tab_predict_G = np.zeros(n)
Tab_features_G = np.zeros((n,1024))


for j in range(n):
    
    """ Get all images """
    imgs = []
    #for it in range(n):
    #    # imgs.append(caffe.io.load_image('./ph2014-dev-set-handshape-annotations/' + it[0]))
    
    file_path = '/people/belissen/Videos/DictaSign/convert/img_hands/DictaSign_lsf_S'+session+'_T'+task+'_'+AB+'_front/' + str(j+1).zfill(5) + '_G.png'
    if os.path.exists(file_path):
        imgs.append(caffe.io.load_image(file_path))
        imgs = np.asarray(imgs)
        
        #print(str(100*j/n)+' %')
        
        
        
        
        
        """ Set img transformer """
        transformer = caffe.io.Transformer({'data': hand_net.blobs['data'].data.shape})
        transformer.set_mean('data', np.array([np.mean(imgs[:, :, :, 0])*255,  # use 0~255 range
                                               np.mean(imgs[:, :, :, 1])*255, 
                                               np.mean(imgs[:, :, :, 2])*255]))  
        transformer.set_transpose('data', (2,0,1))  # change dimension order
        transformer.set_channel_swap('data', (2,1,0))  # change colomns in colar dimenson (1st dim)
        transformer.set_raw_scale('data', 255.0)  # amplify to 0~255 range
        
        
        
        
        
        """ Form a batch of inputs """
        imgBatch = []
        #for it in range(n):
        imgBatch.append(transformer.preprocess('data', imgs[0, :, :, :]))
        
        """ Forward """
        imgBatch = np.asanyarray(imgBatch)  # have to batchlize; otherwise will regard channel as batch dimension
        # out = hand_net.forward_all(data = imgBatch)  # use forward all to process a batch of data
        
        hand_net.blobs['data'].reshape(*imgBatch.shape)
        hand_net.blobs['data'].data[...] = imgBatch  # to get all features
        out = hand_net.forward()
        
        features = hand_net.blobs['pool5/7x7_s1'].data[0,:,0,0].copy()
        Tab_features_G[j,:] = features
    
        
        """ Get labels """
    
        predict = out['loss3/loss3'].argmax(axis = 1)
        prob = [probs[predict[it]] for it, probs in enumerate(out['loss3/loss3'])]
        
        Tab_probs_G[j,:] = probs[:]
        Tab_predict_G[j] = predict

Tab_predict_G = Tab_predict_G.astype(int)


np.savez('/people/belissen/Videos/DictaSign/results/hands/'+'S'+session+'_task'+task+'_'+AB, Tab_probs_G=Tab_probs_G,Tab_predict_G=Tab_predict_G,Tab_features_G=Tab_features_G,Tab_probs_D=Tab_probs_D,Tab_predict_D=Tab_predict_D,Tab_features_D=Tab_features_D)
