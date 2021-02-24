import numpy as np
import os
os.environ['GLOG_minloglevel'] = '2'

import caffe

from PIL import Image

import sys
import os.path


caffe.set_mode_gpu()
caffe.set_device(0)



nimg            = int(sys.argv[1])
vidName         = sys.argv[2]
nDigits         = int(sys.argv[3])
path2features   = sys.argv[4]
path2handFrames = sys.argv[5]
path2caffeModel = sys.argv[6]

""" Load net """
hand_net = caffe.Net(path2caffeModel+'mod_submit-net.prototxt',
                     path2caffeModel+'1miohands-v2.caffemodel',
                     caffe.TEST)

suffixes = ['_L', '_R']


for hand_suffix in suffixes:

    Tab_probs = np.zeros((nimg, 61))
    #Tab_predict  = np.zeros(n)
    #Tab_features = np.zeros((n, 1024))

    for j in range(nimg):

        """ Get all images """
        imgs = []
        #for it in range(n):
        #    # imgs.append(caffe.io.load_image('./ph2014-dev-set-handshape-annotations/' + it[0]))

        file_path = path2handFrames+vidName+'/'+str(j+1).zfill(nDigits)+hand_suffix+'.png'
        #if os.path.exists(file_path):
        imgs.append(caffe.io.load_image(file_path))
        imgs = np.asarray(imgs)

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
        imgBatch.append(transformer.preprocess('data', imgs[0, :, :, :]))

        """ Forward """
        imgBatch = np.asanyarray(imgBatch)  # have to batchlize; otherwise will regard channel as batch dimension
        # out = hand_net.forward_all(data = imgBatch)  # use forward all to process a batch of data

        hand_net.blobs['data'].reshape(*imgBatch.shape)
        hand_net.blobs['data'].data[...] = imgBatch  # to get all features
        out = hand_net.forward()

        #features = hand_net.blobs['pool5/7x7_s1'].data[0,:,0,0].copy()
        #Tab_features[j,:] = features

        """ Get temps """
        #predict = out['loss3/loss3'].argmax(axis = 1)
        #prob = [probs[predict[it]] for it, probs in enumerate(out['loss3/loss3'])]

        Tab_probs[j,:] = out['loss3/loss3'].ravel()#probs[:]
        #Tab_predict[j] = predict

    #Tab_predict = Tab_predict.astype(int)

    np.save(path2features+'temp/'+vidName+'_HS_probs'+hand_suffix, Tab_probs)
