import numpy as np
import traitementOpenposeData.traitementPrincipal as trOP
import reconstruction.normalization as nor
import orientationTete.orientationTete as OT

from keras.models import model_from_json
from scipy.spatial import procrustes
from scipy import signal
import scipy.io

import Image
import sys

nimg            = int(sys.argv[1])
vidName         = sys.argv[2]
path2features   = sys.argv[3]
handOP          = bool(sys.argv[4])
faceOP          = bool(sys.argv[5])
body3D          = bool(sys.argv[6])
face3D          = bool(sys.argv[7])


# remove legs
pts_kept = np.array([0,1,2,3,4,5,6,7,8,11,14,15,16,17])

clean_data = np.load(path2features+'openpose/clean_data/'+vidName+'.npz', allow_pickle=True)
a3 = clean_data['a3']
b3 = clean_data['b3']
c3 = clean_data['c3']
d3 = clean_data['d3']

np.save(path2features+'final/'+vidName+'_2DBody', a3[:, pts_kept, :])

if handOP:
    np.savez(path2features+'final/'+vidName'+_2DHands.npz', handL_2D=c3, handR_2D=d3)

if faceOP:
    np.save(path2features+'final/'+vidName'+_2DFace', b3)

# Determination des angles de suivi de la tete
nez_ind      = np.where(pts_kept == 0)[0][0]
oeil1_ind    = np.where(pts_kept == 14)[0][0]
oeil2_ind    = np.where(pts_kept == 15)[0][0]
oreille1_ind = np.where(pts_kept == 16)[0][0]
oreille2_ind = np.where(pts_kept == 17)[0][0]

data_pose2D = np.copy(a3[:,:,0:2])

# Transfo des donnees pour pouvoir faire tourner la reconstruction
data_pose2D = trOP.transfo_data_OP_recons(data_pose2D, pts_kept)

# On change le signe des donnees pour que z soit vers le haut
# Attention, x va de droite a gauche quand on est en face !
data_pose2D = -data_pose2D

if body3D:
    model = model_from_json(open('reconstruction/models/mocap_COCOhanches_mlp.json').read())
    model.load_weights('reconstruction/models/mocap_COCOhanches_val_best.h5')

    mx = data_pose2D[:, 0::2].mean(axis = 1)
    mz = data_pose2D[:, 1::2].mean(axis = 1)

    # Prediction de la profondeur
    data_pose2D_normalized, temp = nor.normalizedataX(data_pose2D)
    depth_pose2D_normalized      = model.predict(data_pose2D_normalized, batch_size=data_pose2D_normalized.shape[0], verbose=2)

    # Donnees dimensionnelles
    # data_pose2D_dim  = np.multiply(data_pose2D_normalized,  np.outer(temp, np.ones((1, data_pose2D_normalized.shape[1]))))
    depth_pose3D_dim = np.multiply(depth_pose2D_normalized, np.outer(temp, np.ones((1, depth_pose2D_normalized.shape[1]))))

    #data_pose2D_dim[:, 0::2] = data_pose2D_dim[:, 0::2] + np.outer(mx, np.ones((1, (data_pose2D_dim.shape[1])/2)))
    #data_pose2D_dim[:, 1::2] = data_pose2D_dim[:, 1::2] + np.outer(mz, np.ones((1, (data_pose2D_dim.shape[1])/2)))

    data_pose3D_final = np.zeros((a3.shape[0], a3.shape[1], 4)) # x,y,z,conf
    data_pose3D_final[:,:,0] = a3[:, :, 0]
    data_pose3D_final[:,:,1] = depth_pose3D_dim
    data_pose3D_final[:,:,2] = a3[:, :, 1]
    data_pose3D_final[:,:,3] = a3[:, :, 2]

    np.save(path2features+'final/'+vidName'+_3DBody', data_pose3D_final)

    indices    = np.array([nez_ind, oeil1_ind, oeil2_ind, oreille1_ind, oreille2_ind])
    anglesTete = OT.eulerAnglesTete(data_pose2D_normalized, depth_pose2D_normalized, indices)
    anglesTete = np.mod(anglesTete, 360)-180

    np.save(path2features+'final/'+vidName'+_headAngles_from_3Dbody', anglesTete)


if face3D:
    # Distance moyenne entre les yeux
    if body3D:
        oeil1_OP = np.transpose(np.array([data_pose2D_dim[:, 2*oeil1_ind], output2[:, oeil1_ind], data_pose2D_dim[:, 2*oeil1_ind+1]]))
        oeil2_OP = np.transpose(np.array([data_pose2D_dim[:, 2*oeil2_ind], output2[:, oeil2_ind], data_pose2D_dim[:, 2*oeil2_ind+1]]))
    else:
        oeil1_OP = np.transpose(np.array([data_pose2D[:, 2*oeil1_ind], np.zeros(nimg), data_pose2D[:, 2*oeil1_ind+1]]))
        oeil1_OP = np.transpose(np.array([data_pose2D[:, 2*oeil2_ind], np.zeros(nimg), data_pose2D[:, 2*oeil2_ind+1]]))

    oeil12_OP     = oeil2_OP-oeil1_OP
    d_moy_OP_yeux = np.mean(np.sqrt(np.sum(np.square(oeil12_OP), axis=1)))

    mil_yeux_OP = (oeil2_OP+oeil1_OP)/2

    # Donnees visage 3D
    tabTot = np.load(path2features+'final/'+vidName'+_3DFace_predict_raw_temp.npy')
    tabTot[:,     0, :] *=-1
    tabTot[:,     1, :] *=-1
    tabTot[:, [1,2], :]  = tabTot[:, [2,1], :]
    oeil1_3DFace = np.mean(tabTot[36:42, :, :], axis=0)
    oeil2_3DFace = np.mean(tabTot[42:48, :, :], axis=0)
    oeil12_3DFace = oeil2_3DFace - oeil1_3DFace
    d_moy_3DFace_yeux = np.mean(np.sqrt(np.sum(np.square(oeil12_3DFace), axis=0)))

    tabTot = tabTot*d_moy_OP_yeux/d_moy_3DFace_yeux

    oeil1_3DFace = np.mean(tabTot[36:42, :, :], axis=0)
    oeil2_3DFace = np.mean(tabTot[42:48, :, :], axis=0)
    mil_yeux_3DFace = (oeil2_3DFace+oeil1_3DFace)/2

    for i in range(tabTot.shape[0]):
        tabTot[i, :, :] = tabTot[i, :, :]-mil_yeux_3DFace+np.transpose(mil_yeux_OP)

    tabTotFinal = np.zeros((nimg, tabTot.shape[0], 3))
    for i in range(tabTot.shape[0]):
        tabTotFinal[:, i, 0] = tabTot[i, 0, :]
        tabTotFinal[:, i, 1] = tabTot[i, 1, :]
        tabTotFinal[:, i, 2] = tabTot[i, 2, :]
    np.save(path2features+'final/'+vidName'+_3DFace_predict_raw', tabTotFinal)

    indices2 = np.array([30, 29, 28, 27, 0, 1, 2, 14, 15, 16, 31, 32, 33, 34, 35])
    tabTotXZ = np.zeros((nimg, 2*tabTot.shape[0]))
    tabTotY  = np.zeros((nimg,   tabTot.shape[0]))

    for i in range(tabTot.shape[0]):
        tabTotXZ[:, 2*i]   = tabTot[i, 0, :]
        tabTotXZ[:, 2*i+1] = tabTot[i, 2, :]
        tabTotY[:, i]      = tabTot[i, 1, :]

    anglesTete2 = OT.eulerAnglesTete(tabTotXZ,tabTotY,indices2)
    anglesTete2 = np.mod(anglesTete2, 360)-180

    np.save(path2features+'final/'+vidName'+_headAngles_from_3Dface', anglesTete2)
