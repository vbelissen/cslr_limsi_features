import numpy as np
import traitementOpenposeData.traitementPrincipal as trOP

from scipy.spatial import procrustes
from scipy import signal
import scipy.io

from PIL import Image
import sys

nimg            = int(sys.argv[1])
vidName         = sys.argv[2]
path2features   = sys.argv[3]

confMoy = 0.05
confMinPose = 0.05#0.2
confMinFace = 0.2
confMinHand = 0.1
savitzky_window = 17
savitzky_order = 6

(a, b, c, d) = trOP.dataReadTabPoseFaceHandLR(nimg, 0, path2features+'openpose/json/'+vidName+'/'+vidName+'_', 12, '_keypoints.json', typeData='pfh', typePose='COCO')

# Mise a nan des genoux et chevilles
a[:,  9:11, 0:2] = np.nan
a[:, 12:14, 0:2] = np.nan
a[:,  9:11,   2] = 0
a[:, 12:14,   2] = 0

(a0, b0, c0, d0) = trOP.prolongationNanDebutFin(a, b, c, d, typeData='pfh', typePose='COCO')


(a1, b1, c1, d1) = trOP.nettoyageComplet(a0, b0, c0, d0, confMoy, confMinPose, confMinFace, confMinHand, typeData='pfh')
(a1bis, b1bis, c1bis, d1bis) = trOP.prolongationNanDebutFin(a1, b1, c1, d1, typeData='pfh', typePose='COCO')
(a2, b2, c2, d2) = trOP.interpNan(a1bis, b1bis, c1bis, d1bis, nimg, typeData='pfh', typePose='COCO')
(a3, b3, c3, d3) = trOP.filtrageSavGol(a2, b2, c2, d2, savitzky_window, savitzky_order, typeData='pfh', typePose='COCO')
#(a3, b3, c3, d3) = trOP.interpNan(a1,b1,c1,d1,nimg,typeData='pfh', typePose='COCO')

np.savez(path2features+'openpose/clean_data/'+vidName+'_openpose_clean', a3=a3 , b3=b3, c3=c3, d3=d3)
