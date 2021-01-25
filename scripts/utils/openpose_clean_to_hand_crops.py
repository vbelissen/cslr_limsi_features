import numpy as np
import traitementOpenposeData.traitementPrincipal as trOP
import reconstruction.normalization as nor
import orientationTete.orientationTete as OT

from scipy.spatial import procrustes
from scipy import signal
import scipy.io

import Image
import sys

nimg            = int(sys.argv[1])
path2frames     = sys.argv[2]
vidName         = sys.argv[3]
framesExt       = sys.argv[4]
nDigits         = int(sys.argv[5])
path2features   = sys.argv[6]
path2handFrames = sys.argv[7]

clean_data = np.load(path2features+'openpose/clean_data/'+vidName+'.npz', allow_pickle=True)
a3 = clean_data['a3']
b3 = clean_data['b3']
c3 = clean_data['c3']
d3 = clean_data['d3']

larg_epaules_moy = np.mean(np.sqrt(np.square(a3[:,5,0]-a3[:,2,0])+np.square(a3[:,5,1]-a3[:,2,1])))
centreMainGD     = trOP.centreMainGD(a3[:,3,0:2],a3[:,4,0:2],a3[:,6,0:2],a3[:,7,0:2])

width  = int(1.3*larg_epaules_moy)
height = width

for i in range(nimg):
    im = Image.open(path2frames+vidName+'/'+str(i+1).zfill(nDigits)+'.'+framesExt)
    leftG = int(round(centreMainGD[i,0,1]-width/2))
    topG = int(round(centreMainGD[i,1,1]-height/2))
    boxG = (leftG, topG, leftG+width, topG+height)
    areaG = im.crop(boxG)
    areaG.save(path2handFrames+vidName+'/'+str(i+1).zfill(nDigits)+'_G.png', "PNG")
    #areaG_mirr = areaG.transpose(Image.FLIP_LEFT_RIGHT)
    #areaG_mirr.save('./results/testsLexique/lex_05_MainGD/'+str(i+1).zfill(4)+'_G_mirr.png', "PNG")
    leftD = int(round(centreMainGD[i,0,0]-width/2))
    topD = int(round(centreMainGD[i,1,0]-height/2))
    boxD = (leftD, topD, leftD+width, topD+height)
    areaD = im.crop(boxD)
    areaD.save(path2handFrames+vidName+'/'+str(i+1).zfill(nDigits)+'_D.png', "PNG")
