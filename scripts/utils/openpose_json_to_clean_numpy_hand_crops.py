import numpy as np
import traitementOpenposeData.traitementPrincipal as trOP
import reconstruction.normalization as nor
import orientationTete.orientationTete as OT


from keras.models import model_from_json
from scipy.spatial import procrustes
from scipy import signal
import scipy.io
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

import Image

import sys



model = model_from_json(open('reconstruction/models/mocap_COCOhanches_mlp.json').read())
model.load_weights('reconstruction/models/mocap_COCOhanches_val_best.h5')
#model.load_weights('mocap_mlp.h5')

session = sys.argv[1]
task = sys.argv[2]
AB = sys.argv[3]
n = int(sys.argv[4])

s1 = '/people/belissen/Videos/DictaSign/results/poses/openpose_files/DictaSign_lsf_S'+session+'_T'+task+'_'+AB+'_front/DictaSign_lsf_S'+session+'_T'+task+'_'+AB+'_front_'
s11 = ''
chiffres = 12
s2 ='_keypoints.json'

Ximg = 720
Yimg = 576
fps = 25
confMoy = 0.1 
confMinPose = 0.08#0.2
confMinFace = 0.4
confMinHand = 0.2
savitzky_window = 17
savitzky_order = 6
K = 4
separ = False
indice_personne = 0
typePose = 'COCO'
typeData = 'p'




(a,b,c,d) = trOP.dataReadTabPoseFaceHandLR(separ,n,indice_personne,s1,s11,chiffres,s2,typeData,typePose)
#plot_XYConf_PoseFaceHandLR(a,b,c,d,n,Ximg,Yimg,fps,colorPose,colorFace,colorHand,typeData,typePose)

# Mise a nan des genoux et chevilles
a[:,9:11,0:2] = np.nan
a[:,12:14,0:2] = np.nan
a[:,9:11,2] = 0
a[:,12:14,2] = 0

(a1,b1,c1,d1) = trOP.nettoyageComplet(a,b,c,d,confMoy,confMinPose,confMinFace,confMinHand,typeData)
#plot_XYConf_PoseFaceHandLR(a1,b1,c1,d1,n,Ximg,Yimg,fps,colorPose,colorFace,colorHand,typeData,typePose)

(a2,b2,c2,d2) = trOP.interpNan(a1,b1,c1,d1,n,typeData,typePose)
#plot_XYConf_PoseFaceHandLR(a2,b2,c2,d2,n,Ximg,Yimg,fps,colorPose,colorFace,colorHand,typeData,typePose)

(a3,b3,c3,d3) = trOP.filtrageSavGol(a2,b2,c2,d2,savitzky_window,savitzky_order,typeData,typePose)
#plot_XYConf_PoseFaceHandLR(a3,b3,c3,d3,n,Ximg,Yimg,fps,colorPose,colorFace,colorHand,typeData,typePose)

#(a4,b4,c4,d4) = effaceZeroKconsecutif(a3,b3,c3,d3,K,confMinPose,confMinFace,confMinHand,typeData,typePose)



# JUSTE POUR TESTER UN TRUC
#for i in range(n):
#    if(np.isnan(a3[i,8,0])):
#        a3[i,8,0] = 270
#        a3[i,8,1] = 400
#    
#    if(np.isnan(a3[i,11,0])):
#        a3[i,11,0] = 380
#        a3[i,11,1] = 400






larg_epaules_moy = np.mean(np.sqrt(np.square(a3[:,5,0]-a3[:,2,0])+np.square(a3[:,5,1]-a3[:,2,1])))

centreMainGD = trOP.centreMainGD(a3[:,3,0:2],a3[:,4,0:2],a3[:,6,0:2],a3[:,7,0:2])

width = int(1.3*larg_epaules_moy)
height = width

for i in range(n):
    im = Image.open('/people/belissen/Videos/DictaSign/convert/img/DictaSign_lsf_S'+session+'_T'+task+'_'+AB+'_front/'+str(i+1).zfill(5)+'.jpg')
    leftG = int(round(centreMainGD[i,0,1]-width/2))
    topG = int(round(centreMainGD[i,1,1]-height/2))
    boxG = (leftG, topG, leftG+width, topG+height)
    areaG = im.crop(boxG)
    areaG.save('/people/belissen/Videos/DictaSign/convert/img_hands/DictaSign_lsf_S'+session+'_T'+task+'_'+AB+'_front/'+str(i+1).zfill(5)+'_G.png', "PNG")
    #areaG_mirr = areaG.transpose(Image.FLIP_LEFT_RIGHT)
    #areaG_mirr.save('./results/testsLexique/lex_05_MainGD/'+str(i+1).zfill(4)+'_G_mirr.png', "PNG")
    leftD = int(round(centreMainGD[i,0,0]-width/2))
    topD = int(round(centreMainGD[i,1,0]-height/2))
    boxD = (leftD, topD, leftD+width, topD+height)
    areaD = im.crop(boxD)
    areaD.save('/people/belissen/Videos/DictaSign/convert/img_hands/DictaSign_lsf_S'+session+'_T'+task+'_'+AB+'_front/'+str(i+1).zfill(5)+'_D.png', "PNG")







# remove legs
pts_kept = np.array([0,1,2,3,4,5,6,7,8,11,14,15,16,17])

data1_pose = a3[:,:,0:2]
#data1_face = b3[:,:,0:2]

# Transfo des donnees pour pouvoir faire tourner la reconstruction
data2_pose = trOP.transfo_data_OP_recons(data1_pose,pts_kept)

# On change le signe des donnees pour que z soit vers le haut
# Attention, x va de droite a gauche quand on est en face !
data2_pose = -data2_pose
#data2_face = -data1_face

mx = data2_pose[:, 0::2].mean(axis = 1)
mz = data2_pose[:, 1::2].mean(axis = 1)


# Prediction de la profondeur
X_test,  temp5 = nor.normalizedataX(data2_pose)
output = model.predict(X_test, batch_size=X_test.shape[0], verbose=2)

# Donnees dimensionnelles
X_test2 = np.multiply(X_test, np.outer(temp5, np.ones((1, X_test.shape[1]))))
output2 = np.multiply(output, np.outer(temp5, np.ones((1, output.shape[1]))))

X_test2[:, 0::2] = X_test2[:, 0::2] + np.outer(mx, np.ones((1, (X_test2.shape[1])/2)))
X_test2[:, 1::2] = X_test2[:, 1::2] + np.outer(mz, np.ones((1, (X_test2.shape[1])/2)))



# Determination des angles de suivi de la tete
nez_ind = np.where(pts_kept == 0)[0][0]
oeil1_ind = np.where(pts_kept == 14)[0][0]
oeil2_ind = np.where(pts_kept == 15)[0][0]
oreille1_ind = np.where(pts_kept == 16)[0][0]
oreille2_ind = np.where(pts_kept == 17)[0][0]

indices = np.array([nez_ind,oeil1_ind,oeil2_ind,oreille1_ind,oreille2_ind])
anglesTete = OT.eulerAnglesTete(X_test,output,indices)


# Points a garder en dehors du contour du visage :
#pts_kept_no_contour = np.array(range(17,70))

# Points du visage censes etre fixes :
#pts_kept_vis_fixes = np.array([list(range(42,43))+list(range(27,37))+list(range(39,42))+list(range(45,48))+list(range(50,53))+list(range(61,64))])


# Determination d'un visage aplati, filtre (methode 1)
#visage_droit = OT.visDroit(data2_face,anglesTete,30,pts_kept_no_contour,pts_kept_vis_fixes,17,4)



# Distance moyenne entre les yeux
oeil1_OP = np.transpose(np.array([X_test2[:,2*oeil1_ind],output2[:,oeil1_ind],X_test2[:,2*oeil1_ind+1]]))
oeil2_OP = np.transpose(np.array([X_test2[:,2*oeil2_ind],output2[:,oeil2_ind],X_test2[:,2*oeil2_ind+1]]))
oeil12_OP = oeil2_OP-oeil1_OP
d_moy_OP_yeux = np.mean(np.sqrt(np.sum(np.square(oeil12_OP),axis=1)))

mil_yeux_OP = (oeil2_OP+oeil1_OP)/2

# Donnees visage 3D
tabTot = np.load('/people/belissen/Videos/DictaSign/results/faces/face_S'+session+'_task'+task+'_'+AB+'.npy')
tabTot[:,0,:] *=-1
tabTot[:,1,:] *=-1
tabTot[:,[1,2],:] = tabTot[:,[2,1],:]
oeil1_3DFace = np.mean(tabTot[36:42,:,:],axis = 0)
oeil2_3DFace = np.mean(tabTot[42:48,:,:],axis = 0)
oeil12_3DFace = oeil2_3DFace - oeil1_3DFace
d_moy_3DFace_yeux = np.mean(np.sqrt(np.sum(np.square(oeil12_3DFace),axis=0)))

tabTot = tabTot*d_moy_OP_yeux/d_moy_3DFace_yeux

oeil1_3DFace = np.mean(tabTot[36:42,:,:],axis = 0)
oeil2_3DFace = np.mean(tabTot[42:48,:,:],axis = 0)
mil_yeux_3DFace = (oeil2_3DFace+oeil1_3DFace)/2

for i in range(tabTot.shape[0]):
    tabTot[i,:,:] = tabTot[i,:,:]-mil_yeux_3DFace+np.transpose(mil_yeux_OP)


indices2 = np.array([30,29,28,27,0,1,2,14,15,16,31,32,33,34,35])
tabTotXZ = np.zeros((n,2*tabTot.shape[0]))
tabTotY = np.zeros((n,tabTot.shape[0]))

for i in range(tabTot.shape[0]):
    tabTotXZ[:,2*i] = tabTot[i,0,:] 
    tabTotXZ[:,2*i+1] = tabTot[i,2,:]
    tabTotY[:,i] = tabTot[i,1,:] 

anglesTete2 = OT.eulerAnglesTete(tabTotXZ,tabTotY,indices2)

tmp = True
while tmp == True:
    tmp = False
    for i in range(n):
        for j in range(3):
            if anglesTete[i,j]>180:
                anglesTete[i,j] -= 360
                tmp = True
            if anglesTete[i,j]<-180:
                anglesTete[i,j] += 360
                tmp = True
            if anglesTete2[i,j]>180:
                anglesTete2[i,j] -= 360
                tmp = True
            if anglesTete2[i,j]<-180:
                anglesTete2[i,j] += 360
                tmp = True




#
#
#
#plt.figure()
#ax = plt.subplot(311)
#ax.plot(anglesTete[:,0],label='tete 1 - angX')
#ax.plot(anglesTete2[:,0],label='tete 2 - angX')
#
#ax.legend()
#ax = plt.subplot(312)
#ax.plot(anglesTete[:,1],label='tete 1 - angY')
#ax.plot(anglesTete2[:,1],label='tete 2 - angY')
#
#ax.legend()
#ax = plt.subplot(313)
#ax.plot(anglesTete[:,2],label='tete 1 - angZ')
#ax.plot(anglesTete2[:,2],label='tete 2 - angZ')
#
#ax.legend()








    
#tabLienPose = np.array([]).astype(int)
#
#for i in range(0,4):
#    tabLienPose = np.append(tabLienPose,[i,i+1])
#tabLienPose = np.append(tabLienPose,[1,5])
#for i in range(5,7):
#    tabLienPose = np.append(tabLienPose,[i,i+1])
#tabLienPose = np.append(tabLienPose,[1,8])
##for i in range(8,10):
##    tabLienPose = np.append(tabLienPose,[i,i+1])
#tabLienPose = np.append(tabLienPose,[1,11])
##for i in range(11,13):
##    tabLienPose = np.append(tabLienPose,[i,i+1])
#tabLienPose = np.append(tabLienPose,[0,14,14,16,0,15,15,17])
#    
#tabLienPose = tabLienPose.reshape(-1,2)





#nimg = 33
#
#
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#ax.elev = 30
#ax.azim = 60







#xdata = X_test2[nimg, 0::2] 
#ydata = output2[nimg,:]
#zdata = X_test2[nimg, 1::2] 
#
#xmin = min(xdata)
#xmax = max(xdata)
#ymin = min(ydata)
#ymax = max(ydata)
#zmin = min(zdata)
#zmax = max(zdata)
#
#xmoy = (xmax + xmin)/2
#ymoy = (ymax + ymin)/2
#zmoy = (zmax + zmin)/2
#
#xdata = xdata - xmoy
#ydata = ydata - ymoy
#zdata = zdata - zmoy
#
#ax.scatter(xdata,ydata,zdata)
#
#ax.scatter(tabTot[:,0,nimg]-xmoy,tabTot[:,1,nimg]-ymoy,tabTot[:,2,nimg]-zmoy,s=0.5)
#
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
#
#Ax = xmax - xmin
#Ay = ymax - ymin
#Az = zmax - zmin
#
#Amax = max([Ax,Ay,Az])
#
#ax.set_xlim(-0.6*Amax, 0.6*Amax)
#ax.set_ylim(-0.6*Amax, 0.6*Amax)
#ax.set_zlim(-0.6*Amax, 0.6*Amax)
#
#ax.plot3D(tabTot[:17,0,nimg]-xmoy,tabTot[:17,1,nimg]-ymoy, tabTot[:17,2,nimg]-zmoy, color='blue' )
#ax.plot3D(tabTot[17:22,0,nimg]-xmoy,tabTot[17:22,1,nimg]-ymoy, tabTot[17:22,2,nimg]-zmoy, color='blue')
#ax.plot3D(tabTot[22:27,0,nimg]-xmoy,tabTot[22:27,1,nimg]-ymoy, tabTot[22:27,2,nimg]-zmoy, color='blue' )
#ax.plot3D(tabTot[27:31,0,nimg]-xmoy,tabTot[27:31,1,nimg]-ymoy, tabTot[27:31,2,nimg]-zmoy, color='blue' )
#ax.plot3D(tabTot[31:36,0,nimg]-xmoy,tabTot[31:36,1,nimg]-ymoy, tabTot[31:36,2,nimg]-zmoy, color='blue' )
#ax.plot3D(tabTot[36:42,0,nimg]-xmoy,tabTot[36:42,1,nimg]-ymoy, tabTot[36:42,2,nimg]-zmoy, color='blue' )
#ax.plot3D(tabTot[42:48,0,nimg]-xmoy,tabTot[42:48,1,nimg]-ymoy, tabTot[42:48,2,nimg]-zmoy, color='blue' )
#ax.plot3D(tabTot[48:,0,nimg]-xmoy,tabTot[48:,1,nimg]-ymoy, tabTot[48:,2,nimg]-zmoy, color='blue' )
#
#
#for i in range(tabLienPose.shape[0]):
#    ideb = tabLienPose[i,0]
#    ind_deb = np.where(pts_kept == ideb)[0][0]
#    ifin = tabLienPose[i,1]
#    ind_fin = np.where(pts_kept == ifin)[0][0]
#    #On ne cree un lien que si le point a ete detecte, c'est a dire
#    #si ses coordonnees ne sont pas a 0
#    plt.plot([xdata[ind_deb],xdata[ind_fin]],[ydata[ind_deb],ydata[ind_fin]],[zdata[ind_deb],zdata[ind_fin]])#,color = 'blue')
#
#fig.tight_layout()
#
#
#
#
#
#
#fig = plt.figure()
#
##centrage au niveau des hanches
#
#
#
#xmin = np.min(X_test2[:, 0::2])
#xmax = np.max(X_test2[:, 0::2])
#ymin = np.min(output2)
#ymax = np.max(output2)
#zmin = np.min(X_test2[:, 1::2])
#zmax = np.max(X_test2[:, 1::2])
#
#xmoy = (xmax + xmin)/2
#ymoy = (ymax + ymin)/2
#zmoy = (zmax + zmin)/2
#
#Ax = xmax - xmin
#Ay = ymax - ymin
#Az = zmax - zmin
#
#Amax = max([Ax,Ay,Az])
#
#for nimg in range(data1_pose.shape[0]):
#    
#    print(str(int(float(nimg)/data1_pose.shape[0]*100))+' %')
#    plt.clf()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.elev = 10#30
#    ax.azim = 90+30*np.sin(nimg*1.0/80*np.pi) #80#+nimg
#
#
#    xdata = X_test2[nimg, 0::2] 
#    ydata = output2[nimg,:]
#    zdata = X_test2[nimg, 1::2] 
#    
#    xdata = xdata - xmoy
#    ydata = ydata - ymoy
#    zdata = zdata - zmoy
#
#
#
#    ax.scatter(xdata,ydata,zdata,s=1)
#    
#    ax.scatter(tabTot[:,0,nimg]-xmoy,tabTot[:,1,nimg]-ymoy,tabTot[:,2,nimg]-zmoy,s=0.5)
#    
#    ax.set_xlabel('X Label')
#    ax.set_ylabel('Y Label')
#    ax.set_zlabel('Z Label')
#    
#    
#    
#    ax.set_xlim(-0.6*Amax, 0.6*Amax)
#    ax.set_ylim(-0.6*Amax, 0.6*Amax)
#    ax.set_zlim(-0.6*Amax, 0.6*Amax)
#
#    ax.plot3D(tabTot[:17,0,nimg]-xmoy,tabTot[:17,1,nimg]-ymoy, tabTot[:17,2,nimg]-zmoy, color='blue' )
#    ax.plot3D(tabTot[17:22,0,nimg]-xmoy,tabTot[17:22,1,nimg]-ymoy, tabTot[17:22,2,nimg]-zmoy, color='blue')
#    ax.plot3D(tabTot[22:27,0,nimg]-xmoy,tabTot[22:27,1,nimg]-ymoy, tabTot[22:27,2,nimg]-zmoy, color='blue' )
#    ax.plot3D(tabTot[27:31,0,nimg]-xmoy,tabTot[27:31,1,nimg]-ymoy, tabTot[27:31,2,nimg]-zmoy, color='blue' )
#    ax.plot3D(tabTot[31:36,0,nimg]-xmoy,tabTot[31:36,1,nimg]-ymoy, tabTot[31:36,2,nimg]-zmoy, color='blue' )
#    ax.plot3D(tabTot[36:42,0,nimg]-xmoy,tabTot[36:42,1,nimg]-ymoy, tabTot[36:42,2,nimg]-zmoy, color='blue' )
#    ax.plot3D(tabTot[42:48,0,nimg]-xmoy,tabTot[42:48,1,nimg]-ymoy, tabTot[42:48,2,nimg]-zmoy, color='blue' )
#    ax.plot3D(tabTot[48:,0,nimg]-xmoy,tabTot[48:,1,nimg]-ymoy, tabTot[48:,2,nimg]-zmoy, color='blue' )
#
#    for i in range(tabLienPose.shape[0]):
#        ideb = tabLienPose[i,0]
#        ind_deb = np.where(pts_kept == ideb)[0][0]
#        ifin = tabLienPose[i,1]
#        ind_fin = np.where(pts_kept == ifin)[0][0]
#        #On ne cree un lien que si le point a ete detecte, c'est a dire
#        #si ses coordonnees ne sont pas a 0
#        plt.plot([xdata[ind_deb],xdata[ind_fin]],[ydata[ind_deb],ydata[ind_fin]],[zdata[ind_deb],zdata[ind_fin]])#,color = 'blue')
#    
#    fig.tight_layout()
#    #plt.savefig('./results/testsLexique/lex_05/'+str(nimg).zfill(4)+'.jpg',dpi=250)
#    plt.pause(0.0001)

np.savez('/people/belissen/Videos/DictaSign/results/poses/3D_npy/'+'S'+session+'_task'+task+'_'+AB, tabTot=tabTot, X_test2=X_test2, output2=output2, anglesTete=anglesTete, anglesTete2=anglesTete2)
