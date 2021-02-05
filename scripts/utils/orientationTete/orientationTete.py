import numpy as np
from scipy.optimize import minimize
from scipy import signal

# Fonction qui contruit une tete et une tete moyennee
# il faut lui donner les indices correspondant a la tete
def constructionTeteTeteMoy(X,Y,indices):
    ptsTete = indices.shape[0]
    tete = np.zeros(((X.shape[0],ptsTete,3)))
    for i in range(ptsTete):
        tete[:,i,0] = X[:,2*indices[i]]-X[:,2*indices[0]]
        tete[:,i,1] = Y[:,indices[i]]-Y[:,indices[0]]
        tete[:,i,2] = X[:,2*indices[i]+1]-X[:,2*indices[0]+1]
    tete_moy = np.mean(tete,axis=0)
    return tete, tete_moy

# Fonction qui fait tourner une matrice d'un certain angle
# autour d'un certain axe
def rotateMatrix(matrix,angle,axis):
    angleRad = angle*np.pi/180

    mult = 1
    if axis == 'y':
        mult = -1

    rotMat = np.zeros((3,3))
    rotMat1 = np.array([[np.cos(angleRad),-mult*np.sin(angleRad)],[mult*np.sin(angleRad),np.cos(angleRad)]])

    if axis == 'x':
        rotMat[1:3,1:3] = rotMat1
        rotMat[0,0] = 1
    elif axis == 'y':
        rotMat[0,0] = rotMat1[0,0]
        rotMat[0,2] = rotMat1[0,1]
        rotMat[2,0] = rotMat1[1,0]
        rotMat[2,2] = rotMat1[1,1]
        rotMat[1,1] = 1
    elif axis == 'z':
        rotMat[0:2,0:2] = rotMat1
        rotMat[2,2] = 1

    return np.transpose(np.matmul(rotMat,np.transpose(matrix)))

# Fonctio qui va rendre la tete moyennee parfaitement
# symetrique
def symmetrizeTeteMoy(tete_moy):

    angleTestZ = np.linspace(-45,45,300)
    nb_angles = angleTestZ.shape[0]
    disymetrieY = np.zeros(nb_angles)
    for i in range(nb_angles):
        rotated = rotateMatrix(tete_moy,angleTestZ[i],'z')
        disymetrieY[i] = np.square(rotated[1,1]-rotated[2,1])+np.square(rotated[3,1]-rotated[4,1])
    indminZ = np.argmin(disymetrieY)
    tete_moyZ = rotateMatrix(tete_moy,angleTestZ[indminZ],'z')
    #ax.scatter(tete_moyZ[:,0],tete_moyZ[:,1],tete_moyZ[:,2],color='red')


    angleTestY = np.linspace(-20,20,200)
    nb_angles = angleTestY.shape[0]
    disymetrieZ = np.zeros(nb_angles)
    for i in range(nb_angles):
        rotated = rotateMatrix(tete_moyZ,angleTestY[i],'y')
        disymetrieZ[i] = np.square(rotated[1,2]-rotated[2,2])+np.square(rotated[3,2]-rotated[4,2])
    indminY = np.argmin(disymetrieZ)
    tete_moyZY = rotateMatrix(tete_moyZ,angleTestY[indminY],'y')
    #ax.scatter(tete_moyZY[:,0],tete_moyZY[:,1],tete_moyZY[:,2],color='green')
    return tete_moyZY




# Fonction qui calcule pour les 3 angles d'Euler (inverses) la difference entre les 2 tetes
def diffRotationXYZ(tete,tete_centree,alpha,theta,psi):
    return np.sum(np.square(rotateMatrix(rotateMatrix(rotateMatrix(tete,alpha,'x'),theta,'y'),psi,'z')-tete_centree))

# Fonction qui va calculer a chaque pas de temps les angles d'Euler
# qui suivent l'orentation de la tete
def eulerAnglesTete(X,Y,indices):

    anglesTete = np.zeros((X.shape[0],3))

    tete, tete_moy = constructionTeteTeteMoy(X,Y,indices)
    tete_moy_symm = tete_moy#symmetrizeTeteMoy(tete_moy)

    # tmp = 0
    for i in range(X.shape[0]):
        # if (100*i/X.shape[0])%5 == 0 and tmp == 0:
        #     print('Getting Euler angles: '+str(100*i/X.shape[0])+' %')
        #     tmp = 1
        # if (100*i/X.shape[0])%5 != 0:
        #     tmp = 0
        def erreurAngles(angles):
            return diffRotationXYZ(tete[i,:,:],tete_moy_symm,angles[0],angles[1],angles[2])

        res = minimize(erreurAngles, [0,0,0], method='BFGS',options={'gtol': 1e-6, 'disp': False})
        anglesTete[i,:] = -res.x
    # print('Getting Euler angles: '+str(100)+' %')
    return anglesTete


# Plusieurs fonctions servant a obtenir un visage plus droit
# Cette fonction applique une rotation uniforme et ou proportionnelle a la
# distance au centre
def rotationDeformation(X,theta_0,theta_1):

    r = np.sqrt(np.square(X[0])+np.square(X[1]))
    theta = theta_0 + theta_1*r

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([X[0]*c-X[1]*s,X[0]*s+X[1]*c])


# Jacobienne, 1ere ligne = gradient de rotationDeformation[0], 2eme ligne = gradient de rotationDeformation[1]
#def rotationDeformation_grad(X,theta_0,theta_1):
#
#    r = np.sqrt(np.square(X[0])+np.square(X[1]))
#
#    return np.outer(np.matmul(np.array([[0,-1],[1,0]]),rotationDeformation(X,theta_0,theta_1)),np.array([1,r]))
#
#def rotationDeformation_dX(X,theta_0,theta_1):
#
#    r = np.sqrt(np.square(X[0])+np.square(X[1]))
#    theta = theta_0 + theta_1*r
#
#    c = np.cos(theta)
#    s = np.sin(theta)
#
#    return np.transpose(np.array([[c,s],[-s,c]]))

# Fonction qui etire l'espace de alpha suivant x et de beta suivant z
def stretching(X,alpha,beta):

    return np.array([X[0]*alpha,X[1]*beta])

#def stretching_grad(X,alpha,beta):
#
#    return np.transpose(np.vstack([stretching(X,1,0),stretching(X,0,1)]))
#
#def stretching_dX(X,alpha,beta):
#
#    return np.transpose([[alpha,0],[0,beta]])

# Fonction qui effectue une translation de l'espace
# en respectant toutefois que le centre ne doit pas bouger
def ecrasementTranslation(X,X01,X02,gamma):

    r = np.sqrt(np.square(X[0])+np.square(X[1]))

    return X + (1-np.exp(-gamma*r))*np.array([X01,X02])
    #return X + (r**gamma)*np.array([X01,X02])

#def ecrasementTranslation_grad(X,X01,X02,gamma):
#
#    r = np.sqrt(np.square(X[0])+np.square(X[1]))
#
#    G1 = (1-np.exp(-gamma*r))*np.array([1,0])
#    G2 = (1-np.exp(-gamma*r))*np.array([0,1])
#    G3 = r*np.exp(-gamma*r)*np.array([X01,X02])
#
#    return np.transpose(np.vstack([G1,G2,G3]))
#
#def ecrasementTranslation_dX(X,X01,X02,gamma):
#
#    return np.identity(2)

def courbe(X,phi):

    if X[1]>0:
        t = 1
    else:
        t = 0
    return np.array([X[0]+phi*t*X[1],X[1]])


# Fonction qui applique successivement les 3 transformations precedentes
def transfoVisage(X,theta_0,theta_1,alpha,beta,X01,X02,gamma):

    return ecrasementTranslation(stretching(rotationDeformation(X,theta_0,theta_1),alpha,beta),X01,X02,gamma) #5151
    #return stretching(ecrasementTranslation(rotationDeformation(X,theta_0,theta_1),X01,X02,gamma),alpha,beta) #5148
    #return stretching(rotationDeformation(ecrasementTranslation(X,X01,X02,gamma),theta_0,theta_1),alpha,beta) #nan
    #return rotationDeformation(stretching(ecrasementTranslation(X,X01,X02,gamma),alpha,beta),theta_0,theta_1) #nan
    #return rotationDeformation(ecrasementTranslation(stretching(X,alpha,beta),X01,X02,gamma),theta_0,theta_1) #nan
    #return ecrasementTranslation(rotationDeformation(stretching(X,alpha,beta),theta_0,theta_1),X01,X02,gamma) # 5161


#def transfoVisage_grad(X,theta_0,theta_1,alpha,beta,X01,X02,gamma):
#
#    return ecrasementTranslation_grad(stretching_grad(rotationDeformation_grad(X,theta_0,theta_1),alpha,beta),X01,X02,gamma)

# Fonction qui calcule la difference entre un visage et le visage moyen
# (a n'appliquer que sur les points censes rester fixes du visage)
def diffVisVisMoy(vis,vis_moy,theta_0,theta_1,alpha,beta,X01,X02,gamma):
    diff = 0
    visTransfo = np.zeros((vis.shape))
    for i in range(vis.shape[0]):
        visTransfo[i,:] = transfoVisage(vis[i,:],theta_0,theta_1,alpha,beta,X01,X02,gamma)
        diff = diff + np.square(visTransfo[i,0]-vis_moy[i,0]) + np.square(visTransfo[i,1]-vis_moy[i,1])
    return diff

#def diffVisVisMoy_grad(vis,vis_moy,theta_0,theta_1,alpha,beta,X01,X02,gamma):
#    sum1 = 0
#    sum2 = 0
#    sum3 = 0
#    visTransfo = np.zeros((vis.shape))
#    for i in range(vis.shape[0]):
#        visTransfo[i,:] = transfoVisage(vis[i,:],theta_0,theta_1,alpha,beta,X01,X02,gamma)
#        sum1 = sum1 + visTransfo[i,0]-vis_moy[i,0]
#        sum2 = sum2 + visTransfo[i,1]-vis_moy[i,1]
#        sum3 = 2*np.outer(np.transpose(transfoVisage_grad(vis[i,:],theta_0,theta_1,alpha,beta,X01,X02,gamma)),(sum1+sum2))
#
#    return sum3

# Fonction qui obtient pour chaque pas de temps les parametres
# des 3 transformations qui minimisent l'ecart entre le visage
# et le visage moyen, en plus d'empecher les parametres
# de prendre de trop grandes valeurs
def paramsTransfo(visRedTot,vis_RedMoyTot):

    params = np.zeros((visRedTot.shape[0],7))
    params_init = [0,0,1,1,0,0,0.5]

    tmp = 0
    for i in range(visRedTot.shape[0]):
        if (100*i/visRedTot.shape[0])%5 == 0 and tmp == 0:
            print('Rectifying face: '+str(100*i/visRedTot.shape[0])+' %')
        tmp = 1
        if (100*i/visRedTot.shape[0])%5 != 0:
            tmp = 0
        def erreurParams(parametres):
            return 0.05*np.sum(np.square(parametres - params_init)) + diffVisVisMoy(visRedTot[i,:,:],vis_RedMoyTot[:,:],parametres[0],parametres[1],parametres[2],parametres[3],parametres[4],parametres[5],parametres[6])

#        def gradParams(parametres):
#            return 0.1*2*np.outer(np.transpose(parametres),np.sum(parametres-[0,0,1,1,0,0,1]))+diffVisVisMoy_grad(visRedTot[i,:,:],vis_RedMoyTot[:,:],parametres[0],parametres[1],parametres[2],parametres[3],parametres[4],parametres[5],parametres[6])
#

        res = minimize(erreurParams, params_init, method='BFGS',options={'gtol': 1e-4, 'disp': False})

        params[i,:] = res.x
    print('Rectifying face: '+str(100)+' %')
    return params


# Fonction qui remet un visage droit
def visDroit(data_visage,angles,indice_nez,pts_kept_no_contour,pts_kept_vis_fixes,sav_window,sav_order):

    # vecteur normal au plan de l'image :
    v = np.array([0,1,0])

    # vecteur normal dans le plan du visage
    v_vis = np.zeros((angles.shape[0],3))
    for i in range(angles.shape[0]):
        v_vis[i,:] = rotateMatrix(rotateMatrix(rotateMatrix(v,angles[i,2],'z'),angles[i,1],'y'),angles[i,0],'x')

    profondeur_visage = np.zeros((data_visage.shape[0],data_visage.shape[1]))
    boutdunez_xz = data_visage[:,indice_nez,:]


    data_visage[:,:,0] = data_visage[:,:,0] -  np.outer(boutdunez_xz[:,0], np.ones((1, data_visage.shape[1])))
    data_visage[:,:,1] = data_visage[:,:,1] -  np.outer(boutdunez_xz[:,1], np.ones((1, data_visage.shape[1])))
    for i in range(data_visage.shape[1]):
        profondeur_visage[:,i] = -np.divide((np.multiply(v_vis[:,0],data_visage[:,i,0])+np.multiply(v_vis[:,2],data_visage[:,i,1])),v_vis[:,1])

    visage_aplati = np.zeros((data_visage.shape[0],data_visage.shape[1],3))
    visage_aplati[:,:,0] = data_visage[:,:,0]
    visage_aplati[:,:,1] = profondeur_visage[:,:]
    visage_aplati[:,:,2] = data_visage[:,:,1]

    visage_droit1 = np.zeros(visage_aplati.shape)
    for i in range(angles.shape[0]):
        for j in range(data_visage.shape[1]):
            visage_droit1[i,j,:] = rotateMatrix(rotateMatrix(rotateMatrix(visage_aplati[i,j,:],-angles[i,0],'x'),-angles[i,1],'y'),-angles[i,2],'z')

    visage_droit_reduit = visage_droit1[:,pts_kept_no_contour,:]

    var_moy_z = np.std(visage_droit_reduit[:,:,2])
    visage_droit_reduit = visage_droit_reduit/var_moy_z
    visage_droit2 = visage_droit1/var_moy_z

    visage_dr_optim = visage_droit2[:,pts_kept_vis_fixes[0,:],:]
    visage_dr_optim_moyen = np.mean(visage_dr_optim,axis = 0)

    visage_dr_optimXZ = visage_dr_optim[:,:,[0,2]]
    visage_dr_optim_moyenXZ = visage_dr_optim_moyen[:,[0,2]]

    p = paramsTransfo(visage_dr_optimXZ,visage_dr_optim_moyenXZ)

    visage_dr_XZ_corr = np.zeros((visage_droit_reduit.shape[0],visage_droit_reduit.shape[1],2))
    for i in range(visage_dr_XZ_corr.shape[0]):
        for j in range(visage_dr_XZ_corr.shape[1]):
            visage_dr_XZ_corr[i,j,:] = transfoVisage(visage_droit_reduit[i,j,[0,2]],p[i,0],p[i,1],p[i,2],p[i,3],p[i,4],p[i,5],p[i,6])

    for i in range(p.shape[1]):
        p[:,i] = signal.savgol_filter(p[:,i], sav_window, sav_order)


    visage_dr_XZ_corr_filt = np.zeros(visage_dr_XZ_corr.shape)
#    for i in range(visage_dr_XZ_corr_filt.shape[0]):
#        for j in range(visage_dr_XZ_corr_filt.shape[1]):
#            visage_dr_XZ_corr_filt[i,j,:] = transfoVisage(visage_droit_reduit[i,j,[0,2]],p[i,0],p[i,1],p[i,2],p[i,3],p[i,4],p[i,5],p[i,6])
    for i in range(visage_dr_XZ_corr_filt.shape[1]):
        for j in range(2):
            visage_dr_XZ_corr_filt[:,i,j] = signal.savgol_filter(visage_dr_XZ_corr[:,i,j], sav_window, sav_order)

    return visage_dr_XZ_corr_filt
