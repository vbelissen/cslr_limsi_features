# -*- coding: utf-8 -*-

# Programme qui lit les fichiers JSON de sortie d'openpose (avec mains et visage),
# et qui reconstruit image par image le film avec prise en compte de la confiance
# dans la reconnaissance des points

from __future__ import division

import numpy as np
import json
import matplotlib.pyplot as plt

from scipy import signal
from scipy.misc import imread
from scipy.interpolate import interp1d


plt.ion()

plt.style.use('classic')


# Le type de Pose (COCO ou MPI) définit un nombre de points
def nbPtsPose(typePose):
    if typePose == 'COCO':
        return 18
    elif typePose == 'MPI':
        return 15
    else:
        return 0




# Fonction qui permet de créer un dégradé linéaire de couleur sur n valeurs
def echelle(color_begin, color_end, n_vals):
    r1, g1, b1 = color_begin
    r2, g2, b2 = color_end
    degrade = []
    etendue = n_vals - 1
    for i in range(n_vals):
        alpha = 1 - i / etendue
        beta = i / etendue
        r = int(r1 * alpha + r2 * beta)
        g = int(g1 * alpha + g2 * beta)
        b = int(b1 * alpha + b2 * beta)
        degrade.append((r, g, b))
    return degrade


# Creation des couleurs du corps, des mains et du visage
def colors(typePose):
    colorFace = 'white'

    rouge = (255, 0, 0)
    vert = (0, 255, 0)
    bleu = (0, 0, 255)

    nPP = nbPtsPose(typePose)

    colorPose1 = echelle(rouge,vert,int(nPP/3))
    colorPose2 = echelle(vert,bleu,int(nPP/3))
    colorPose3 = echelle(bleu,rouge,int(nPP/3))

    #Tableau de 18 couleurs pour la pose qui passe par le rouge, vert, bleu, puis rouge
    colorPose = np.array([colorPose1[0:int(nPP/3)],colorPose2[0:int(nPP/3)],colorPose3[0:int(nPP/3)]]).ravel().reshape(-1,3)
    colorPose = colorPose/255

    #Tableau de couleurs pour les mains
    colorHand = np.array(['black'])
    for i in range(1,5):
        colorHand= np.append(colorHand,'red')
    for i in range(5,9):
        colorHand= np.append(colorHand,'yellow')
    for i in range(9,13):
        colorHand= np.append(colorHand,'green')
    for i in range(13,17):
        colorHand= np.append(colorHand,'blue')
    for i in range(17,21):
        colorHand= np.append(colorHand,'magenta')

    return colorPose, colorFace, colorHand


# Tableau qui donne tous les noeuds liés
def tabLien(typePose):

    tabLienPose = np.array([]).astype(int)

    if typePose == 'COCO':
        for i in range(0,4):
            tabLienPose = np.append(tabLienPose,[i,i+1])
        tabLienPose = np.append(tabLienPose,[1,5])
        for i in range(5,7):
            tabLienPose = np.append(tabLienPose,[i,i+1])
        tabLienPose = np.append(tabLienPose,[1,8])
        for i in range(8,10):
            tabLienPose = np.append(tabLienPose,[i,i+1])
        tabLienPose = np.append(tabLienPose,[1,11])
        for i in range(11,13):
            tabLienPose = np.append(tabLienPose,[i,i+1])
        tabLienPose = np.append(tabLienPose,[0,14,14,16,0,15,15,17])
    elif typePose == 'MPI':
        for i in range(0,4):
            tabLienPose = np.append(tabLienPose,[i,i+1])

        for i in range(5,7):
            tabLienPose = np.append(tabLienPose,[i,i+1])

        for i in range(8,10):
            tabLienPose = np.append(tabLienPose,[i,i+1])

        for i in range(11,13):
            tabLienPose = np.append(tabLienPose,[i,i+1])
        tabLienPose = np.append(tabLienPose,[1,5,1,14,8,14,11,14])

    tabLienPose = tabLienPose.reshape(-1,2)

    tabLienFace = np.array([]).astype(int)
    for i in range(0,16):
        tabLienFace = np.append(tabLienFace,[i,i+1])
    for i in range(17,21):
        tabLienFace = np.append(tabLienFace,[i,i+1])
    for i in range(22,26):
        tabLienFace = np.append(tabLienFace,[i,i+1])
    for i in range(27,30):
        tabLienFace = np.append(tabLienFace,[i,i+1])
    for i in range(31,35):
        tabLienFace = np.append(tabLienFace,[i,i+1])
    for i in range(36,41):
        tabLienFace = np.append(tabLienFace,[i,i+1])
    tabLienFace = np.append(tabLienFace,[41,36])
    for i in range(42,47):
        tabLienFace = np.append(tabLienFace,[i,i+1])
    tabLienFace = np.append(tabLienFace,[47,42])
    for i in range(48,59):
        tabLienFace = np.append(tabLienFace,[i,i+1])
    tabLienFace = np.append(tabLienFace,[59,48])
    for i in range(60,67):
        tabLienFace = np.append(tabLienFace,[i,i+1])
    tabLienFace = np.append(tabLienFace,[67,60])

    tabLienFace = tabLienFace.reshape(-1,2)

    tabLienHand = np.array([]).astype(int)
    for i in range(0,4):
        tabLienHand = np.append(tabLienHand,[i,i+1])
    tabLienHand = np.append(tabLienHand,[0,5,0,9,0,13,0,17])
    for i in range(5,8):
        tabLienHand = np.append(tabLienHand,[i,i+1])
    for i in range(9,12):
        tabLienHand = np.append(tabLienHand,[i,i+1])
    for i in range(13,16):
        tabLienHand = np.append(tabLienHand,[i,i+1])
    for i in range(17,20):
        tabLienHand = np.append(tabLienHand,[i,i+1])

    tabLienHand = tabLienHand.reshape(-1,2)

    return tabLienPose, tabLienFace, tabLienHand



# Fonction pour gérer les nan
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]



#Fonction qui permet d'afficher ou non les points et les liens proportionnellement à la confiance
def coeffConf(choix, coeff):
    if  choix:
        return np.power(coeff,0.5)
    else:
        return 1


# Fonction de lecture des données de n fichiers JSON, et qui suit une personne
# censée ne pas quitter la caméra (indice_personne sur la première image)
#
# Inputs:
# n : numéro de la dernière image
# indice_personne : indice de la personne qu'on veut suivre sur la 1ere image
# s1 : chaine de caractères avant le numéro de l'image
# chiffres : nombre de chiffres couverts par le nombre des images
# s2 : chaine de caractères après le numéro de l'image
# Exemple : s1 = '/people/belissen/openpose/output/testDictasign/DictaSign_lsf_S1_T3_2view_00000000'
#           chiffres = 4
#           s2 = '_keypoints.json'
# typeData: 'pfh', 'pf', 'ph', 'p' (pose+face+hands, pose+face, pose+hands, pose)
# typePose : 'COCO' ou 'MPI'
#
# output: 4 tableaux a,b,c,d

def dataReadTabPoseFaceHandLR(n,indice_personne,s1,chiffres,s2,typeData,typePose):

    nPP = nbPtsPose(typePose)

    a=np.zeros((n,nPP,3))
    b=np.zeros((n,70,3))
    c=np.zeros((n,21,3))
    d=np.zeros((n,21,3))

    # Boucle sur l'ensemble des JSON (un par frame)
    for j in range(n):

        # Sur l'image 0 on suppose que la bonne personne est à l'indice
        # indice_personne
        # A partir de l'image 1, il faut rechercher parmi les personnes présentes
        # celle qui est la plus proche de celle de l'image précédente
        if j==0:
            # Importation des donnees de la 1ere image
            json_data = open(s1+str(j).zfill(chiffres)+s2)
            data = json.load(json_data)

            # Extraction visage, corps, mains
            a[j]=np.array(data['people'][indice_personne]['pose_keypoints']).ravel().reshape(-1,3)
            if typeData == 'pfh' or typeData == 'pf':
                b[j]=np.array(data['people'][indice_personne]['face_keypoints']).ravel().reshape(-1,3)
            if typeData == 'pfh' or typeData == 'ph':
                c[j]=np.array(data['people'][indice_personne]['hand_left_keypoints']).ravel().reshape(-1,3)
                d[j]=np.array(data['people'][indice_personne]['hand_right_keypoints']).ravel().reshape(-1,3)
        else:
            # Importation des donnees sur une image
            json_data_current=open(s1+str(j).zfill(chiffres)+s2)
            data_current = json.load(json_data_current)

            nb_personnes = len(data_current['people'])
            if nb_personnes>1:
                distance_moy = np.zeros(nb_personnes)
                for i in range(nb_personnes):
                    a_i = np.array(data_current['people'][i]['pose_keypoints']).ravel().reshape(-1,3)
                    if typeData == 'pfh' or typeData == 'pf':
                        b_i = np.array(data_current['people'][i]['face_keypoints']).ravel().reshape(-1,3)
                    if typeData == 'pfh' or typeData == 'ph':
                        c_i = np.array(data_current['people'][i]['hand_left_keypoints']).ravel().reshape(-1,3)
                        d_i = np.array(data_current['people'][i]['hand_right_keypoints']).ravel().reshape(-1,3)
                    distance_moy[i] = np.sum(np.abs(a[j-1,:,0:2]-a_i[:,0:2]))
                    if typeData == 'pfh':
                        distance_moy[i] += np.sum(np.abs(b[j-1,:,0:2]-b_i[:,0:2]))+np.sum(np.abs(c[j-1,:,0:2]-c_i[:,0:2]))+np.sum(np.abs(d[j-1,:,0:2]-d_i[:,0:2]))
                    elif typeData == 'pf':
                        distance_moy[i] += np.sum(np.abs(b[j-1,:,0:2]-b_i[:,0:2]))
                    elif typeData == 'ph':
                        distance_moy[i] += np.sum(np.abs(c[j-1,:,0:2]-c_i[:,0:2]))+np.sum(np.abs(d[j-1,:,0:2]-d_i[:,0:2]))
                ind_distance_min = np.argmin(distance_moy)
                a[j]=np.array(data_current['people'][ind_distance_min]['pose_keypoints']).ravel().reshape(-1,3)
                if typeData == 'pfh' or typeData == 'pf':
                    b[j]=np.array(data_current['people'][ind_distance_min]['face_keypoints']).ravel().reshape(-1,3)
                if typeData == 'pfh' or typeData == 'ph':
                    c[j]=np.array(data_current['people'][ind_distance_min]['hand_left_keypoints']).ravel().reshape(-1,3)
                    d[j]=np.array(data_current['people'][ind_distance_min]['hand_right_keypoints']).ravel().reshape(-1,3)
            else:
                a[j]=np.array(data_current['people'][0]['pose_keypoints']).ravel().reshape(-1,3)
                if typeData == 'pfh' or typeData == 'pf':
                    b[j]=np.array(data_current['people'][0]['face_keypoints']).ravel().reshape(-1,3)
                if typeData == 'pfh' or typeData == 'ph':
                    c[j]=np.array(data_current['people'][0]['hand_left_keypoints']).ravel().reshape(-1,3)
                    d[j]=np.array(data_current['people'][0]['hand_right_keypoints']).ravel().reshape(-1,3)

    return (a,b,c,d)

def dataReadTabPoseFaceHandLR_debut_fin(separ,n_deb,n_fin,indice_personne,s1,s11,chiffres,s2,typeData,typePose):

    n = n_fin-n_deb+1

    nPP = nbPtsPose(typePose)

    a=np.zeros((n,nPP,3))
    b=np.zeros((n,70,3))
    c=np.zeros((n,21,3))
    d=np.zeros((n,21,3))


    # Boucle sur l'ensemble des JSON (un par frame)
    for j in range(n):

        # Sur l'image 0 on suppose que la bonne personne est à l'indice
        # indice_personne
        # A partir de l'image 1, il faut rechercher parmi les personnes présentes
        # celle qui est la plus proche de celle de l'image précédente
        if j==0:
            # Importation des donnees de la 1ere image
            json_data = open(s1+str(j+n_deb).zfill(chiffres)+s2)
            data = json.load(json_data)

            if typeData == 'pfh' or typeData == 'ph':
                json_data_hands = open(s11+str(j+n_deb).zfill(chiffres)+s2)
                data_hands = json.load(json_data_hands)

            # Extraction visage, corps, mains
            a[j]=np.array(data['people'][indice_personne]['pose_keypoints']).ravel().reshape(-1,3)
            if typeData == 'pfh' or typeData == 'pf':
                b[j]=np.array(data['people'][indice_personne]['face_keypoints']).ravel().reshape(-1,3)
            if typeData == 'pfh' or typeData == 'ph':
                if separ:
                    c[j]=np.array(data_hands['people'][indice_personne]['hand_left_keypoints']).ravel().reshape(-1,3)
                    d[j]=np.array(data_hands['people'][indice_personne]['hand_right_keypoints']).ravel().reshape(-1,3)
                else:
                    c[j]=np.array(data['people'][indice_personne]['hand_left_keypoints']).ravel().reshape(-1,3)
                    d[j]=np.array(data['people'][indice_personne]['hand_right_keypoints']).ravel().reshape(-1,3)
        else:
            # Importation des donnees sur une image
            json_data_current=open(s1+str(j+n_deb).zfill(chiffres)+s2)
            data_current = json.load(json_data_current)

            if typeData == 'pfh' or typeData == 'ph':
                json_data_hands_current = open(s11+str(j+n_deb).zfill(chiffres)+s2)
                data_hands_current = json.load(json_data_hands_current)

            nb_personnes = len(data_current['people'])
            if nb_personnes>1:
                distance_moy = np.zeros(nb_personnes)
                for i in range(nb_personnes):
                    a_i = np.array(data_current['people'][i]['pose_keypoints']).ravel().reshape(-1,3)
                    if typeData == 'pfh' or typeData == 'pf':
                        b_i = np.array(data_current['people'][i]['face_keypoints']).ravel().reshape(-1,3)
                    if typeData == 'pfh' or typeData == 'ph':
                        if separ:
                            c_i = np.array(data_hands_current['people'][i]['hand_left_keypoints']).ravel().reshape(-1,3)
                            d_i = np.array(data_hands_current['people'][i]['hand_right_keypoints']).ravel().reshape(-1,3)
                        else:
                            c_i = np.array(data_current['people'][i]['hand_left_keypoints']).ravel().reshape(-1,3)
                            d_i = np.array(data_current['people'][i]['hand_right_keypoints']).ravel().reshape(-1,3)
                    distance_moy[i] = np.sum(np.abs(a[j-1,:,0:2]-a_i[:,0:2]))
                    if typeData == 'pfh':
                        distance_moy[i] += np.sum(np.abs(b[j-1,:,0:2]-b_i[:,0:2]))+np.sum(np.abs(c[j-1,:,0:2]-c_i[:,0:2]))+np.sum(np.abs(d[j-1,:,0:2]-d_i[:,0:2]))
                    elif typeData == 'pf':
                        distance_moy[i] += np.sum(np.abs(b[j-1,:,0:2]-b_i[:,0:2]))
                    elif typeData == 'ph':
                        distance_moy[i] += np.sum(np.abs(c[j-1,:,0:2]-c_i[:,0:2]))+np.sum(np.abs(d[j-1,:,0:2]-d_i[:,0:2]))
                ind_distance_min = np.argmin(distance_moy)
                a[j]=np.array(data_current['people'][ind_distance_min]['pose_keypoints']).ravel().reshape(-1,3)
                if typeData == 'pfh' or typeData == 'pf':
                    b[j]=np.array(data_current['people'][ind_distance_min]['face_keypoints']).ravel().reshape(-1,3)
                if typeData == 'pfh' or typeData == 'ph':
                    if separ:
                        c[j]=np.array(data_hands_current['people'][ind_distance_min]['hand_left_keypoints']).ravel().reshape(-1,3)
                        d[j]=np.array(data_hands_current['people'][ind_distance_min]['hand_right_keypoints']).ravel().reshape(-1,3)
                    else:
                        c[j]=np.array(data_current['people'][ind_distance_min]['hand_left_keypoints']).ravel().reshape(-1,3)
                        d[j]=np.array(data_current['people'][ind_distance_min]['hand_right_keypoints']).ravel().reshape(-1,3)
            else:
                a[j]=np.array(data_current['people'][0]['pose_keypoints']).ravel().reshape(-1,3)
                if typeData == 'pfh' or typeData == 'pf':
                    b[j]=np.array(data_current['people'][0]['face_keypoints']).ravel().reshape(-1,3)
                if typeData == 'pfh' or typeData == 'ph':
                    if separ:
                        c[j]=np.array(data_hands_current['people'][0]['hand_left_keypoints']).ravel().reshape(-1,3)
                        d[j]=np.array(data_hands_current['people'][0]['hand_right_keypoints']).ravel().reshape(-1,3)
                    else:
                        c[j]=np.array(data_current['people'][0]['hand_left_keypoints']).ravel().reshape(-1,3)
                        d[j]=np.array(data_current['people'][0]['hand_right_keypoints']).ravel().reshape(-1,3)

    return (a,b,c,d)



# Fonction qui affiche les courbes a, b, c et d en fonction du temps, et
# la confiance associée à chacun des points
def plot_XYConf_PoseFaceHandLR(a,b,c,d,n,Ximg,Yimg,fps,typeData,typePose):

    colorPose, colorFace, colorHand = colors(typePose)

    if typeData == 'pfh':
        k = 4
    elif typeData == 'pf':
        k = 2
    elif typeData == 'ph':
        k = 3
    elif typeData == 'p':
        k = 1

    deltaT = 1/fps
    plt.figure(figsize=(4.5*k,8))
    for i in range(nbPtsPose(typePose)):
        plt.subplot(3,k,1)
        plt.xlim(0,n*deltaT)
        plt.ylim(0,Ximg)
        plt.title('Pose, X')
        plt.plot(np.asarray(range(n))*deltaT,a[:,i,0],color=colorPose[i,:],label=i)
        plt.legend(bbox_to_anchor=(-0.25, 1), loc=2, borderaxespad=0., prop={'size': 8})

    for i in range(nbPtsPose(typePose)):
        plt.subplot(3,k,1+k)
        plt.xlim(0,n*deltaT)
        plt.ylim(0,Yimg)
        plt.title('Pose, Y')
        plt.gca().invert_yaxis()
        plt.plot(np.asarray(range(n))*deltaT,a[:,i,1],color=colorPose[i,:])

    for i in range(nbPtsPose(typePose)):
        plt.subplot(3,k,1+2*k)
        plt.xlim(0,n*deltaT)
        plt.ylim(0,1)
        plt.title('Pose, confidence')
        plt.plot(np.asarray(range(n))*deltaT,a[:,i,2],color=colorPose[i,:])

    if typeData == 'pfh' or typeData == 'pf':
        for i in range(70):
            plt.subplot(3,k,2)
            plt.xlim(0,n*deltaT)
            plt.ylim(0,Ximg)
            plt.title('Face, X')
            plt.plot(np.asarray(range(n))*deltaT,b[:,i,0],color='black')

        for i in range(70):
            plt.subplot(3,k,2+k)
            plt.xlim(0,n*deltaT)
            plt.ylim(0,Yimg)
            plt.title('Face, Y')
            plt.gca().invert_yaxis()
            plt.plot(np.asarray(range(n))*deltaT,b[:,i,1],color='black')

        for i in range(70):
            plt.subplot(3,k,2+2*k)
            plt.xlim(0,n*deltaT)
            plt.ylim(0,1)
            plt.title('Face, confidence')
            plt.plot(np.asarray(range(n))*deltaT,b[:,i,2],color='black')

    if typeData == 'pfh' or typeData == 'ph':
        for i in range(21):
            plt.subplot(3,k,k-1)
            plt.xlim(0,n*deltaT)
            plt.ylim(0,Ximg)
            plt.title('Left Hand, X')
            plt.plot(np.asarray(range(n))*deltaT,c[:,i,0],color=colorHand[i])

        for i in range(21):
            plt.subplot(3,k,k-1+k)
            plt.xlim(0,n*deltaT)
            plt.ylim(0,Yimg)
            plt.title('Left Hand, Y')
            plt.gca().invert_yaxis()
            plt.plot(np.asarray(range(n))*deltaT,c[:,i,1],color=colorHand[i])

        for i in range(21):
            plt.subplot(3,k,k-1+2*k)
            plt.xlim(0,n*deltaT)
            plt.ylim(0,1)
            plt.title('Left Hand, confidence')
            plt.plot(np.asarray(range(n))*deltaT,c[:,i,2],color=colorHand[i])

        for i in range(21):
            plt.subplot(3,k,k)
            plt.xlim(0,n*deltaT)
            plt.ylim(0,Ximg)
            plt.title('Right Hand, X')
            plt.plot(np.asarray(range(n))*deltaT,d[:,i,0],color=colorHand[i],label=i)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 8})

        for i in range(21):
            plt.subplot(3,k,k+k)
            plt.xlim(0,n*deltaT)
            plt.ylim(0,Yimg)
            plt.title('Right Hand, Y')
            plt.gca().invert_yaxis()
            plt.plot(np.asarray(range(n))*deltaT,d[:,i,1],color=colorHand[i])

        for i in range(21):
            plt.subplot(3,k,k+2*k)
            plt.xlim(0,n*deltaT)
            plt.ylim(0,1)
            plt.title('Right Hand, confidence')
            plt.plot(np.asarray(range(n))*deltaT,d[:,i,2],color=colorHand[i])



# Fonction qui met à 0 les points dont la confiance moyenne est inférieure à un seuil
# Permet d'enlever des points qui ne sont détectés que très rarement
def nettoyageValMoy(a,b,c,d,confMoy,typeData):

    a1 = np.array(a)
    b1 = np.array(b)
    c1 = np.array(c)
    d1 = np.array(d)

    confMoyPose = np.average(a[:,:,2],axis=0)
    if typeData == 'pfh' or typeData == 'pf':
        confMoyFace = np.average(b[:,:,2],axis=0)
    if typeData == 'pfh' or typeData == 'ph':
        confMoyHandL = np.average(c[:,:,2],axis=0)
        confMoyHandR = np.average(d[:,:,2],axis=0)

    low_values_flags_Pose = confMoyPose < confMoy
    a1[:,low_values_flags_Pose,0] = 0
    a1[:,low_values_flags_Pose,1] = 0

    if typeData == 'pfh' or typeData == 'pf':
        low_values_flags_Face = confMoyFace < confMoy
        b1[:,low_values_flags_Face,0] = 0
        b1[:,low_values_flags_Face,1] = 0

    if typeData == 'pfh' or typeData == 'ph':
        low_values_flags_HandL = confMoyHandL < confMoy
        c1[:,low_values_flags_HandL,0] = 0
        c1[:,low_values_flags_HandL,1] = 0

        low_values_flags_HandR = confMoyHandR < confMoy
        d1[:,low_values_flags_HandR,0] = 0
        d1[:,low_values_flags_HandR,1] = 0

    return(a1,b1,c1,d1)



# Fonction qui met à 0 les points dont la confiance est inférieure à un seuil
# Permet de ne pas prendre en compte des points incertains
def nettoyageVal(a,b,c,d,confMinPose,confMinFace,confMinHand,typeData):

    a1 = np.array(a)
    b1 = np.array(b)
    c1 = np.array(c)
    d1 = np.array(d)

    low_values_flags_PoseMin = a[:,:,2] < confMinPose
    low_values_flags_PoseMin[:2,:] =False
    low_values_flags_PoseMin[-2:,:]=False
    a1[low_values_flags_PoseMin,0] = 0
    a1[low_values_flags_PoseMin,1] = 0

    if typeData == 'pfh' or typeData == 'pf':
        low_values_flags_FaceMin = b[:,:,2] < confMinFace
        low_values_flags_FaceMin[:2,:] =False
        low_values_flags_FaceMin[-2:,:]=False
        b1[low_values_flags_FaceMin,0] = 0
        b1[low_values_flags_FaceMin,1] = 0

    if typeData == 'pfh' or typeData == 'ph':
        low_values_flags_HandLMin = c[:,:,2] < confMinHand
        low_values_flags_HandLMin[:2,:] =False
        low_values_flags_HandLMin[-2:,:]=False
        c1[low_values_flags_HandLMin,0] = 0
        c1[low_values_flags_HandLMin,1] = 0

        low_values_flags_HandRMin = d[:,:,2] < confMinHand
        low_values_flags_HandRMin[:2,:] =False
        low_values_flags_HandRMin[-2:,:]=False
        d1[low_values_flags_HandRMin,0] = 0
        d1[low_values_flags_HandRMin,1] = 0

    return(a1,b1,c1,d1)


#Transformation des 0 en Nan (sauf pour la confidence)
def zeroToNan(a,b,c,d,typeData):

    a1 = np.array(a)
    b1 = np.array(b)
    c1 = np.array(c)
    d1 = np.array(d)

    aTestNan = (a==0)
    aTestNan[:,:,2]=False
    a1[aTestNan] = np.nan
    if typeData == 'pfh' or typeData == 'pf':
        bTestNan = (b==0)
        bTestNan[:,:,2]=False
        b1[bTestNan] = np.nan
    if typeData == 'pfh' or typeData == 'ph':
        cTestNan = (c==0)
        cTestNan[:,:,2]=False
        c1[cTestNan] = np.nan
        dTestNan = (d==0)
        dTestNan[:,:,2]=False
        d1[dTestNan] = np.nan

    return(a1,b1,c1,d1)

def nettoyageComplet(a,b,c,d,confMoy,confMinPose,confMinFace,confMinHand,typeData):

    a1 = np.array(a)
    b1 = np.array(b)
    c1 = np.array(c)
    d1 = np.array(d)

    (a1,b1,c1,d1) = nettoyageValMoy(a1,b1,c1,d1,confMoy,typeData)
    (a1,b1,c1,d1) = nettoyageVal(a1,b1,c1,d1,confMinPose,confMinFace,confMinHand,typeData)
    (a1,b1,c1,d1) = zeroToNan(a1,b1,c1,d1,typeData)

    return(a1,b1,c1,d1)


# Fonction d'interpolation sur les valeurs nan

#Interpolation sur les valeurs Nan
def interpNan(a,b,c,d,n,typeData,typePose):

    a1 = np.array(a)
    b1 = np.array(b)
    c1 = np.array(c)
    d1 = np.array(d)

    nPP = nbPtsPose(typePose)

    for i in range(nPP):
        nans, x = nan_helper(a1[:,i,0])
        if a1[~nans,i,0].size != 0:
            f2 = interp1d(x(~nans), a1[~nans,i,0], kind='linear',bounds_error=False)
            a1[:,i,0]= f2(range(n))

        nans, x = nan_helper(a1[:,i,1])
        if a1[~nans,i,1].size != 0:
            f2 = interp1d(x(~nans), a1[~nans,i,1], kind='linear',bounds_error=False)
            a1[:,i,1]= f2(range(n))

    if typeData == 'pfh' or typeData == 'pf':
        for i in range(70):
            nans, x = nan_helper(b1[:,i,0])
            if b1[~nans,i,0].size != 0:
                f2 = interp1d(x(~nans), b1[~nans,i,0], kind='linear',bounds_error=False)
                b1[:,i,0]= f2(range(n))

            nans, x = nan_helper(b1[:,i,1])
            if b1[~nans,i,1].size != 0:
                f2 = interp1d(x(~nans), b1[~nans,i,1], kind='linear',bounds_error=False)
                b1[:,i,1]= f2(range(n))


    if typeData == 'pfh' or typeData == 'ph':
        for i in range(21):
            nans, x = nan_helper(c1[:,i,0])
            if c1[~nans,i,0].size != 0:
                f2 = interp1d(x(~nans),c1[~nans,i,0], kind='linear',bounds_error=False)
                c1[:,i,0]= f2(range(n))

            nans, x = nan_helper(c1[:,i,1])
            if c1[~nans,i,1].size != 0:
                f2 = interp1d(x(~nans),c1[~nans,i,1], kind='linear',bounds_error=False)
                c1[:,i,1]= f2(range(n))

            nans, x = nan_helper(d1[:,i,0])
            if d1[~nans,i,0].size != 0:
                f2 = interp1d(x(~nans), d1[~nans,i,0], kind='linear',bounds_error=False)
                d1[:,i,0]= f2(range(n))

            nans, x = nan_helper(d1[:,i,1])
            if d1[~nans,i,1].size != 0:
                f2 = interp1d(x(~nans), d1[~nans,i,1], kind='linear',bounds_error=False)
                d1[:,i,1]= f2(range(n))

    return(a1,b1,c1,d1)


# Fonction de filtrage de Savitzky-Golay
def filtrageSavGol(a,b,c,d,savitzky_window,savitzky_order,typeData,typePose):

    a1 = np.array(a)
    b1 = np.array(b)
    c1 = np.array(c)
    d1 = np.array(d)

    nPP = nbPtsPose(typePose)

    for i in range(nPP):
        a1[:,i,0] = signal.savgol_filter(a1[:,i,0], savitzky_window, savitzky_order)
        a1[:,i,1] = signal.savgol_filter(a1[:,i,1], savitzky_window, savitzky_order)

    if typeData == 'pfh' or typeData == 'pf':
        for i in range(70):
            b1[:,i,0] = signal.savgol_filter(b1[:,i,0], savitzky_window, savitzky_order)
            b1[:,i,1] = signal.savgol_filter(b1[:,i,1], savitzky_window, savitzky_order)

    if typeData == 'pfh' or typeData == 'ph':
        for i in range(21):
            c1[:,i,0] = signal.savgol_filter(c1[:,i,0], savitzky_window, savitzky_order)
            c1[:,i,1] = signal.savgol_filter(c1[:,i,1], savitzky_window, savitzky_order)
            d1[:,i,0] = signal.savgol_filter(d1[:,i,0], savitzky_window, savitzky_order)
            d1[:,i,1] = signal.savgol_filter(d1[:,i,1], savitzky_window, savitzky_order)

    return(a1,b1,c1,d1)


# Fonction qui met à nan les points qui sont en dessous de la confiance min plus de K fois de suite
def effaceZeroKconsecutif(a,b,c,d,n,K,confMinPose,confMinFace,confMinHand,typeData,typePose):

    a1 = np.array(a)
    b1 = np.array(b)
    c1 = np.array(c)
    d1 = np.array(d)

    nPP = nbPtsPose(typePose)

    low_values_flags_PoseMin = a1[:,:,2] < confMinPose
    if typeData == 'pfh' or typeData == 'pf':
        low_values_flags_FaceMin = b1[:,:,2] < confMinFace
    if typeData == 'pfh' or typeData == 'ph':
        low_values_flags_HandLMin = c1[:,:,2] < confMinHand
        low_values_flags_HandRMin = d1[:,:,2] < confMinHand

    for i in range(nPP):
        tmp = 0
        for k in range(n-K,-1,-1):
            if low_values_flags_PoseMin[k,i]:
                tmp += 1
            if k>0:
                if low_values_flags_PoseMin[k,i] and ~low_values_flags_PoseMin[k-1,i]:
                    if tmp > K:
                        a1[k:k+tmp,i,0:2]=np.nan
                    tmp =0
            elif tmp > K:
                a1[k:k+tmp,i,0]=np.nan

    if typeData == 'pfh' or typeData == 'pf':
        for i in range(70):
            tmp = 0
            for k in range(n-K,-1,-1):
                if low_values_flags_FaceMin[k,i]:
                    tmp += 1
                if k>0:
                    if low_values_flags_FaceMin[k,i] and ~low_values_flags_FaceMin[k-1,i]:
                        if tmp > K:
                            b1[k:k+tmp,i,0:2]=np.nan
                        tmp =0
                elif tmp > K:
                    b1[k:k+tmp,i,0:2]=np.nan

    if typeData == 'pfh' or typeData == 'ph':
        for i in range(21):
            tmp = 0
            for k in range(n-K,-1,-1):
                if low_values_flags_HandLMin[k,i]:
                    tmp += 1
                if k>0:
                    if low_values_flags_HandLMin[k,i] and ~low_values_flags_HandLMin[k-1,i]:
                        if tmp > K:
                            c1[k:k+tmp,i,0:2]=np.nan
                        tmp =0
                elif tmp > K:
                    c1[k:k+tmp,i,0:2]=np.nan

        for i in range(21):
            tmp = 0
            for k in range(n-K,-1,-1):
                if low_values_flags_HandRMin[k,i]:
                    tmp += 1
                if k>0:
                    if low_values_flags_HandRMin[k,i] and ~low_values_flags_HandRMin[k-1,i]:
                        if tmp > K:
                            d1[k:k+tmp,i,0:2]=np.nan
                        tmp =0
                elif tmp > K:
                    d1[k:k+tmp,i,0:2]=np.nan

    return(a1,b1,c1,d1)




# Fonction d'enregistrement et éventuellement d'affichage de chacune des images reconstruites
# Si affiche_fond == True, on rajoute la photo en fond
# Si affiche_direct, on fait une pause à chaque image pour pouvoir la voir
def savePlotFrames(a,b,c,d,n,Ximg,Yimg,choixCoeffConf,affiche_fond,dossierFond,chiffresFond,extensionFond,affiche_direct,dossierSave,chiffresSave,extensionSave,typeData,typePose,save):

    colorPose, colorFace, colorHand = colors(typePose)
    tabLienPose, tabLienFace, tabLienHand = tabLien(typePose)

    plt.style.use('dark_background')

    if affiche_direct:
        plt.ion()
    else:
        plt.ioff()

    plt.figure(figsize=(Ximg/Yimg*8,8))

    for j in range(n):
        # Min et max à redéfinir en fonction de la video
        plt.xlim(0,Ximg)
        plt.ylim(0,Yimg)
        plt.gca().invert_yaxis()

        #Iteration sur tous les liens existant
        for i in range(tabLienPose.shape[0]):
            ideb = tabLienPose[i,0]
            ifin = tabLienPose[i,1]
            #On ne crée un lien que si le point a ete detecte, c'est a dire
            #si ses coordonnees ne sont pas a 0

            if abs(a[j,ideb,0]*a[j,ifin,0]*a[j,ideb,1]*a[j,ifin,1])>0.01:
                #Plot des liens
    	    #La taille des liens est proportionnelle ou pas à la confiance moyenne des 2 points, selon le choix du début
    	    # On rajoute une petite valeur pour ne pas que la largeur soit exactement 0, sinon erreur
                plt.plot([a[j,ideb,0],a[j,ifin,0]],[a[j,ideb,1],a[j,ifin,1]], lw = 3*0.5*coeffConf(choixCoeffConf,(a[j,ideb,2]+a[j,ifin,2]+0.02)), color = 0.5*(colorPose[ideb]+colorPose[ifin]))

        if typeData == 'pfh' or typeData == 'pf':
            for i in range(tabLienFace.shape[0]):
                ideb = tabLienFace[i,0]
                ifin = tabLienFace[i,1]
                #On ne crée un lien que si le point a ete detecte, c'est a dire
                #si ses coordonnees ne sont pas a 0
                if abs(b[j,ideb,0]*b[j,ifin,0]*b[j,ideb,1]*b[j,ifin,1])>0.01:
                    #Plot des liens
                    plt.plot([b[j,ideb,0],b[j,ifin,0]],[b[j,ideb,1],b[j,ifin,1]], lw = 2*0.5*coeffConf(choixCoeffConf,(b[j,ideb,2]+b[j,ifin,2]+0.02)), color = 'white')

        if typeData == 'pfh' or typeData == 'ph':
            for i in range(tabLienHand.shape[0]):
                ideb = tabLienHand[i,0]
                ifin = tabLienHand[i,1]
                #On ne crée un lien que si le point a ete detecte, c'est a dire
                #si ses coordonnees ne sont pas a 0
                if abs(c[j,ideb,0]*c[j,ifin,0]*c[j,ideb,1]*c[j,ifin,1])>0.01:
                    #Plot des liens
                    plt.plot([c[j,ideb,0],c[j,ifin,0]],[c[j,ideb,1],c[j,ifin,1]], lw = 3*0.5*coeffConf(choixCoeffConf,(c[j,ideb,2]+c[j,ifin,2]+0.02)), color = colorHand[ifin])
                if abs(d[j,ideb,0]*d[j,ifin,0]*d[j,ideb,1]*d[j,ifin,1])>0.01:
                    #Plot des liens
                    plt.plot([d[j,ideb,0],d[j,ifin,0]],[d[j,ideb,1],d[j,ifin,1]], lw = 3*0.5*coeffConf(choixCoeffConf,(d[j,ideb,2]+d[j,ifin,2]+0.02)), color = colorHand[ifin])

        #Plot des points
        #Taille des points proportionnelle ou pas (selon le choix du début) à la confiance de ce point
        plt.scatter(a[j,:,0],a[j,:,1], s=100*coeffConf(choixCoeffConf,(a[j,:,2]+0.01)), facecolors = colorPose, edgecolor = 'none')
        if typeData == 'pfh' or typeData == 'pf':
            plt.scatter(b[j,:,0],b[j,:,1], s=3*coeffConf(choixCoeffConf,(b[j,:,2]+0.01)), facecolors = colorFace, edgecolor = 'none')
        if typeData == 'pfh' or typeData == 'ph':
            plt.scatter(c[j,:,0],c[j,:,1], s=30*coeffConf(choixCoeffConf,(c[j,:,2]+0.01)), facecolors = colorHand, edgecolor = 'none')
            plt.scatter(d[j,:,0],d[j,:,1], s=30*coeffConf(choixCoeffConf,(d[j,:,2]+0.01)), facecolors = colorHand, edgecolor = 'none')

        if affiche_fond:
            img = imread(dossierFond+'/'+str(j+1).zfill(chiffresFond)+'.'+extensionFond)
            plt.imshow(img,zorder=0)


        #Pause pour l'affichage, peut-etre à enlevr
        if affiche_direct:
            plt.pause(0.001)

        #Sauvegarde du plot
        if save:
            plt.savefig(dossierSave+'/'+str(j).zfill(chiffresSave)+'.'+extensionSave)
        print(str(int(100*j/float(n)))+' %')
        #Effaçage de l'image
        plt.clf()
    plt.ion()


# Fonction de transformation des donnees format openpose vers un format
# utilisable pour la reconstruction
# Il faut indiquer les indices des points a garder
# Il faut lui passer les donnees en excluant la confiance
def transfo_data_OP_recons(data,pts_kept):
    subset = 2*pts_kept
    tmp1 = sorted(list(subset)+list(subset+1))

    data1 = np.zeros((data.shape[0],len(tmp1)))
    ind = 0
    for j in tmp1:
	#print(j/2)
	#print(j//2)
        data1[:,ind] = data[:,j//2,j%2]
        ind = ind+1

    return data1


# Fonction qui renvoie les coordonnées de la main gauche et de la main droite
# Hypothèse d'une longueur de main d'environ 40% de la longueur de
# l'avant bras, et donc un milieu de la main à 20%
def centreMainGD(coudeG,poignetG,coudeD,poignetD):
    nT = coudeG.shape[0]
    dataMainGD = np.zeros((nT,2,2))

    dataMainGD[:,:,0] = coudeG + (poignetG-coudeG)*1.2
    dataMainGD[:,:,1] = coudeD + (poignetD-coudeD)*1.2

    return dataMainGD
