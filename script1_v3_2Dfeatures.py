import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib 
#from mpl_toolkits.mplot3d import Axes3D

import csv
import os, os.path

#import ctypes
#import sys

#import time

from sklearn import preprocessing

fps = 25.0
array_A = ['A8','A11','A2','A1','A9','A6','A10','A7','A3']
array_B = ['B12','B15','B0','B14','B17','B13','B16','B4','B5']


with open("annotations/transcripts_decalages.csv", "r") as open_transcripts_decalages:
    reader_transcripts_decalages = csv.reader(open_transcripts_decalages)#, delimiter='\t')
    transcripts_decalages = {}
    for i in reader_transcripts_decalages:
        transcripts_decalages[reader_transcripts_decalages.line_num] = i

with open("annotations/Movies.csv", "r") as open_videos:
    reader_videos = csv.reader(open_videos, delimiter='\t')
    videos = {}
    for i in reader_videos:
        videos[reader_videos.line_num] = i

with open("annotations/Transcripts.csv", "r") as open_transcripts:
    reader_transcripts = csv.reader(open_transcripts, delimiter='\t')
    transcripts = {}
    for i in reader_transcripts:
        transcripts[reader_transcripts.line_num] = i

with open("annotations/Tiers.csv", "r") as open_tiers:
    reader_tiers = csv.reader(open_tiers, delimiter='\t')
    tiers = {}
    for i in reader_tiers:
        tiers[reader_tiers.line_num] = i

with open("annotations/Tags2.csv", "r") as open_tags:
    reader_tags = csv.reader(open_tags, delimiter='\t')
    tags = {}
    for i in reader_tags:
        tags[reader_tags.line_num] = i

with open("annotations/Tokens.csv", "r") as open_tokens:
    reader_tokens = csv.reader(open_tokens, delimiter='\t')
    tokens = {}
    for i in reader_tokens:
        tokens[reader_tokens.line_num] = i

with open("annotations/Types.csv", "r") as open_types:
    reader_types = csv.reader(open_types, delimiter='\t')
    types = {}
    for i in reader_types:
        types[reader_types.line_num] = i





dataBrut_pfh = []

dataFace = []
dataPose = []
dataAnglesTete = []
dataAnglesTete2 = []

dataFace_vit = []
dataPose_vit = []
dataAnglesTete_vit = []
dataAnglesTete2_vit = []

dataFace_acc = []
dataPose_acc = []
dataAnglesTete_acc = []
dataAnglesTete2_acc = []

main1_relpt1 = []
main2_relpt1 = []
main1_relpt1_vit = []
main2_relpt1_vit = []
main1_relpt1_acc = []
main2_relpt1_acc = []
boucheGD_dist = []
boucheHB_dist = []
sourcil1_relHautNez_dist = []
sourcil2_relHautNez_dist = []
main1_relCoude = []
main2_relCoude = []
main1_relCoude_vit = []
main2_relCoude_vit = []
main1_relCoude_acc = []
main2_relCoude_acc = []
main1_relNez = []
main2_relNez = []
main1_relNez_vit = []
main2_relNez_vit = []
main1_relNez_acc = []
main2_relNez_acc = []
coude1_relEpaule = []
coude2_relEpaule = []
coude1_relEpaule_vit = []
coude2_relEpaule_vit = []
coude1_relEpaule_acc = []
coude2_relEpaule_acc = []
epaule1_relpt1 = []
epaule2_relpt1 = []
epaule1_relpt1_vit = []
epaule2_relpt1_vit = []
epaule1_relpt1_acc = []
epaule2_relpt1_acc = []
nez_relpt1 = []

main1_vit_norm = []
main2_vit_norm = []
main1_acc_norm = []
main2_acc_norm = []
main1_rel_Main2_vect = []
main1_rel_Main2_vit = []
main1_rel_Main2_acc = []
main1_rel_Main2_dist = []
#proba_Main1_pondere = []
#proba_Main2_pondere = []
cos_coude1 = []
cos_coude2 = []
cos_coude1_vit = []
cos_coude2_vit = []
cos_coude1_acc = []
cos_coude2_acc = []
ortho_coude1 = []
ortho_coude2 = []
ortho_coude1_vit = []
ortho_coude2_vit = []
ortho_coude1_acc = []
ortho_coude2_acc = []
cos_epaule1 = []
cos_epaule2 = []
cos_epaule1_vit = []
cos_epaule2_vit = []
cos_epaule1_acc = []
cos_epaule2_acc = []
ortho_epaule1 = []
ortho_epaule2 = []
ortho_epaule1_vit = []
ortho_epaule2_vit = []
ortho_epaule1_acc = []
ortho_epaule2_acc = []


for i_transcripts_decalages in range(2,len(transcripts_decalages)+1):
    S = int(transcripts_decalages[i_transcripts_decalages][0])
    T = int(transcripts_decalages[i_transcripts_decalages][1])
    for i_locut in range(2):
        if i_locut == 0:
            locut = array_A[S - 1]
            transcript = transcripts_decalages[i_transcripts_decalages][2]
        else:
            locut = array_B[S - 1]
            transcript = transcripts_decalages[i_transcripts_decalages][3]

        dir_to_test = '/people/belissen/Videos/DictaSign/convert/img_decal_full/DictaSign_lsf_S'+str(S)+'_T'+str(T)+'_'+locut+'_front_decal/'
        olddir = '/people/belissen/Videos/DictaSign/convert/img/DictaSign_lsf_S' + str(S) + '_T' + str(T) + '_' + locut + '_front/'

        if os.path.isdir(dir_to_test) and transcript != 'abs':
            print(dir_to_test)

            nb_img = len([name for name in os.listdir(dir_to_test) if os.path.isfile(os.path.join(dir_to_test, name))])

            retard = int(transcripts_decalages[i_transcripts_decalages][4 + i_locut])
            gaucher = int(transcripts_decalages[i_transcripts_decalages][6 + i_locut])



            data3D = np.load('/people/belissen/Videos/DictaSign/results/poses/3D_npy/S'+str(S)+'_task'+str(T)+'_'+locut+'.npz')
            xdata = data3D['X_test2'][:, 0::2]
            ydata = data3D['output2']
            zdata = data3D['X_test2'][:, 1::2]
            
            if gaucher == 1:
                xdata *= -1

            dataPoseTmp = np.zeros((ydata.shape[1], 3, ydata.shape[0]))
            dataPoseTmp[:, 0, :] = np.swapaxes(xdata, 0, 1)
            dataPoseTmp[:, 1, :] = np.swapaxes(ydata, 0, 1)
            dataPoseTmp[:, 2, :] = np.swapaxes(zdata, 0, 1)

            dataFaceTmp = data3D['tabTot']

            if gaucher == 1:
                dataFaceTmp[:, 0, :] *= -1

            larg_epaules_moy = np.mean(np.sqrt(np.square(dataPoseTmp[5, 0, :] - dataPoseTmp[2, 0, :]) + np.square(dataPoseTmp[5, 1, :] - dataPoseTmp[2, 1, :]) + np.square(dataPoseTmp[5, 2, :] - dataPoseTmp[2, 2, :])))

            # Pose relative au point 1
            dataPoseTempo1 = np.copy(dataPoseTmp)

            for i in range(dataPoseTmp.shape[0]):
                for j in range(dataPoseTmp.shape[2]):
                    dataPoseTmp[i, :, j] = (dataPoseTmp[i, :, j] - dataPoseTempo1[1, :, j]) * 2 / larg_epaules_moy

            for i in range(dataFaceTmp.shape[0]):
                for j in range(dataFaceTmp.shape[2]):
                    dataFaceTmp[i, :, j] = (dataFaceTmp[i, :, j] - dataPoseTempo1[1, :, j]) * 2 / larg_epaules_moy

            dataFace.append(dataFaceTmp)
            dataPose.append(dataPoseTmp)

            angles1tmp = data3D['anglesTete']
            angles2tmp = data3D['anglesTete2']
            if gaucher == 1:
                angles1tmp[:, 1] *= -1
                angles1tmp[:, 2] *= -1
                angles2tmp[:, 1] *= -1
                angles2tmp[:, 2] *= -1

            dataAnglesTete.append(angles1tmp[:,1])
            dataAnglesTete2.append(angles2tmp[:,1])
            
            


            dataFace_vit_tmp = np.zeros(dataFaceTmp.shape)
            dataFace_acc_tmp = np.zeros(dataFaceTmp.shape)
            dataPose_vit_tmp = np.zeros(dataPoseTmp.shape)
            dataPose_acc_tmp = np.zeros(dataPoseTmp.shape)
            dataAnglesTete_vit_tmp = np.zeros(data3D['anglesTete'].shape)
            dataAnglesTete2_vit_tmp = np.zeros(data3D['anglesTete2'].shape)
            dataAnglesTete_acc_tmp = np.zeros(data3D['anglesTete'].shape)
            dataAnglesTete2_acc_tmp = np.zeros(data3D['anglesTete2'].shape)

            for i in range(dataFaceTmp.shape[2]):
                if i == 0:
                    ideb = 0
                    ifin = 1
                else:
                    ideb = i - 1
                    ifin = i
                dataFace_vit_tmp[:, :, i] = dataFaceTmp[:, :, ifin] - dataFaceTmp[:, :, ideb]
                dataPose_vit_tmp[:, :, i] = dataPoseTmp[:, :, ifin] - dataPoseTmp[:, :, ideb]
                dataAnglesTete_vit_tmp[i, :] = data3D['anglesTete'][ifin, :] - data3D['anglesTete'][ideb, :]
                dataAnglesTete2_vit_tmp[i, :] = data3D['anglesTete2'][ifin, :] - data3D['anglesTete2'][ideb, :]

            dataFace_vit.append(dataFace_vit_tmp * 1 / fps)
            dataPose_vit.append(dataPose_vit_tmp * 1 / fps)
            dataAnglesTete_vit.append(dataAnglesTete_vit_tmp[:,1] * 1 / fps)
            dataAnglesTete2_vit.append(dataAnglesTete2_vit_tmp[:,1] * 1 / fps)

            for i in range(dataFaceTmp.shape[2]):
                if i == 0:
                    ideb = 0
                    ifin = 1
                else:
                    ideb = i - 1
                    ifin = i
                dataFace_acc_tmp[:, :, i] = dataFace_vit_tmp[:, :, ifin] - dataFace_vit_tmp[:, :, ideb]
                dataPose_acc_tmp[:, :, i] = dataPose_vit_tmp[:, :, ifin] - dataPose_vit_tmp[:, :, ideb]
                dataAnglesTete_acc_tmp[i, :] = dataAnglesTete_vit_tmp[ifin, :] - dataAnglesTete_vit_tmp[ideb, :]
                dataAnglesTete2_acc_tmp[i, :] = dataAnglesTete2_vit_tmp[ifin, :] - dataAnglesTete2_vit_tmp[ideb, :]

            dataFace_acc.append(dataPose_acc_tmp * 1 / fps)
            dataPose_acc.append(dataPose_acc_tmp * 1 / fps)
            dataAnglesTete_acc.append(dataAnglesTete_acc_tmp[:,1] * 1 / fps)
            dataAnglesTete2_acc.append(dataAnglesTete2_acc_tmp[:,1] * 1 / fps)

            # Position des mains 1 et 2 (relative au point 1)
            # Attention main 1 = main droite (a gauche de limage)
            main1_relpt1.append(np.swapaxes(dataPoseTmp[4, ::2, :], 0, 1))
            main2_relpt1.append(np.swapaxes(dataPoseTmp[7, ::2, :], 0, 1))

            # Vitesse de deplacement des mains 1 et 2
            main1_relpt1_vit.append(np.swapaxes(dataPose_vit[-1][4, ::2, :], 0, 1))
            main2_relpt1_vit.append(np.swapaxes(dataPose_vit[-1][7, ::2, :], 0, 1))

            # Acc des mains 1 et 2
            main1_relpt1_acc.append(np.swapaxes(dataPose_acc[-1][4, ::2, :], 0, 1))
            main2_relpt1_acc.append(np.swapaxes(dataPose_acc[-1][7, ::2, :], 0, 1))

            # Sourcils relatifs au nez
            sourcil1_relHautNez_vect = np.swapaxes(0.5 * (dataFaceTmp[19, ::2, :] + dataFaceTmp[20, ::2, :]) - dataFaceTmp[27, ::2, :], 0, 1)
            sourcil2_relHautNez_vect = np.swapaxes(0.5 * (dataFaceTmp[23, ::2, :] + dataFaceTmp[24, ::2, :]) - dataFaceTmp[27, ::2, :], 0, 1)

            sourcil1_relHautNez_dist.append(np.sqrt(np.sum(np.square(sourcil1_relHautNez_vect), axis=1)))
            sourcil2_relHautNez_dist.append(np.sqrt(np.sum(np.square(sourcil2_relHautNez_vect), axis=1)))

            # Vecteur bouche gauche-droite
            boucheGD_vect = np.swapaxes(dataFaceTmp[64, ::2, :] - dataFaceTmp[60, ::2, :], 0, 1)

            # Distance bouche gauche-droite
            boucheGD_dist.append(np.sqrt(np.sum(np.square(boucheGD_vect), axis=1)))

            # Vecteur bouche haut-bas
            boucheHB_vect = np.swapaxes(dataFaceTmp[66, ::2, :] - dataFaceTmp[62, ::2, :], 0, 1)

            # Distance bouche haut-bas
            boucheHB_dist.append(np.sqrt(np.sum(np.square(boucheHB_vect), axis=1)))

            # Position des mains 1 et 2 relative aux coudes
            main1_relCoude.append(np.swapaxes(dataPoseTmp[4, ::2, :] - dataPoseTmp[3, ::2, :], 0, 1))
            main2_relCoude.append(np.swapaxes(dataPoseTmp[7, ::2, :] - dataPoseTmp[6, ::2, :], 0, 1))

            # Vitesse de deplacement des mains 1 et 2
            main1_relCoude_vit.append(np.swapaxes(dataPose_vit[-1][4, ::2, :]-dataPose_vit[-1][3, ::2, :], 0, 1))
            main2_relCoude_vit.append(np.swapaxes(dataPose_vit[-1][7, ::2, :]-dataPose_vit[-1][6, ::2, :], 0, 1))

            # Acc des mains 1 et 2
            main1_relCoude_acc.append(np.swapaxes(dataPose_acc[-1][4, ::2, :]-dataPose_acc[-1][3, ::2, :], 0, 1))
            main2_relCoude_acc.append(np.swapaxes(dataPose_acc[-1][7, ::2, :]-dataPose_acc[-1][6, ::2, :], 0, 1))

            # Position des mains 1 et 2 relative au nez
            main1_relNez.append(np.swapaxes(dataPoseTmp[4, ::2, :] - dataPoseTmp[0, ::2, :], 0, 1))
            main2_relNez.append(np.swapaxes(dataPoseTmp[7, ::2, :] - dataPoseTmp[0, ::2, :], 0, 1))

            # Vitesse de deplacement des mains 1 et 2
            main1_relNez_vit.append(np.swapaxes(dataPose_vit[-1][4, ::2, :]-dataPose_vit[-1][0, ::2, :], 0, 1))
            main2_relNez_vit.append(np.swapaxes(dataPose_vit[-1][7, ::2, :]-dataPose_vit[-1][0, ::2, :], 0, 1))

            # Acc des mains 1 et 2
            main1_relNez_acc.append(np.swapaxes(dataPose_acc[-1][4, ::2, :]-dataPose_vit[-1][0, ::2, :], 0, 1))
            main2_relNez_acc.append(np.swapaxes(dataPose_acc[-1][7, ::2, :]-dataPose_vit[-1][0, ::2, :], 0, 1))

            # Position des coudes 1 et 2 relative aux epaules
            coude1_relEpaule.append(np.swapaxes(dataPoseTmp[3, ::2, :] - dataPoseTmp[2, ::2, :], 0, 1))
            coude2_relEpaule.append(np.swapaxes(dataPoseTmp[6, ::2, :] - dataPoseTmp[5, ::2, :], 0, 1))

            coude1_relEpaule_vit.append(np.swapaxes(dataPose_vit[-1][3, ::2, :] - dataPose_vit[-1][2, ::2, :], 0, 1))
            coude2_relEpaule_vit.append(np.swapaxes(dataPose_vit[-1][6, ::2, :] - dataPose_vit[-1][5, ::2, :], 0, 1))

            coude1_relEpaule_acc.append(np.swapaxes(dataPose_acc[-1][3, ::2, :] - dataPose_acc[-1][2, ::2, :], 0, 1))
            coude2_relEpaule_acc.append(np.swapaxes(dataPose_acc[-1][6, ::2, :] - dataPose_acc[-1][5, ::2, :], 0, 1))

            # Position des epaules 1 et 2 relative au pt 1
            epaule1_relpt1.append(np.swapaxes(dataPoseTmp[2, ::2, :] - dataPoseTmp[1, ::2, :], 0, 1))
            epaule2_relpt1.append(np.swapaxes(dataPoseTmp[5, ::2, :] - dataPoseTmp[1, ::2, :], 0, 1))

            epaule1_relpt1_vit.append(np.swapaxes(dataPose_vit[-1][2, ::2, :] - dataPose_vit[-1][1, ::2, :], 0, 1))
            epaule2_relpt1_vit.append(np.swapaxes(dataPose_vit[-1][5, ::2, :] - dataPose_vit[-1][1, ::2, :], 0, 1))

            epaule1_relpt1_acc.append(np.swapaxes(dataPose_acc[-1][2, ::2, :] - dataPose_acc[-1][1, ::2, :], 0, 1))
            epaule2_relpt1_acc.append(np.swapaxes(dataPose_acc[-1][5, ::2, :] - dataPose_acc[-1][1, ::2, :], 0, 1))

            # Position du nez relative au pt 1
            nez_relpt1.append(np.swapaxes(dataFaceTmp[30, ::2, :] - dataPoseTmp[1, ::2, :], 0, 1))

            # Vitesse de deplacement normalisee (indique plutot une direction)
            main1_vit_norm.append(preprocessing.normalize(main1_relpt1_vit[-1], norm='l2'))
            main2_vit_norm.append(preprocessing.normalize(main2_relpt1_vit[-1], norm='l2'))

            # Acceleration normalisee (indique plutot une direction)
            main1_acc_norm.append(preprocessing.normalize(main1_relpt1_acc[-1], norm='l2'))
            main2_acc_norm.append(preprocessing.normalize(main2_relpt1_acc[-1], norm='l2'))

            # Vecteur main 1 - main 2
            main1_rel_Main2_vect.append(main1_relpt1[-1] - main2_relpt1[-1])

            main1_rel_Main2_vit.append(main1_relpt1_vit[-1] - main2_relpt1_vit[-1])

            main1_rel_Main2_acc.append(main1_relpt1_acc[-1] - main2_relpt1_acc[-1])

            # Distance main 1 - main 2
            main1_rel_Main2_dist.append(np.sqrt(np.sum(np.square(main1_rel_Main2_vect[-1]), axis=1)))

            # Cosinus de langle coude
            cos_coude1.append(np.sum(np.multiply(main1_relCoude[-1], coude1_relEpaule[-1]), axis=1) / (
                        np.sqrt(np.sum(np.square(main1_relCoude[-1]), axis=1)) * np.sqrt(
                    np.sum(np.square(coude1_relEpaule[-1]), axis=1))))
            cos_coude2.append(np.sum(np.multiply(main2_relCoude[-1], coude2_relEpaule[-1]), axis=1) / (
                        np.sqrt(np.sum(np.square(main2_relCoude[-1]), axis=1)) * np.sqrt(
                    np.sum(np.square(coude2_relEpaule[-1]), axis=1))))

            #print(cos_coude1[-1].shape)
            cos_coude1_tmp = cos_coude1[-1]
            cos_coude2_tmp = cos_coude2[-1]
            cos_coude1_vit_tmp = np.zeros(cos_coude1[-1].shape)
            cos_coude1_acc_tmp = np.zeros(cos_coude1[-1].shape)
            cos_coude2_vit_tmp = np.zeros(cos_coude1[-1].shape)
            cos_coude2_acc_tmp = np.zeros(cos_coude1[-1].shape)
            for i in range(cos_coude1[-1].shape[0]):
                if i == 0:
                    ideb = 0
                    ifin = 1
                else:
                    ideb = i - 1
                    ifin = i
                cos_coude1_vit_tmp[i] = cos_coude1_tmp[ifin] - cos_coude1_tmp[ideb]
                cos_coude2_vit_tmp[i] = cos_coude2_tmp[ifin] - cos_coude2_tmp[ideb]

            cos_coude1_vit.append(cos_coude1_vit_tmp * 1 / fps)
            cos_coude2_vit.append(cos_coude2_vit_tmp * 1 / fps)

            for i in range(cos_coude1[-1].shape[0]):
                if i == 0:
                    ideb = 0
                    ifin = 1
                else:
                    ideb = i - 1
                    ifin = i
                cos_coude1_acc_tmp[i] = cos_coude1_vit_tmp[ifin] - cos_coude1_vit_tmp[ideb]
                cos_coude2_acc_tmp[i] = cos_coude2_vit_tmp[ifin] - cos_coude2_vit_tmp[ideb]

            cos_coude1_acc.append(cos_coude1_acc_tmp * 1 / fps)
            cos_coude2_acc.append(cos_coude2_acc_tmp * 1 / fps)


            # Cosinus de langle epaule
            cos_epaule1.append(np.sum(np.multiply(coude1_relEpaule[-1], epaule1_relpt1[-1]), axis=1) / (
                        np.sqrt(np.sum(np.square(coude1_relEpaule[-1]), axis=1)) * np.sqrt(
                    np.sum(np.square(epaule1_relpt1[-1]), axis=1))))
            cos_epaule2.append(np.sum(np.multiply(coude2_relEpaule[-1], epaule2_relpt1[-1]), axis=1) / (
                        np.sqrt(np.sum(np.square(coude2_relEpaule[-1]), axis=1)) * np.sqrt(
                    np.sum(np.square(epaule2_relpt1[-1]), axis=1))))

            #print(cos_epaule1[-1].shape)
            cos_epaule1_tmp = cos_epaule1[-1]
            cos_epaule2_tmp = cos_epaule2[-1]
            cos_epaule1_vit_tmp = np.zeros(cos_epaule1[-1].shape)
            cos_epaule1_acc_tmp = np.zeros(cos_epaule1[-1].shape)
            cos_epaule2_vit_tmp = np.zeros(cos_epaule1[-1].shape)
            cos_epaule2_acc_tmp = np.zeros(cos_epaule1[-1].shape)
            for i in range(cos_epaule1[-1].shape[0]):
                if i == 0:
                    ideb = 0
                    ifin = 1
                else:
                    ideb = i - 1
                    ifin = i
                cos_epaule1_vit_tmp[i] = cos_epaule1_tmp[ifin] - cos_epaule1_tmp[ideb]
                cos_epaule2_vit_tmp[i] = cos_epaule2_tmp[ifin] - cos_epaule2_tmp[ideb]

            cos_epaule1_vit.append(cos_epaule1_vit_tmp * 1 / fps)
            cos_epaule2_vit.append(cos_epaule2_vit_tmp * 1 / fps)

            for i in range(cos_epaule1[-1].shape[0]):
                if i == 0:
                    ideb = 0
                    ifin = 1
                else:
                    ideb = i - 1
                    ifin = i
                cos_epaule1_acc_tmp[i] = cos_epaule1_vit_tmp[ifin] - cos_epaule1_vit_tmp[ideb]
                cos_epaule2_acc_tmp[i] = cos_epaule2_vit_tmp[ifin] - cos_epaule2_vit_tmp[ideb]

            cos_epaule1_acc.append(cos_epaule1_acc_tmp * 1 / fps)
            cos_epaule2_acc.append(cos_epaule2_acc_tmp * 1 / fps)


            dataBrut_pfh.append(np.column_stack((dataAnglesTete2[-1],
                                                 dataAnglesTete2_vit[-1],
                                                 dataAnglesTete2_acc[-1],
                                                 main1_relpt1[-1],
                                                 main2_relpt1[-1],
                                                 main1_relpt1_vit[-1],
                                                 main2_relpt1_vit[-1],
                                                 main1_relpt1_acc[-1],
                                                 main2_relpt1_acc[-1],
                                                 boucheGD_dist[-1],
                                                 boucheHB_dist[-1],
                                                 sourcil1_relHautNez_dist[-1],
                                                 sourcil2_relHautNez_dist[-1],
                                                 main1_relCoude[-1],
                                                 main2_relCoude[-1],
                                                 main1_relCoude_vit[-1],
                                                 main2_relCoude_vit[-1],
                                                 main1_relCoude_acc[-1],
                                                 main2_relCoude_acc[-1],
                                                 main1_relNez[-1],
                                                 main2_relNez[-1],
                                                 main1_relNez_vit[-1],
                                                 main2_relNez_vit[-1],
                                                 main1_relNez_acc[-1],
                                                 main2_relNez_acc[-1],
                                                 coude1_relEpaule[-1],
                                                 coude2_relEpaule[-1],
                                                 coude1_relEpaule_vit[-1],
                                                 coude2_relEpaule_vit[-1],
                                                 coude1_relEpaule_acc[-1],
                                                 coude2_relEpaule_acc[-1],
                                                 epaule1_relpt1[-1],
                                                 epaule2_relpt1[-1],
                                                 epaule1_relpt1_vit[-1],
                                                 epaule2_relpt1_vit[-1],
                                                 epaule1_relpt1_acc[-1],
                                                 epaule2_relpt1_acc[-1],
                                                 nez_relpt1[-1],
                                                 main1_vit_norm[-1],
                                                 main2_vit_norm[-1],
                                                 main1_acc_norm[-1],
                                                 main2_acc_norm[-1],
                                                 main1_rel_Main2_vect[-1],
                                                 main1_rel_Main2_vit[-1],
                                                 main1_rel_Main2_acc[-1],
                                                 main1_rel_Main2_dist[-1],
                                                 cos_coude1[-1],
                                                 cos_coude2[-1],
                                                 cos_coude1_vit[-1],
                                                 cos_coude2_vit[-1],
                                                 cos_coude1_acc[-1],
                                                 cos_coude2_acc[-1],
                                                 cos_epaule1[-1],
                                                 cos_epaule2[-1],
                                                 cos_epaule1_vit[-1],
                                                 cos_epaule2_vit[-1],
                                                 cos_epaule1_acc[-1],
                                                 cos_epaule2_acc[-1]
                                                 )))

            #dataBrut_fls.append(np.column_stack((dataBrut_fls_rh[-1], dataBrut_fls_2h[-1], dataBrut_fls_lh[-1])))

# normalisation (pas les probas ni le score de confiance des mains)
dataBrut_pfh_norm_tmp = dataBrut_pfh[0]#dataBrut_pfh[0][:,122:]
for i in range(1,len(dataBrut_pfh)):
    dataBrut_pfh_norm_tmp = np.vstack([dataBrut_pfh_norm_tmp, dataBrut_pfh[i]])#[:,122:]])
dataBrut_pfh_norm_moy = np.mean(dataBrut_pfh_norm_tmp,axis=0)
dataBrut_pfh_norm_std = np.std(dataBrut_pfh_norm_tmp,axis=0)

dataBrut_pfh_norm = list(dataBrut_pfh)
for i in range(len(dataBrut_pfh_norm)):
    dataBrut_pfh_norm[i] = dataBrut_pfh_norm[i] - dataBrut_pfh_norm_moy
    dataBrut_pfh_norm[i] = dataBrut_pfh_norm[i] / dataBrut_pfh_norm_std


np.save('/people/belissen/Python/RNN_DictaSign/databis/save1_2Dfeatures',dataBrut_pfh)
#np.savez_compressed('/people/belissen/Python/RNN_DictaSign/data/save1_pfh_mains',dataBrut_pfh=dataBrut_pfh)
np.save('/people/belissen/Python/RNN_DictaSign/databis/save1_2Dfeatures_norm',dataBrut_pfh_norm)
np.save('/people/belissen/Python/RNN_DictaSign/databis/save1_2Dfeatures_norm_moy',dataBrut_pfh_norm_moy)
np.save('/people/belissen/Python/RNN_DictaSign/databis/save1_2Dfeatures_norm_std',dataBrut_pfh_norm_std)
#np.save('/people/belissen/Python/RNN_DictaSign/data/save1_fls',dataBrut_fls)