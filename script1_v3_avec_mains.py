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
dataBrut_fls = []
dataBrut_fls_rh = []
dataBrut_fls_2h = []
dataBrut_fls_lh = []

dataFace = []
dataPose = []
dataHandProb_G = []
dataHandProb_D = []
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

main1_full_relpoignet = []
main2_full_relpoignet = []

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

            fls_rh_tmp = np.zeros((nb_img, 1))
            fls_2h_tmp = np.zeros((nb_img, 1))
            fls_lh_tmp = np.zeros((nb_img, 1))

            for i_tiers in range(2, len(tiers) + 1):
                idx_transcript2 = tiers[i_tiers][3]
                # print(idx_transcript2)
                name_tier = tiers[i_tiers][0]
                # print(name_tier)
                if (idx_transcript2 == transcript and name_tier == 'RH FLS'):
                    idx_tier = tiers[i_tiers][1]
                    # print(idx_tier)
                    for i_tags in range(2, len(tags) + 1):
                        idx_tier2 = tags[i_tags][1]
                        if (idx_tier2 == idx_tier):
                            idx_token2 = tags[i_tags][4]
                            for i_tokens in range(2, len(tokens) + 1):
                                idx_token = tokens[i_tokens][0]
                                if (idx_token2 == idx_token):
                                    idx_type2 = tokens[i_tokens][1]
                                    for i_types in range(2, len(types) + 1):
                                        idx_type = types[i_types][1]
                                        if (idx_type2 == idx_type):
                                            #glose = types[i_types][0]
                                            idx_glose = types[i_types][1]
                                            debut = tags[i_tags][2]
                                            fin = tags[i_tags][3]
                                            debut_img = 25 * 60 * int(debut[3:5]) + 25 * int(debut[6:8]) + int(
                                                debut[9:11])
                                            fin_img = 25 * 60 * int(fin[3:5]) + 25 * int(fin[6:8]) + int(fin[9:11])
                                            debut_decal = debut_img - retard
                                            fin_decal = fin_img - retard
                                            if (debut_decal >= 0 and fin_decal <= nb_img - 1):
                                                for i_img in range(debut_decal, fin_decal + 1):
                                                    if gaucher == 0:
                                                        fls_rh_tmp[i_img] = idx_glose
                                                    else:
                                                        fls_lh_tmp[i_img] = idx_glose
                if (idx_transcript2 == transcript and name_tier == '2H FLS'):
                    idx_tier = tiers[i_tiers][1]
                    # print(idx_tier)
                    for i_tags in range(2, len(tags) + 1):
                        idx_tier2 = tags[i_tags][1]
                        if (idx_tier2 == idx_tier):
                            idx_token2 = tags[i_tags][4]
                            for i_tokens in range(2, len(tokens) + 1):
                                idx_token = tokens[i_tokens][0]
                                if (idx_token2 == idx_token):
                                    idx_type2 = tokens[i_tokens][1]
                                    for i_types in range(2, len(types) + 1):
                                        idx_type = types[i_types][1]
                                        if (idx_type2 == idx_type):
                                            # glose = types[i_types][0]
                                            idx_glose = types[i_types][1]
                                            debut = tags[i_tags][2]
                                            fin = tags[i_tags][3]
                                            debut_img = 25 * 60 * int(debut[3:5]) + 25 * int(debut[6:8]) + int(
                                                debut[9:11])
                                            fin_img = 25 * 60 * int(fin[3:5]) + 25 * int(fin[6:8]) + int(fin[9:11])
                                            debut_decal = debut_img - retard
                                            fin_decal = fin_img - retard
                                            if (debut_decal >= 0 and fin_decal <= nb_img - 1):
                                                for i_img in range(debut_decal, fin_decal + 1):
                                                    fls_2h_tmp[i_img] = idx_glose
                if (idx_transcript2 == transcript and name_tier == 'LH FLS'):
                    idx_tier = tiers[i_tiers][1]
                    # print(idx_tier)
                    for i_tags in range(2, len(tags) + 1):
                        idx_tier2 = tags[i_tags][1]
                        if (idx_tier2 == idx_tier):
                            idx_token2 = tags[i_tags][4]
                            for i_tokens in range(2, len(tokens) + 1):
                                idx_token = tokens[i_tokens][0]
                                if (idx_token2 == idx_token):
                                    idx_type2 = tokens[i_tokens][1]
                                    for i_types in range(2, len(types) + 1):
                                        idx_type = types[i_types][1]
                                        if (idx_type2 == idx_type):
                                            # glose = types[i_types][0]
                                            idx_glose = types[i_types][1]
                                            debut = tags[i_tags][2]
                                            fin = tags[i_tags][3]
                                            debut_img = 25 * 60 * int(debut[3:5]) + 25 * int(debut[6:8]) + int(
                                                debut[9:11])
                                            fin_img = 25 * 60 * int(fin[3:5]) + 25 * int(fin[6:8]) + int(fin[9:11])
                                            debut_decal = debut_img - retard
                                            fin_decal = fin_img - retard
                                            if (debut_decal >= 0 and fin_decal <= nb_img - 1):
                                                for i_img in range(debut_decal, fin_decal + 1):
                                                    if gaucher == 0:
                                                        fls_lh_tmp[i_img] = idx_glose
                                                    else:
                                                        fls_rh_tmp[i_img] = idx_glose

            dataBrut_fls_rh.append(fls_rh_tmp.astype(int))
            dataBrut_fls_2h.append(fls_2h_tmp.astype(int))
            dataBrut_fls_lh.append(fls_lh_tmp.astype(int))

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

            dataMains = np.load(
                '/people/belissen/Videos/DictaSign/results/hands/S'+str(S)+'_task'+str(T)+'_'+locut+'.npz')

            if gaucher == 0:
                dataHandProb_G.append(dataMains['Tab_probs_G'])
                dataHandProb_D.append(dataMains['Tab_probs_D'])
            else:
                dataHandProb_G.append(dataMains['Tab_probs_D'])
                dataHandProb_D.append(dataMains['Tab_probs_G'])


            angles1tmp = data3D['anglesTete']
            angles2tmp = data3D['anglesTete2']
            if gaucher == 1:
                angles1tmp[:, 1] *= -1
                angles1tmp[:, 2] *= -1
                angles2tmp[:, 1] *= -1
                angles2tmp[:, 2] *= -1
            dataAnglesTete.append(angles1tmp)
            dataAnglesTete2.append(angles2tmp)
            
            
            data2D = np.load('/people/belissen/Videos/DictaSign/results/2D_hands/S'+str(S)+'_task'+str(T)+'_'+locut+'.npz')
            
            if gaucher == 0:
                mainG_2D = data2D['mainG_2D']
                mainD_2D = data2D['mainD_2D']
            else:
                mainG_2D = data2D['mainD_2D']
                mainD_2D = data2D['mainG_2D']
                mainG_2D[:,:,0] *= -1
                mainD_2D[:,:,0] *= -1
            
            mainG_2D = np.swapaxes(np.swapaxes(mainG_2D,0,1),1,2)
            mainD_2D = np.swapaxes(np.swapaxes(mainD_2D,0,1),1,2)
            
            # Mains relatives au poignet
            dataMainG_2D_tmp = np.copy(mainG_2D)
            dataMainD_2D_tmp = np.copy(mainD_2D)

            for i in range(mainG_2D.shape[0]):
                for j in range(mainG_2D.shape[2]):
                    mainG_2D[i, 0:2, j] = (mainG_2D[i, 0:2, j] - dataMainG_2D_tmp[0, 0:2, j]) * 2 / larg_epaules_moy
                    mainD_2D[i, 0:2, j] = (mainD_2D[i, 0:2, j] - dataMainD_2D_tmp[0, 0:2, j]) * 2 / larg_epaules_moy

            mainG_2D_tab = np.zeros((mainG_2D.shape[2],63))
            mainD_2D_tab = np.zeros((mainD_2D.shape[2],63))
            
            for i_point_main in range(21):
                mainG_2D_tab[:,3*i_point_main]   = mainG_2D[i_point_main,0,:]
                mainG_2D_tab[:,3*i_point_main+1] = mainG_2D[i_point_main,1,:]
                mainG_2D_tab[:,3*i_point_main+2] = mainG_2D[i_point_main,2,:]
                mainD_2D_tab[:,3*i_point_main]   = mainD_2D[i_point_main,0,:]
                mainD_2D_tab[:,3*i_point_main+1] = mainD_2D[i_point_main,1,:]
                mainD_2D_tab[:,3*i_point_main+2] = mainD_2D[i_point_main,2,:]
            
            main1_full_relpoignet.append(mainD_2D_tab[:,2:])
            main2_full_relpoignet.append(mainG_2D_tab[:,2:])
            # (on enleve les deux premieres colonnes qui sont les coordonnees XY
            # du point 0 de la main, toujours a 0)
            

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
            dataAnglesTete_vit.append(dataAnglesTete_vit_tmp * 1 / fps)
            dataAnglesTete2_vit.append(dataAnglesTete2_vit_tmp * 1 / fps)

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
            dataAnglesTete_acc.append(dataAnglesTete_acc_tmp * 1 / fps)
            dataAnglesTete2_acc.append(dataAnglesTete2_acc_tmp * 1 / fps)

            # Position des mains 1 et 2 (relative au point 1)
            # Attention main 1 = main droite (a gauche de limage)
            main1_relpt1.append(np.swapaxes(dataPoseTmp[4, :, :], 0, 1))
            main2_relpt1.append(np.swapaxes(dataPoseTmp[7, :, :], 0, 1))

            # Vitesse de deplacement des mains 1 et 2
            main1_relpt1_vit.append(np.swapaxes(dataPose_vit[-1][4, :, :], 0, 1))
            main2_relpt1_vit.append(np.swapaxes(dataPose_vit[-1][7, :, :], 0, 1))

            # Acc des mains 1 et 2
            main1_relpt1_acc.append(np.swapaxes(dataPose_acc[-1][4, :, :], 0, 1))
            main2_relpt1_acc.append(np.swapaxes(dataPose_acc[-1][7, :, :], 0, 1))

            # Sourcils relatifs au nez
            sourcil1_relHautNez_vect = np.swapaxes(0.5 * (dataFaceTmp[19, :, :] + dataFaceTmp[20, :, :]) - dataFaceTmp[27, :, :], 0, 1)
            sourcil2_relHautNez_vect = np.swapaxes(0.5 * (dataFaceTmp[23, :, :] + dataFaceTmp[24, :, :]) - dataFaceTmp[27, :, :], 0, 1)

            sourcil1_relHautNez_dist.append(np.sqrt(np.sum(np.square(sourcil1_relHautNez_vect), axis=1)))
            sourcil2_relHautNez_dist.append(np.sqrt(np.sum(np.square(sourcil2_relHautNez_vect), axis=1)))

            # Vecteur bouche gauche-droite
            boucheGD_vect = np.swapaxes(dataFaceTmp[64, :, :] - dataFaceTmp[60, :, :], 0, 1)

            # Distance bouche gauche-droite
            boucheGD_dist.append(np.sqrt(np.sum(np.square(boucheGD_vect), axis=1)))

            # Vecteur bouche haut-bas
            boucheHB_vect = np.swapaxes(dataFaceTmp[66, :, :] - dataFaceTmp[62, :, :], 0, 1)

            # Distance bouche haut-bas
            boucheHB_dist.append(np.sqrt(np.sum(np.square(boucheHB_vect), axis=1)))

            # Position des mains 1 et 2 relative aux coudes
            main1_relCoude.append(np.swapaxes(dataPoseTmp[4, :, :] - dataPoseTmp[3, :, :], 0, 1))
            main2_relCoude.append(np.swapaxes(dataPoseTmp[7, :, :] - dataPoseTmp[6, :, :], 0, 1))

            # Vitesse de deplacement des mains 1 et 2
            main1_relCoude_vit.append(np.swapaxes(dataPose_vit[-1][4, :, :]-dataPose_vit[-1][3, :, :], 0, 1))
            main2_relCoude_vit.append(np.swapaxes(dataPose_vit[-1][7, :, :]-dataPose_vit[-1][6, :, :], 0, 1))

            # Acc des mains 1 et 2
            main1_relCoude_acc.append(np.swapaxes(dataPose_acc[-1][4, :, :]-dataPose_acc[-1][3, :, :], 0, 1))
            main2_relCoude_acc.append(np.swapaxes(dataPose_acc[-1][7, :, :]-dataPose_acc[-1][6, :, :], 0, 1))

            # Position des mains 1 et 2 relative au nez
            main1_relNez.append(np.swapaxes(dataPoseTmp[4, :, :] - dataPoseTmp[0, :, :], 0, 1))
            main2_relNez.append(np.swapaxes(dataPoseTmp[7, :, :] - dataPoseTmp[0, :, :], 0, 1))

            # Vitesse de deplacement des mains 1 et 2
            main1_relNez_vit.append(np.swapaxes(dataPose_vit[-1][4, :, :]-dataPose_vit[-1][0, :, :], 0, 1))
            main2_relNez_vit.append(np.swapaxes(dataPose_vit[-1][7, :, :]-dataPose_vit[-1][0, :, :], 0, 1))

            # Acc des mains 1 et 2
            main1_relNez_acc.append(np.swapaxes(dataPose_acc[-1][4, :, :]-dataPose_vit[-1][0, :, :], 0, 1))
            main2_relNez_acc.append(np.swapaxes(dataPose_acc[-1][7, :, :]-dataPose_vit[-1][0, :, :], 0, 1))

            # Position des coudes 1 et 2 relative aux epaules
            coude1_relEpaule.append(np.swapaxes(dataPoseTmp[3, :, :] - dataPoseTmp[2, :, :], 0, 1))
            coude2_relEpaule.append(np.swapaxes(dataPoseTmp[6, :, :] - dataPoseTmp[5, :, :], 0, 1))

            coude1_relEpaule_vit.append(np.swapaxes(dataPose_vit[-1][3, :, :] - dataPose_vit[-1][2, :, :], 0, 1))
            coude2_relEpaule_vit.append(np.swapaxes(dataPose_vit[-1][6, :, :] - dataPose_vit[-1][5, :, :], 0, 1))

            coude1_relEpaule_acc.append(np.swapaxes(dataPose_acc[-1][3, :, :] - dataPose_acc[-1][2, :, :], 0, 1))
            coude2_relEpaule_acc.append(np.swapaxes(dataPose_acc[-1][6, :, :] - dataPose_acc[-1][5, :, :], 0, 1))

            # Position des epaules 1 et 2 relative au pt 1
            epaule1_relpt1.append(np.swapaxes(dataPoseTmp[2, :, :] - dataPoseTmp[1, :, :], 0, 1))
            epaule2_relpt1.append(np.swapaxes(dataPoseTmp[5, :, :] - dataPoseTmp[1, :, :], 0, 1))

            epaule1_relpt1_vit.append(np.swapaxes(dataPose_vit[-1][2, :, :] - dataPose_vit[-1][1, :, :], 0, 1))
            epaule2_relpt1_vit.append(np.swapaxes(dataPose_vit[-1][5, :, :] - dataPose_vit[-1][1, :, :], 0, 1))

            epaule1_relpt1_acc.append(np.swapaxes(dataPose_acc[-1][2, :, :] - dataPose_acc[-1][1, :, :], 0, 1))
            epaule2_relpt1_acc.append(np.swapaxes(dataPose_acc[-1][5, :, :] - dataPose_acc[-1][1, :, :], 0, 1))

            # Position du nez relative au pt 1
            nez_relpt1.append(np.swapaxes(dataFaceTmp[30, :, :] - dataPoseTmp[1, :, :], 0, 1))

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

            # Vecteur orthonormal au plan du coude
            cross_coude1_tmp = np.cross(main1_relCoude[-1], coude1_relEpaule[-1])
            cross_coude2_tmp = np.cross(main2_relCoude[-1], coude2_relEpaule[-1])
            for i in range(3):
                cross_coude1_tmp[:, i] /= np.sqrt(np.sum(np.square(main1_relCoude[-1]), axis=1)) * np.sqrt(
                    np.sum(np.square(coude1_relEpaule[-1]), axis=1)) * np.sqrt(1 - np.square(cos_coude1[-1]))
                cross_coude2_tmp[:, i] /= np.sqrt(np.sum(np.square(main2_relCoude[-1]), axis=1)) * np.sqrt(
                    np.sum(np.square(coude2_relEpaule[-1]), axis=1)) * np.sqrt(1 - np.square(cos_coude2[-1]))

            ortho_coude1.append(cross_coude1_tmp)
            ortho_coude2.append(cross_coude2_tmp)

            #print(ortho_coude1[-1].shape)
            ortho_coude1_tmp = ortho_coude1[-1]
            ortho_coude2_tmp = ortho_coude2[-1]
            ortho_coude1_vit_tmp = np.zeros(ortho_coude1[-1].shape)
            ortho_coude1_acc_tmp = np.zeros(ortho_coude1[-1].shape)
            ortho_coude2_vit_tmp = np.zeros(ortho_coude1[-1].shape)
            ortho_coude2_acc_tmp = np.zeros(ortho_coude1[-1].shape)
            for i in range(ortho_coude1[-1].shape[0]):
                if i == 0:
                    ideb = 0
                    ifin = 1
                else:
                    ideb = i - 1
                    ifin = i
                ortho_coude1_vit_tmp[i,:] = ortho_coude1_tmp[ifin,:] - ortho_coude1_tmp[ideb,:]
                ortho_coude2_vit_tmp[i,:] = ortho_coude2_tmp[ifin,:] - ortho_coude2_tmp[ideb,:]

            ortho_coude1_vit.append(ortho_coude1_vit_tmp * 1 / fps)
            ortho_coude2_vit.append(ortho_coude2_vit_tmp * 1 / fps)

            for i in range(ortho_coude1[-1].shape[0]):
                if i == 0:
                    ideb = 0
                    ifin = 1
                else:
                    ideb = i - 1
                    ifin = i
                ortho_coude1_acc_tmp[i,:] = ortho_coude1_vit_tmp[ifin,:] - ortho_coude1_vit_tmp[ideb,:]
                ortho_coude2_acc_tmp[i,:] = ortho_coude2_vit_tmp[ifin,:] - ortho_coude2_vit_tmp[ideb,:]

            ortho_coude1_acc.append(ortho_coude1_acc_tmp * 1 / fps)
            ortho_coude2_acc.append(ortho_coude2_acc_tmp * 1 / fps)

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

            # Vecteur orthonormal au plan de lepaule
            cross_epaule1_tmp = np.cross(coude1_relEpaule[-1], epaule1_relpt1[-1])
            cross_epaule2_tmp = np.cross(coude2_relEpaule[-1], epaule2_relpt1[-1])
            for i in range(3):
                cross_epaule1_tmp[:, i] /= np.sqrt(np.sum(np.square(coude1_relEpaule[-1]), axis=1)) * np.sqrt(
                    np.sum(np.square(epaule1_relpt1[-1]), axis=1)) * np.sqrt(1 - np.square(cos_epaule1[-1]))
                cross_epaule2_tmp[:, i] /= np.sqrt(np.sum(np.square(coude2_relEpaule[-1]), axis=1)) * np.sqrt(
                    np.sum(np.square(epaule2_relpt1[-1]), axis=1)) * np.sqrt(1 - np.square(cos_epaule2[-1]))

            ortho_epaule1.append(cross_epaule1_tmp)
            ortho_epaule2.append(cross_epaule2_tmp)

            ortho_epaule1_tmp = ortho_epaule1[-1]
            ortho_epaule2_tmp = ortho_epaule2[-1]
            ortho_epaule1_vit_tmp = np.zeros(ortho_epaule1[-1].shape)
            ortho_epaule1_acc_tmp = np.zeros(ortho_epaule1[-1].shape)
            ortho_epaule2_vit_tmp = np.zeros(ortho_epaule1[-1].shape)
            ortho_epaule2_acc_tmp = np.zeros(ortho_epaule1[-1].shape)
            for i in range(ortho_epaule1[-1].shape[0]):
                if i == 0:
                    ideb = 0
                    ifin = 1
                else:
                    ideb = i - 1
                    ifin = i
                ortho_epaule1_vit_tmp[i, :] = ortho_epaule1_tmp[ifin, :] - ortho_epaule1_tmp[ideb, :]
                ortho_epaule2_vit_tmp[i, :] = ortho_epaule2_tmp[ifin, :] - ortho_epaule2_tmp[ideb, :]

            ortho_epaule1_vit.append(ortho_epaule1_vit_tmp * 1 / fps)
            ortho_epaule2_vit.append(ortho_epaule2_vit_tmp * 1 / fps)

            for i in range(ortho_epaule1[-1].shape[0]):
                if i == 0:
                    ideb = 0
                    ifin = 1
                else:
                    ideb = i - 1
                    ifin = i
                ortho_epaule1_acc_tmp[i, :] = ortho_epaule1_vit_tmp[ifin, :] - ortho_epaule1_vit_tmp[ideb, :]
                ortho_epaule2_acc_tmp[i, :] = ortho_epaule2_vit_tmp[ifin, :] - ortho_epaule2_vit_tmp[ideb, :]

            ortho_epaule1_acc.append(ortho_epaule1_acc_tmp * 1 / fps)
            ortho_epaule2_acc.append(ortho_epaule2_acc_tmp * 1 / fps)

            dataBrut_pfh.append(np.column_stack((dataHandProb_G[-1],
                                                 dataHandProb_D[-1],
                                                 main1_full_relpoignet[-1],
                                                 main2_full_relpoignet[-1],
                                                 dataAnglesTete2[-1],
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
                                                 ortho_coude1[-1],
                                                 ortho_coude2[-1],
                                                 ortho_coude1_vit[-1],
                                                 ortho_coude2_vit[-1],
                                                 ortho_coude1_acc[-1],
                                                 ortho_coude2_acc[-1],
                                                 cos_epaule1[-1],
                                                 cos_epaule2[-1],
                                                 cos_epaule1_vit[-1],
                                                 cos_epaule2_vit[-1],
                                                 cos_epaule1_acc[-1],
                                                 cos_epaule2_acc[-1],
                                                 ortho_epaule1[-1],
                                                 ortho_epaule2[-1],
                                                 ortho_epaule1_vit[-1],
                                                 ortho_epaule2_vit[-1],
                                                 ortho_epaule1_acc[-1],
                                                 ortho_epaule2_acc[-1]
                                                 )))

            dataBrut_fls.append(np.column_stack((dataBrut_fls_rh[-1], dataBrut_fls_2h[-1], dataBrut_fls_lh[-1])))

# normalisation (pas les probas ni le score de confiance des mains)
dataBrut_pfh_norm_tmp = dataBrut_pfh[0]#dataBrut_pfh[0][:,122:]
for i in range(1,len(dataBrut_pfh)):
    dataBrut_pfh_norm_tmp = np.vstack([dataBrut_pfh_norm_tmp, dataBrut_pfh[i]])#[:,122:]])
dataBrut_pfh_norm_moy = np.mean(dataBrut_pfh_norm_tmp,axis=0)
dataBrut_pfh_norm_std = np.std(dataBrut_pfh_norm_tmp,axis=0)

dataBrut_pfh_norm_moy[0:122] = 0
dataBrut_pfh_norm_std[0:122] = 1

dataBrut_pfh_norm_moy[122:122+3*21:3] = 0
dataBrut_pfh_norm_moy[183:183+3*21:3] = 0
dataBrut_pfh_norm_std[122:122+3*21:3] = 1
dataBrut_pfh_norm_std[183:183+3*21:3] = 1

dataBrut_pfh_norm = list(dataBrut_pfh)
for i in range(len(dataBrut_pfh_norm)):
    dataBrut_pfh_norm[i] = dataBrut_pfh_norm[i] - dataBrut_pfh_norm_moy
    dataBrut_pfh_norm[i] = dataBrut_pfh_norm[i] / dataBrut_pfh_norm_std


np.save('/people/belissen/Python/RNN_DictaSign/databis/save1_pfh_mains',dataBrut_pfh)
#np.savez_compressed('/people/belissen/Python/RNN_DictaSign/data/save1_pfh_mains',dataBrut_pfh=dataBrut_pfh)
np.save('/people/belissen/Python/RNN_DictaSign/databis/save1_pfh_mains_norm',dataBrut_pfh_norm)
np.save('/people/belissen/Python/RNN_DictaSign/databis/save1_pfh_mains_norm_moy',dataBrut_pfh_norm_moy)
np.save('/people/belissen/Python/RNN_DictaSign/databis/save1_pfh_mains_norm_std',dataBrut_pfh_norm_std)
#np.save('/people/belissen/Python/RNN_DictaSign/data/save1_fls',dataBrut_fls)