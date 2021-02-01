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
                    dataPoseTmp[i, :, j] = dataPoseTmp[i, :, j] * 2 / larg_epaules_moy#(dataPoseTmp[i, :, j] - dataPoseTempo1[1, :, j]) * 2 / larg_epaules_moy

            for i in range(dataFaceTmp.shape[0]):
                for j in range(dataFaceTmp.shape[2]):
                    dataFaceTmp[i, :, j] = dataFaceTmp[i, :, j] * 2 / larg_epaules_moy#(dataFaceTmp[i, :, j] - dataPoseTempo1[1, :, j]) * 2 / larg_epaules_moy
            
            
            
            dataPoseTmp = np.swapaxes(dataPoseTmp, 0, 2)
            dataPoseTmp = np.reshape(dataPoseTmp, (dataPoseTmp.shape[0], -1))
            
            dataFaceTmp = np.swapaxes(dataFaceTmp, 0, 2)
            dataFaceTmp = np.reshape(dataFaceTmp, (dataFaceTmp.shape[0], -1))
            
            
            dataFace.append(dataFaceTmp)
            dataPose.append(dataPoseTmp)



            

            dataBrut_pfh.append(np.column_stack((dataPose[-1],
                                                 dataFace[-1]
                                                 )))

            dataBrut_fls.append(np.column_stack((dataBrut_fls_rh[-1], dataBrut_fls_2h[-1], dataBrut_fls_lh[-1])))

# normalisation (pas les probas ni le score de confiance des mains)
dataBrut_pfh_norm_tmp = dataBrut_pfh[0]#dataBrut_pfh[0][:,122:]
for i in range(1,len(dataBrut_pfh)):
    dataBrut_pfh_norm_tmp = np.vstack([dataBrut_pfh_norm_tmp, dataBrut_pfh[i]])#[:,122:]])
dataBrut_pfh_norm_moy = np.mean(dataBrut_pfh_norm_tmp,axis=0)
dataBrut_pfh_norm_std = np.std(dataBrut_pfh_norm_tmp,axis=0)


print(dataBrut_pfh_norm_moy)
print(dataBrut_pfh_norm_std)


dataBrut_pfh_norm = list(dataBrut_pfh)
for i in range(len(dataBrut_pfh_norm)):
    dataBrut_pfh_norm[i] = dataBrut_pfh_norm[i] - dataBrut_pfh_norm_moy
    dataBrut_pfh_norm[i] = dataBrut_pfh_norm[i] / dataBrut_pfh_norm_std


np.save('/people/belissen/Python/RNN_DictaSign/databis/save1_pfh_seul_brut',dataBrut_pfh)
np.save('/people/belissen/Python/RNN_DictaSign/databis/save1_pfh_seul_brut_norm',dataBrut_pfh_norm)
np.save('/people/belissen/Python/RNN_DictaSign/databis/save1_pfh_seul_brut_norm_moy',dataBrut_pfh_norm_moy)
np.save('/people/belissen/Python/RNN_DictaSign/databis/save1_pfh_seul_brut_norm_std',dataBrut_pfh_norm_std)
#np.save('/people/belissen/Python/RNN_DictaSign/data/save1_fls',dataBrut_fls)
