import numpy as np
import csv
import sys
from sklearn import preprocessing


nimg                = int(sys.argv[1])
vidName             = sys.argv[2]
fps                 = int(sys.argv[3])
path2features       = sys.argv[4]
load3D              = bool(sys.argv[5])
hsKoller            = bool(sys.argv[6])


path2leftHandedList = '/people/belissen/Python/CSLR_LIMSI/cslr_limsi_features/left_handed/left_handed_dictasign.csv'

def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return None

video_list = []
left_handed_list = []
with open(path2leftHandedList, 'r', encoding='utf-8-sig') as open_left_handed:
    reader_left_handed = csv.reader(open_left_handed, delimiter=';')
    for i in reader_left_handed:
        video_list.append(i[0])
        left_handed_list.append(int(i[1]))

left_handed_found = find_element_in_list(vidName, video_list)
if left_handed_found is None:
    left_handed = False
else:
    left_handed = bool(left_handed_list[left_handed_found])

if left_handed:
    handOrder = ['L', 'R']
else:
    handOrder = ['R', 'L']

if hsKoller:
    final_handShapes = np.zeros((nimg, 122))
    final_handShapes[:, :61] = np.load(path2features+'temp/'+vidName+'_HS_probs_'+handOrder[0]+'.npy')
    final_handShapes[:, 61:] = np.load(path2features+'temp/'+vidName+'_HS_probs_'+handOrder[1]+'.npy')

hand2D_raw  = np.load(path2features+'temp/'+vidName+'_2DHands.npz')
nbPtsHand2D = hand2D_raw['handL_2D'].shape[1]


if load3D:
    body3D_raw = np.load(path2features+'temp/'+vidName+'_3DBody.npy')
    face3D_raw = np.load(path2features+'temp/'+vidName+'_3DFace_predict_raw.npy')
    nbPtsPose   = body3D_raw.shape[1]

    nbPtsFace3D = face3D_raw.shape[1]
    nbPtsFace = nbPtsFace3D

    body3D_raw_xdata = -body3D_raw[:, :, 0]
    body3D_raw_ydata = body3D_raw[:, :, 1]
    body3D_raw_zdata = -body3D_raw[:, :, 2]

    face3D_raw_xdata = -face3D_raw[:, :, 0]
    face3D_raw_ydata = face3D_raw[:, :, 1]
    face3D_raw_zdata = -face3D_raw[:, :, 2]

    headAngles = np.load(path2features+'temp/'+vidName+'_headAngles_from_3Dbody.npy')
    if left_handed:
        headAngles[:, 1] *= -1
        headAngles[:, 2] *= -1
else:
    body2D_raw = np.load(path2features+'temp/'+vidName+'_2DBody.npy')
    face2D_raw = np.load(path2features+'temp/'+vidName+'_2DFace.npy')
    nbPtsPose   = body2D_raw.shape[1]

    nbPtsFace2D = face2D_raw.shape[1]
    nbPtsFace = nbPtsFace2D

    body3D_raw_xdata = body2D_raw[:, :, 0]
    body3D_raw_ydata = np.zeros((nimg, nbPtsPose))
    body3D_raw_zdata = body2D_raw[:, :, 2]

    face3D_raw_xdata = face2D_raw[:, :, 0]
    face3D_raw_ydata = np.zeros((nimg, nbPtsFace2D))
    face3D_raw_zdata = face2D_raw[:, :, 2]


if left_handed == 1:
    body3D_raw_xdata *= -1
    face3D_raw_xdata *= -1

dataPose = np.zeros((nbPtsPose, 3, nimg))
dataPose[:, 0, :] = np.swapaxes(body3D_raw_xdata, 0, 1)
dataPose[:, 1, :] = np.swapaxes(body3D_raw_ydata, 0, 1)
dataPose[:, 2, :] = np.swapaxes(body3D_raw_zdata, 0, 1)


dataFace = np.zeros((nbPtsFace, 3, nimg))
dataFace[:, 0, :] = np.swapaxes(face3D_raw_xdata, 0, 1)
dataFace[:, 1, :] = np.swapaxes(face3D_raw_ydata, 0, 1)
dataFace[:, 2, :] = np.swapaxes(face3D_raw_zdata, 0, 1)


larg_epaules_moy = np.mean(np.sqrt(np.square(dataPose[5, 0, :] - dataPose[2, 0, :]) + np.square(dataPose[5, 1, :] - dataPose[2, 1, :]) + np.square(dataPose[5, 2, :] - dataPose[2, 2, :])))

# Pose relative au point 1
dataPose1 = np.copy(dataPose)

for i in range(nbPtsPose):
    for j in range(nimg):
        dataPose[i, :, j] = (dataPose[i, :, j] - dataPose1[1, :, j]) * 2 / larg_epaules_moy

for i in range(nbPtsFace):
    for j in range(nimg):
        dataFace[i, :, j] = (dataFace[i, :, j] - dataPose1[1, :, j]) * 2 / larg_epaules_moy

if load3D:
    poseRaw_final = np.copy(dataPose)
    faceRaw_final = np.copy(dataFace)
else:
    poseRaw_final = np.copy(dataPose[:, 0::2, :])
    faceRaw_final = np.copy(dataFace[:, 0::2, :])
poseRaw_final = np.swapaxes(poseRaw_final, 0, 2)
poseRaw_final = np.reshape(poseRaw_final, (nimg, -1))
faceRaw_final = np.swapaxes(faceRaw_final, 0, 2)
faceRaw_final = np.reshape(faceRaw_final, (nimg, -1))

hand2D_raw_1 = hand2D_raw['hand'+handOrder[0]+'_2D']
hand2D_raw_2 = hand2D_raw['hand'+handOrder[1]+'_2D']
if left_handed:
    hand2D_raw_1[:,:,0] *= -1
    hand2D_raw_2[:,:,0] *= -1

hand2D_raw_1 = np.swapaxes(np.swapaxes(hand2D_raw_1, 0, 1), 1, 2)
hand2D_raw_2 = np.swapaxes(np.swapaxes(hand2D_raw_2, 0, 1), 1, 2)

# Mains relatives au poignet
hand2D_raw_1_tmp = np.copy(hand2D_raw_1)
hand2D_raw_2_tmp = np.copy(hand2D_raw_2)

for i in range(nbPtsHand2D):
    for j in range(nimg):
        hand2D_raw_1[i, 0:2, j] = (hand2D_raw_1[i, 0:2, j] - hand2D_raw_1_tmp[0, 0:2, j]) * 2 / larg_epaules_moy
        hand2D_raw_2[i, 0:2, j] = (hand2D_raw_2[i, 0:2, j] - hand2D_raw_2_tmp[0, 0:2, j]) * 2 / larg_epaules_moy

hand2D_raw_1_tab = np.zeros((nimg, 63))
hand2D_raw_2_tab = np.zeros((nimg, 63))

for i_hand_keypoint in range(21):
    for j in range(3):
        hand2D_raw_1_tab[:, 3*i_hand_keypoint+j]   = hand2D_raw_1[i_hand_keypoint, j, :]
        hand2D_raw_2_tab[:, 3*i_hand_keypoint+j]   = hand2D_raw_2[i_hand_keypoint, j, :]

hand2D_raw_final = np.zeros((nimg, 2*61))
hand2D_raw_final[:, :61] = hand2D_raw_1_tab[:, 2:]
hand2D_raw_final[:, 61:] = hand2D_raw_2_tab[:, 2:]
# (on enleve les deux premieres colonnes qui sont les coordonnees XY
# du point 0 de la main, toujours a 0)


dataFace_vel = np.zeros(dataFace.shape)
dataFace_acc = np.zeros(dataFace.shape)
dataPose_vel = np.zeros(dataPose.shape)
dataPose_acc = np.zeros(dataPose.shape)
if load3D:
    headAngles_vel = np.zeros(headAngles.shape)
    headAngles_acc = np.zeros(headAngles.shape)

for i in range(nimg):
    if i == 0:
        ideb = 0
        ifin = 1
    else:
        ideb = i - 1
        ifin = i
    dataFace_vel[:, :, i] = dataFace[:, :, ifin] - dataFace[:, :, ideb]
    dataPose_vel[:, :, i] = dataPose[:, :, ifin] - dataPose[:, :, ideb]
    if load3D:
        headAngles_vel[i, :] = headAngles[ifin, :] - headAngles[ideb, :]

dataFace_vel *= (1/fps)
dataPose_vel *= (1/fps)
if load3D:
    headAngles_vel *= (1/fps)

for i in range(nimg):
    if i == 0:
        ideb = 0
        ifin = 1
    else:
        ideb = i - 1
        ifin = i
    dataFace_acc[:, :, i] = dataFace_vel[:, :, ifin] - dataFace_vel[:, :, ideb]
    dataPose_acc[:, :, i] = dataPose_vel[:, :, ifin] - dataPose_vel[:, :, ideb]
    if load3D:
        headAngles_acc[i, :] = headAngles_vel[ifin, :] - headAngles_vel[ideb, :]

dataFace_acc *= (1/fps)
dataPose_acc *= (1/fps)
if load3D:
    headAngles_acc *= (1/fps)


# Position des mains 1 et 2 (relative au point 1)
# Attention main 1 = main droite (a gauche de limage)
hand1_wrtPt1 = np.swapaxes(dataPose[4, :, :], 0, 1)
hand2_wrtPt1 = np.swapaxes(dataPose[7, :, :], 0, 1)

# Vitesse de deplacement des mains 1 et 2
hand1_wrtPt1_vel = np.swapaxes(dataPose_vel[4, :, :], 0, 1)
hand2_wrtPt1_vel = np.swapaxes(dataPose_vel[7, :, :], 0, 1)

# Acc des mains 1 et 2
hand1_wrtPt1_acc = np.swapaxes(dataPose_acc[4, :, :], 0, 1)
hand2_wrtPt1_acc = np.swapaxes(dataPose_acc[7, :, :], 0, 1)

# Sourcils relatifs au nez
eyebrow1_wrtUpperNose_vect = np.swapaxes(0.5 * (dataFace[19, :, :] + dataFace[20, :, :]) - dataFace[27, :, :], 0, 1)
eyebrow2_wrtUpperNose_vect = np.swapaxes(0.5 * (dataFace[23, :, :] + dataFace[24, :, :]) - dataFace[27, :, :], 0, 1)

eyebrow1_wrtUpperNose_dist = np.sqrt(np.sum(np.square(eyebrow1_wrtUpperNose_vect), axis=1))
eyebrow2_wrtUpperNose_dist = np.sqrt(np.sum(np.square(eyebrow2_wrtUpperNose_vect), axis=1))

# Vecteur bouche gauche-droite
boucheGD_vect = np.swapaxes(dataFace[64, :, :] - dataFace[60, :, :], 0, 1)

# Distance bouche gauche-droite
boucheGD_dist = np.sqrt(np.sum(np.square(boucheGD_vect), axis=1))

# Vecteur bouche haut-bas
boucheHB_vect = np.swapaxes(dataFace[66, :, :] - dataFace[62, :, :], 0, 1)

# Distance bouche haut-bas
boucheHB_dist = np.sqrt(np.sum(np.square(boucheHB_vect), axis=1))

# Position des mains 1 et 2 relative aux coudes
hand1_relElbow = np.swapaxes(dataPose[4, :, :] - dataPose[3, :, :], 0, 1)
hand2_relElbow = np.swapaxes(dataPose[7, :, :] - dataPose[6, :, :], 0, 1)

# Vitesse de deplacement des mains 1 et 2
hand1_relElbow_vel = np.swapaxes(dataPose_vel[4, :, :]-dataPose_vel[3, :, :], 0, 1)
hand2_relElbow_vel = np.swapaxes(dataPose_vel[7, :, :]-dataPose_vel[6, :, :], 0, 1)

# Acc des mains 1 et 2
hand1_relElbow_acc = np.swapaxes(dataPose_acc[4, :, :]-dataPose_acc[3, :, :], 0, 1)
hand2_relElbow_acc = np.swapaxes(dataPose_acc[7, :, :]-dataPose_acc[6, :, :], 0, 1)

# Position des mains 1 et 2 relative au nez
hand1_relNez = np.swapaxes(dataPose[4, :, :] - dataPose[0, :, :], 0, 1)
hand2_relNez = np.swapaxes(dataPose[7, :, :] - dataPose[0, :, :], 0, 1)

# Vitesse de deplacement des mains 1 et 2
hand1_relNez_vel = np.swapaxes(dataPose_vel[4, :, :]-dataPose_vel[0, :, :], 0, 1)
hand2_relNez_vel = np.swapaxes(dataPose_vel[7, :, :]-dataPose_vel[0, :, :], 0, 1)

# Acc des mains 1 et 2
hand1_relNez_acc = np.swapaxes(dataPose_acc[4, :, :]-dataPose_vel[0, :, :], 0, 1)
hand2_relNez_acc = np.swapaxes(dataPose_acc[7, :, :]-dataPose_vel[0, :, :], 0, 1)

# Position des coudes 1 et 2 relative aux epaules
elbow1_relShoulder = np.swapaxes(dataPose[3, :, :] - dataPose[2, :, :], 0, 1)
elbow2_relShoulder = np.swapaxes(dataPose[6, :, :] - dataPose[5, :, :], 0, 1)

elbow1_relShoulder_vel = np.swapaxes(dataPose_vel[3, :, :] - dataPose_vel[2, :, :], 0, 1)
elbow2_relShoulder_vel = np.swapaxes(dataPose_vel[6, :, :] - dataPose_vel[5, :, :], 0, 1)

elbow1_relShoulder_acc = np.swapaxes(dataPose_acc[3, :, :] - dataPose_acc[2, :, :], 0, 1)
elbow2_relShoulder_acc = np.swapaxes(dataPose_acc[6, :, :] - dataPose_acc[5, :, :], 0, 1)

# Position des epaules 1 et 2 relative au pt 1
shoulder1_wrtPt1 = np.swapaxes(dataPose[2, :, :] - dataPose[1, :, :], 0, 1)
shoulder2_wrtPt1 = np.swapaxes(dataPose[5, :, :] - dataPose[1, :, :], 0, 1)

shoulder1_wrtPt1_vel = np.swapaxes(dataPose_vel[2, :, :] - dataPose_vel[1, :, :], 0, 1)
shoulder2_wrtPt1_vel = np.swapaxes(dataPose_vel[5, :, :] - dataPose_vel[1, :, :], 0, 1)

shoulder1_wrtPt1_acc = np.swapaxes(dataPose_acc[2, :, :] - dataPose_acc[1, :, :], 0, 1)
shoulder2_wrtPt1_acc = np.swapaxes(dataPose_acc[5, :, :] - dataPose_acc[1, :, :], 0, 1)

# Position du nez relative au pt 1
nez_wrtPt1 = np.swapaxes(dataFace[30, :, :] - dataPose[1, :, :], 0, 1)

# Vitesse de deplacement normalisee (indique plutot une direction)
hand1_vel_norm = preprocessing.normalize(hand1_wrtPt1_vel, norm='l2')
hand2_vel_norm = preprocessing.normalize(hand2_wrtPt1_vel, norm='l2')

# Acceleration normalisee (indique plutot une direction)
hand1_acc_norm = preprocessing.normalize(hand1_wrtPt1_acc, norm='l2')
hand2_acc_norm = preprocessing.normalize(hand2_wrtPt1_acc, norm='l2')

# Vecteur main 1 - main 2
hand1_rel_Hand2_vect = hand1_wrtPt1 - hand2_wrtPt1
hand1_rel_Hand2_vel = hand1_wrtPt1_vel - hand2_wrtPt1_vel
hand1_rel_Hand2_acc = hand1_wrtPt1_acc - hand2_wrtPt1_acc

# Distance main 1 - main 2
hand1_rel_Hand2_dist = np.sqrt(np.sum(np.square(hand1_rel_Hand2_vect), axis=1))

# Cosinus de langle coude
cos_elbow1 = np.sum(np.multiply(hand1_relElbow, elbow1_relShoulder), axis=1) / (
            np.sqrt(np.sum(np.square(hand1_relElbow), axis=1)) * np.sqrt(
        np.sum(np.square(elbow1_relShoulder), axis=1)))
cos_elbow2 = np.sum(np.multiply(hand2_relElbow, elbow2_relShoulder), axis=1) / (
            np.sqrt(np.sum(np.square(hand2_relElbow), axis=1)) * np.sqrt(
        np.sum(np.square(elbow2_relShoulder), axis=1)))

#print(cos_elbow1.shape)
cos_elbow1_tmp = cos_elbow1
cos_elbow2_tmp = cos_elbow2
cos_elbow1_vel_tmp = np.zeros(cos_elbow1.shape)
cos_elbow1_acc_tmp = np.zeros(cos_elbow1.shape)
cos_elbow2_vel_tmp = np.zeros(cos_elbow1.shape)
cos_elbow2_acc_tmp = np.zeros(cos_elbow1.shape)
for i in range(cos_elbow1.shape[0]):
    if i == 0:
        ideb = 0
        ifin = 1
    else:
        ideb = i - 1
        ifin = i
    cos_elbow1_vel_tmp[i] = cos_elbow1_tmp[ifin] - cos_elbow1_tmp[ideb]
    cos_elbow2_vel_tmp[i] = cos_elbow2_tmp[ifin] - cos_elbow2_tmp[ideb]

cos_elbow1_vel = cos_elbow1_vel_tmp * 1 / fps
cos_elbow2_vel = cos_elbow2_vel_tmp * 1 / fps

for i in range(cos_elbow1.shape[0]):
    if i == 0:
        ideb = 0
        ifin = 1
    else:
        ideb = i - 1
        ifin = i
    cos_elbow1_acc_tmp[i] = cos_elbow1_vel_tmp[ifin] - cos_elbow1_vel_tmp[ideb]
    cos_elbow2_acc_tmp[i] = cos_elbow2_vel_tmp[ifin] - cos_elbow2_vel_tmp[ideb]

cos_elbow1_acc = cos_elbow1_acc_tmp * 1 / fps
cos_elbow2_acc = cos_elbow2_acc_tmp * 1 / fps

# Vecteur orthonormal au plan du coude
cross_elbow1_tmp = np.cross(hand1_relElbow, elbow1_relShoulder)
cross_elbow2_tmp = np.cross(hand2_relElbow, elbow2_relShoulder)
for i in range(3):
    cross_elbow1_tmp[:, i] /= np.sqrt(np.sum(np.square(hand1_relElbow), axis=1)) * np.sqrt(
        np.sum(np.square(elbow1_relShoulder), axis=1)) * np.sqrt(1 - np.square(cos_elbow1))
    cross_elbow2_tmp[:, i] /= np.sqrt(np.sum(np.square(hand2_relElbow), axis=1)) * np.sqrt(
        np.sum(np.square(elbow2_relShoulder), axis=1)) * np.sqrt(1 - np.square(cos_elbow2))

ortho_elbow1 = cross_elbow1_tmp
ortho_elbow2 = cross_elbow2_tmp

#print(ortho_elbow1.shape)
ortho_elbow1_tmp = ortho_elbow1
ortho_elbow2_tmp = ortho_elbow2
ortho_elbow1_vel_tmp = np.zeros(ortho_elbow1.shape)
ortho_elbow1_acc_tmp = np.zeros(ortho_elbow1.shape)
ortho_elbow2_vel_tmp = np.zeros(ortho_elbow1.shape)
ortho_elbow2_acc_tmp = np.zeros(ortho_elbow1.shape)
for i in range(ortho_elbow1.shape[0]):
    if i == 0:
        ideb = 0
        ifin = 1
    else:
        ideb = i - 1
        ifin = i
    ortho_elbow1_vel_tmp[i,:] = ortho_elbow1_tmp[ifin,:] - ortho_elbow1_tmp[ideb,:]
    ortho_elbow2_vel_tmp[i,:] = ortho_elbow2_tmp[ifin,:] - ortho_elbow2_tmp[ideb,:]

ortho_elbow1_vel = ortho_elbow1_vel_tmp * 1 / fps
ortho_elbow2_vel = ortho_elbow2_vel_tmp * 1 / fps

for i in range(ortho_elbow1.shape[0]):
    if i == 0:
        ideb = 0
        ifin = 1
    else:
        ideb = i - 1
        ifin = i
    ortho_elbow1_acc_tmp[i,:] = ortho_elbow1_vel_tmp[ifin,:] - ortho_elbow1_vel_tmp[ideb,:]
    ortho_elbow2_acc_tmp[i,:] = ortho_elbow2_vel_tmp[ifin,:] - ortho_elbow2_vel_tmp[ideb,:]

ortho_elbow1_acc = ortho_elbow1_acc_tmp * 1 / fps
ortho_elbow2_acc = ortho_elbow2_acc_tmp * 1 / fps

# Cosinus de langle epaule
cos_shoulder1 = np.sum(np.multiply(elbow1_relShoulder, shoulder1_wrtPt1), axis=1) / (
            np.sqrt(np.sum(np.square(elbow1_relShoulder), axis=1)) * np.sqrt(
        np.sum(np.square(shoulder1_wrtPt1), axis=1)))
cos_shoulder2 = np.sum(np.multiply(elbow2_relShoulder, shoulder2_wrtPt1), axis=1) / (
            np.sqrt(np.sum(np.square(elbow2_relShoulder), axis=1)) * np.sqrt(
        np.sum(np.square(shoulder2_wrtPt1), axis=1)))

#print(cos_shoulder1.shape)
cos_shoulder1_tmp = cos_shoulder1
cos_shoulder2_tmp = cos_shoulder2
cos_shoulder1_vel_tmp = np.zeros(cos_shoulder1.shape)
cos_shoulder1_acc_tmp = np.zeros(cos_shoulder1.shape)
cos_shoulder2_vel_tmp = np.zeros(cos_shoulder1.shape)
cos_shoulder2_acc_tmp = np.zeros(cos_shoulder1.shape)
for i in range(cos_shoulder1.shape[0]):
    if i == 0:
        ideb = 0
        ifin = 1
    else:
        ideb = i - 1
        ifin = i
    cos_shoulder1_vel_tmp[i] = cos_shoulder1_tmp[ifin] - cos_shoulder1_tmp[ideb]
    cos_shoulder2_vel_tmp[i] = cos_shoulder2_tmp[ifin] - cos_shoulder2_tmp[ideb]

cos_shoulder1_vel = cos_shoulder1_vel_tmp * 1 / fps
cos_shoulder2_vel = cos_shoulder2_vel_tmp * 1 / fps

for i in range(cos_shoulder1.shape[0]):
    if i == 0:
        ideb = 0
        ifin = 1
    else:
        ideb = i - 1
        ifin = i
    cos_shoulder1_acc_tmp[i] = cos_shoulder1_vel_tmp[ifin] - cos_shoulder1_vel_tmp[ideb]
    cos_shoulder2_acc_tmp[i] = cos_shoulder2_vel_tmp[ifin] - cos_shoulder2_vel_tmp[ideb]

cos_shoulder1_acc = cos_shoulder1_acc_tmp * 1 / fps
cos_shoulder2_acc = cos_shoulder2_acc_tmp * 1 / fps

# Vecteur orthonormal au plan de lepaule
cross_shoulder1_tmp = np.cross(elbow1_relShoulder, shoulder1_wrtPt1)
cross_shoulder2_tmp = np.cross(elbow2_relShoulder, shoulder2_wrtPt1)
for i in range(3):
    cross_shoulder1_tmp[:, i] /= np.sqrt(np.sum(np.square(elbow1_relShoulder), axis=1)) * np.sqrt(
        np.sum(np.square(shoulder1_wrtPt1), axis=1)) * np.sqrt(1 - np.square(cos_shoulder1))
    cross_shoulder2_tmp[:, i] /= np.sqrt(np.sum(np.square(elbow2_relShoulder), axis=1)) * np.sqrt(
        np.sum(np.square(shoulder2_wrtPt1), axis=1)) * np.sqrt(1 - np.square(cos_shoulder2))

ortho_shoulder1 = cross_shoulder1_tmp
ortho_shoulder2 = cross_shoulder2_tmp

ortho_shoulder1_tmp = ortho_shoulder1
ortho_shoulder2_tmp = ortho_shoulder2
ortho_shoulder1_vel_tmp = np.zeros(ortho_shoulder1.shape)
ortho_shoulder1_acc_tmp = np.zeros(ortho_shoulder1.shape)
ortho_shoulder2_vel_tmp = np.zeros(ortho_shoulder1.shape)
ortho_shoulder2_acc_tmp = np.zeros(ortho_shoulder1.shape)
for i in range(ortho_shoulder1.shape[0]):
    if i == 0:
        ideb = 0
        ifin = 1
    else:
        ideb = i - 1
        ifin = i
    ortho_shoulder1_vel_tmp[i, :] = ortho_shoulder1_tmp[ifin, :] - ortho_shoulder1_tmp[ideb, :]
    ortho_shoulder2_vel_tmp[i, :] = ortho_shoulder2_tmp[ifin, :] - ortho_shoulder2_tmp[ideb, :]

ortho_shoulder1_vel = ortho_shoulder1_vel_tmp * 1 / fps
ortho_shoulder2_vel = ortho_shoulder2_vel_tmp * 1 / fps

for i in range(ortho_shoulder1.shape[0]):
    if i == 0:
        ideb = 0
        ifin = 1
    else:
        ideb = i - 1
        ifin = i
    ortho_shoulder1_acc_tmp[i, :] = ortho_shoulder1_vel_tmp[ifin, :] - ortho_shoulder1_vel_tmp[ideb, :]
    ortho_shoulder2_acc_tmp[i, :] = ortho_shoulder2_vel_tmp[ifin, :] - ortho_shoulder2_vel_tmp[ideb, :]

ortho_shoulder1_acc = ortho_shoulder1_acc_tmp * 1 / fps
ortho_shoulder2_acc = ortho_shoulder2_acc_tmp * 1 / fps




if load3D:
    output_bodyFace_3Draw_hands_None      = np.column_stack((poseRaw_final, faceRaw_final))
    output_bodyFace_3Dfeatures_hands_None = np.column_stack((headAngles,
                                                             headAngles_vel,
                                                             headAngles_acc,
                                                             hand1_wrtPt1,
                                                             hand2_wrtPt1,
                                                             hand1_wrtPt1_vel,
                                                             hand2_wrtPt1_vel,
                                                             hand1_wrtPt1_acc,
                                                             hand2_wrtPt1_acc,
                                                             boucheGD_dist,
                                                             boucheHB_dist,
                                                             eyebrow1_wrtUpperNose_dist,
                                                             eyebrow2_wrtUpperNose_dist,
                                                             hand1_relElbow,
                                                             hand2_relElbow,
                                                             hand1_relElbow_vel,
                                                             hand2_relElbow_vel,
                                                             hand1_relElbow_acc,
                                                             hand2_relElbow_acc,
                                                             hand1_relNez,
                                                             hand2_relNez,
                                                             hand1_relNez_vel,
                                                             hand2_relNez_vel,
                                                             hand1_relNez_acc,
                                                             hand2_relNez_acc,
                                                             elbow1_relShoulder,
                                                             elbow2_relShoulder,
                                                             elbow1_relShoulder_vel,
                                                             elbow2_relShoulder_vel,
                                                             elbow1_relShoulder_acc,
                                                             elbow2_relShoulder_acc,
                                                             shoulder1_wrtPt1,
                                                             shoulder2_wrtPt1,
                                                             shoulder1_wrtPt1_vel,
                                                             shoulder2_wrtPt1_vel,
                                                             shoulder1_wrtPt1_acc,
                                                             shoulder2_wrtPt1_acc,
                                                             nez_wrtPt1,
                                                             hand1_vel_norm,
                                                             hand2_vel_norm,
                                                             hand1_acc_norm,
                                                             hand2_acc_norm,
                                                             hand1_rel_Hand2_vect,
                                                             hand1_rel_Hand2_vel,
                                                             hand1_rel_Hand2_acc,
                                                             hand1_rel_Hand2_dist,
                                                             cos_elbow1,
                                                             cos_elbow2,
                                                             cos_elbow1_vel,
                                                             cos_elbow2_vel,
                                                             cos_elbow1_acc,
                                                             cos_elbow2_acc,
                                                             ortho_elbow1,
                                                             ortho_elbow2,
                                                             ortho_elbow1_vel,
                                                             ortho_elbow2_vel,
                                                             ortho_elbow1_acc,
                                                             ortho_elbow2_acc,
                                                             cos_shoulder1,
                                                             cos_shoulder2,
                                                             cos_shoulder1_vel,
                                                             cos_shoulder2_vel,
                                                             cos_shoulder1_acc,
                                                             cos_shoulder2_acc,
                                                             ortho_shoulder1,
                                                             ortho_shoulder2,
                                                             ortho_shoulder1_vel,
                                                             ortho_shoulder2_vel,
                                                             ortho_shoulder1_acc,
                                                             ortho_shoulder2_acc))

    output_bodyFace_3Draw_hands_OP      = np.column_stack((hand2D_raw_final, output_bodyFace_3Draw_hands_None))
    output_bodyFace_3Dfeatures_hands_OP = np.column_stack((hand2D_raw_final, output_bodyFace_3Dfeatures_hands_None))
    if hsKoller:
        output_bodyFace_3Draw_hands_HS         = np.column_stack((final_handShapes, output_bodyFace_3Draw_hands_None))
        output_bodyFace_3Dfeatures_hands_HS    = np.column_stack((final_handShapes, output_bodyFace_3Dfeatures_hands_None))
        output_bodyFace_3Draw_hands_OP_HS      = np.column_stack((final_handShapes, hand2D_raw_final, poseRaw_final, faceRaw_final))
        output_bodyFace_3Dfeatures_hands_OP_HS = np.column_stack((final_handShapes, hand2D_raw_final, output_bodyFace_3Dfeatures_hands_None))
else:
    output_bodyFace_2Draw_hands_None      = np.column_stack((poseRaw_final, faceRaw_final))
    output_bodyFace_2Dfeatures_hands_None = np.column_stack((hand1_wrtPt1[:, 0::2],
                                                             hand2_wrtPt1[:, 0::2],
                                                             hand1_wrtPt1_vel[:, 0::2],
                                                             hand2_wrtPt1_vel[:, 0::2],
                                                             hand1_wrtPt1_acc[:, 0::2],
                                                             hand2_wrtPt1_acc[:, 0::2],
                                                             boucheGD_dist,
                                                             boucheHB_dist,
                                                             eyebrow1_wrtUpperNose_dist,
                                                             eyebrow2_wrtUpperNose_dist,
                                                             hand1_relElbow[:, 0::2],
                                                             hand2_relElbow[:, 0::2],
                                                             hand1_relElbow_vel[:, 0::2],
                                                             hand2_relElbow_vel[:, 0::2],
                                                             hand1_relElbow_acc[:, 0::2],
                                                             hand2_relElbow_acc[:, 0::2],
                                                             hand1_relNez[:, 0::2],
                                                             hand2_relNez[:, 0::2],
                                                             hand1_relNez_vel[:, 0::2],
                                                             hand2_relNez_vel[:, 0::2],
                                                             hand1_relNez_acc[:, 0::2],
                                                             hand2_relNez_acc[:, 0::2],
                                                             elbow1_relShoulder[:, 0::2],
                                                             elbow2_relShoulder[:, 0::2],
                                                             elbow1_relShoulder_vel[:, 0::2],
                                                             elbow2_relShoulder_vel[:, 0::2],
                                                             elbow1_relShoulder_acc[:, 0::2],
                                                             elbow2_relShoulder_acc[:, 0::2],
                                                             shoulder1_wrtPt1[:, 0::2],
                                                             shoulder2_wrtPt1[:, 0::2],
                                                             shoulder1_wrtPt1_vel[:, 0::2],
                                                             shoulder2_wrtPt1_vel[:, 0::2],
                                                             shoulder1_wrtPt1_acc[:, 0::2],
                                                             shoulder2_wrtPt1_acc[:, 0::2],
                                                             nez_wrtPt1[:, 0::2],
                                                             hand1_vel_norm[:, 0::2],
                                                             hand2_vel_norm[:, 0::2],
                                                             hand1_acc_norm[:, 0::2],
                                                             hand2_acc_norm[:, 0::2],
                                                             hand1_rel_Hand2_vect[:, 0::2],
                                                             hand1_rel_Hand2_vel[:, 0::2],
                                                             hand1_rel_Hand2_acc[:, 0::2],
                                                             hand1_rel_Hand2_dist,
                                                             cos_elbow1,
                                                             cos_elbow2,
                                                             cos_elbow1_vel,
                                                             cos_elbow2_vel,
                                                             cos_elbow1_acc,
                                                             cos_elbow2_acc,
                                                             cos_shoulder1,
                                                             cos_shoulder2,
                                                             cos_shoulder1_vel,
                                                             cos_shoulder2_vel,
                                                             cos_shoulder1_acc,
                                                             cos_shoulder2_acc))

    output_bodyFace_2Draw_hands_OP      = np.column_stack((hand2D_raw_final, output_bodyFace_2Draw_hands_None))
    output_bodyFace_2Dfeatures_hands_OP = np.column_stack((hand2D_raw_final, output_bodyFace_2Dfeatures_hands_None))
    if hsKoller:
        output_bodyFace_2Draw_hands_HS         = np.column_stack((final_handShapes, output_bodyFace_2Draw_hands_None))
        output_bodyFace_2Dfeatures_hands_HS    = np.column_stack((final_handShapes, output_bodyFace_2Dfeatures_hands_None))
        output_bodyFace_2Draw_hands_OP_HS      = np.column_stack((final_handShapes, hand2D_raw_final, output_bodyFace_2Draw_hands_None))
        output_bodyFace_2Dfeatures_hands_OP_HS = np.column_stack((final_handShapes, hand2D_raw_final, output_bodyFace_2Dfeatures_hands_None))


if load3D:
    np.save(path2features+'final/'+vidName+'_bodyFace_3D_raw_hands_None',      output_bodyFace_3Draw_hands_None)
    np.save(path2features+'final/'+vidName+'_bodyFace_3D_features_hands_None', output_bodyFace_3Dfeatures_hands_None)
    np.save(path2features+'final/'+vidName+'_bodyFace_3D_raw_hands_OP',        output_bodyFace_3Draw_hands_OP)
    np.save(path2features+'final/'+vidName+'_bodyFace_3D_features_hands_OP',   output_bodyFace_3Dfeatures_hands_OP)
    if hsKoller:
        np.save(path2features+'final/'+vidName+'_bodyFace_3D_raw_hands_HS',         output_bodyFace_3Draw_hands_HS)
        np.save(path2features+'final/'+vidName+'_bodyFace_3D_features_hands_HS',    output_bodyFace_3Dfeatures_hands_HS)
        np.save(path2features+'final/'+vidName+'_bodyFace_3D_raw_hands_OP_HS',      output_bodyFace_3Draw_hands_OP_HS)
        np.save(path2features+'final/'+vidName+'_bodyFace_3D_features_hands_OP_HS', output_bodyFace_3Dfeatures_hands_OP_HS)
else:
    np.save(path2features+'final/'+vidName+'_bodyFace_2D_raw_hands_None',      output_bodyFace_2Draw_hands_None)
    np.save(path2features+'final/'+vidName+'_bodyFace_2D_features_hands_None', output_bodyFace_2Dfeatures_hands_None)
    np.save(path2features+'final/'+vidName+'_bodyFace_2D_raw_hands_OP',        output_bodyFace_2Draw_hands_OP)
    np.save(path2features+'final/'+vidName+'_bodyFace_2D_features_hands_OP',   output_bodyFace_2Dfeatures_hands_OP)
    if hsKoller:
        np.save(path2features+'final/'+vidName+'_bodyFace_2D_raw_hands_HS',         output_bodyFace_2Draw_hands_HS)
        np.save(path2features+'final/'+vidName+'_bodyFace_2D_features_hands_HS',    output_bodyFace_2Dfeatures_hands_HS)
        np.save(path2features+'final/'+vidName+'_bodyFace_2D_raw_hands_OP_HS',      output_bodyFace_2Draw_hands_OP_HS)
        np.save(path2features+'final/'+vidName+'_bodyFace_2D_features_hands_OP_HS', output_bodyFace_2Dfeatures_hands_OP_HS)
