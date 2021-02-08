import numpy as np
import sys
import os, fnmatch


path2vid      = sys.argv[1]
path2features = sys.argv[2]
vidSuffix     = sys.argv[3]

videoExtensions = ['.mp4', '.mov', '.mpg', '.mpeg', '.avi', '.flv', '.mkv', '.webm', '.wmv', '.ogg', '.3gp']

Untouched_features = {
'bodyFace_2D_raw_hands_None': np.array([]),
'bodyFace_2D_features_hands_None': np.array([]),
'bodyFace_2D_raw_hands_OP': np.sort(np.concatenate([np.arange(0,0+3*21,3), np.arange(61,61+3*21,3)])),
'bodyFace_2D_features_hands_OP': np.sort(np.concatenate([np.arange(0,0+3*21,3), np.arange(61,61+3*21,3)])),
'bodyFace_2D_raw_hands_HS': np.arange(0,122),
'bodyFace_2D_features_hands_HS': np.arange(0,122),
'bodyFace_2D_raw_hands_OP_HS': np.sort(np.concatenate([np.arange(0,122), np.arange(122,122+3*21,3), np.arange(183,183+3*21,3)])),
'bodyFace_2D_features_hands_OP_HS': np.sort(np.concatenate([np.arange(0,122), np.arange(122,122+3*21,3), np.arange(183,183+3*21,3)])),
'bodyFace_3D_raw_hands_None': np.array([]),
'bodyFace_3D_features_hands_None': np.array([]),
'bodyFace_3D_raw_hands_OP': np.sort(np.concatenate([np.arange(0,0+3*21,3), np.arange(61,61+3*21,3)])),
'bodyFace_3D_features_hands_OP': np.sort(np.concatenate([np.arange(0,0+3*21,3), np.arange(61,61+3*21,3)])),
'bodyFace_3D_raw_hands_HS': np.arange(0,122),
'bodyFace_3D_features_hands_HS': np.arange(0,122),
'bodyFace_3D_raw_hands_OP_HS': np.sort(np.concatenate([np.arange(0,122), np.arange(122,122+3*21,3), np.arange(183,183+3*21,3)])),
'bodyFace_3D_features_hands_OP_HS': np.sort(np.concatenate([np.arange(0,122), np.arange(122,122+3*21,3), np.arange(183,183+3*21,3)]))
}

N_features = {
'bodyFace_2D_raw_hands_None': 168,
'bodyFace_2D_features_hands_None': 93,
'bodyFace_2D_raw_hands_OP': 290,
'bodyFace_2D_features_hands_OP': 215,
'bodyFace_2D_raw_hands_HS': 290,
'bodyFace_2D_features_hands_HS': 215,
'bodyFace_2D_raw_hands_OP_HS': 412,
'bodyFace_2D_features_hands_OP_HS': 337,
'bodyFace_3D_raw_hands_None': 246,
'bodyFace_3D_features_hands_None': 176,
'bodyFace_3D_raw_hands_OP': 368,
'bodyFace_3D_features_hands_OP': 298,
'bodyFace_3D_raw_hands_HS': 368,
'bodyFace_3D_features_hands_HS': 298,
'bodyFace_3D_raw_hands_OP_HS': 490,
'bodyFace_3D_features_hands_OP_HS': 420
}

#nVideo  = 0
nFrames = 0

nFeatures = N_features[vidSuffix]
avg       = np.zeros(nFeatures)
avgSquare = np.zeros(nFeatures)

print('Summing all videos')
listOfFiles = os.listdir(path2vid)
for entry in listOfFiles:
    filePart1, filePart2 = os.path.splitext(entry)
    if filePart2 in videoExtensions:
        print(filePart1)
        #if nVideo == 0:
        #    (N_frames, N_features) = (np.load(path2features+'final/'+filePart1+'_'+vidSuffix+'.npy')).shape

        data = np.load(path2features+'final/'+filePart1+'_'+vidSuffix+'.npy')
        N_frames = data.shape[0]
        avg       += np.nansum(data,            axis=0)
        avgSquare += np.nansum(np.square(data), axis=0)
        #nVideo  += 1
        nFrames += N_frames


avg       = avg/nFrames
avgSquare = avgSquare/nFrames

stDev = np.sqrt(avgSquare - np.square(avg))

untouchedFeatures = Untouched_features[vidSuffix]
print(untouchedFeatures)
if untouchedFeatures.size > 0:
    avg[untouchedFeatures]   = 0
    stDev[untouchedFeatures] = 1

np.save(path2features+'final/'+vidSuffix+'-AVERAGE.npy', avg)
np.save(path2features+'final/'+vidSuffix+'-STDEV.npy', stDev)

print('Normalizing all videos')
for entry in listOfFiles:
    filePart1, filePart2 = os.path.splitext(entry)
    if filePart2 in videoExtensions:
        print(filePart1)
        data = np.load(path2features+'final/'+filePart1+'_'+vidSuffix+'.npy')
        data = (data-avg)/stDev
        np.save(path2features+'final/'+filePart1+'_'+vidSuffix+'_normalized.npy', data)
#np.save(path2features+'final/'+vidName+'_bodyFace_3D_raw_hands_None',      output_bodyFace_3Draw_hands_None)
