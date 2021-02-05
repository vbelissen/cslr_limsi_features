import numpy as np
import sys
import os, fnmatch


path2vid      = sys.argv[1]
path2features = sys.argv[2]
vidSuffix     = sys.argv[3]

videoExtensions = ['.mp4', '.mov', '.mpg', '.mpeg', '.avi', '.flv', '.mkv', '.webm', '.wmv', '.ogg', '.3gp']

nVideo  = 0
nFrames = 0
listOfFiles = os.listdir(path2vid)
for entry in listOfFiles:
    filePart1, filePart2 = os.path.splitext(entry)
    if filePart2 in videoExtensions:
        print('Summing all data')
        print(filePart1)
        if nVideo == 0:
            (N_frames, N_features) = (np.load(path2features+'final/'+filePart1+'_'+vidSuffix+'.npy')).shape
            avg       = np.zeros(N_features)
            avgSquare = np.zeros(N_features)
        data = np.load(path2features+'final/'+filePart1+'_'+vidSuffix+'.npy')
        avg       += np.nansum(data,            axis=0)
        avgSquare += np.nansum(np.square(data), axis=0)
        nVideo  += 1
        nFrames += N_frames

print('')

avg       = avg/nFrames
avgSquare = avgSquare/nFrames

stDev = np.sqrt(avgSquare - np.square(avg))

np.save(path2features+'final/'+vidSuffix+'-AVERAGE.npy', avg)
np.save(path2features+'final/'+vidSuffix+'-STDEV.npy', stDev)

for entry in listOfFiles:
    filePart1, filePart2 = os.path.splitext(entry)
    if filePart2 in videoExtensions:
        print('Normalizing all data')
        print(filePart1)
        data = np.load(path2features+'final/'+filePart1+'_'+vidSuffix+'.npy')
        data = (data-avg)/stDev
        np.save(path2features+'final/'+filePart1+'_'+vidSuffix+'_normalized.npy', data)
#np.save(path2features+'final/'+vidName+'_bodyFace_3D_raw_hands_None',      output_bodyFace_3Draw_hands_None)
