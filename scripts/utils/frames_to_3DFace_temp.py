import face_alignment
from skimage import io
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import sys

# Fonction pour gerer les nan
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

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=True, flip_input=False)

nimg          = int(sys.argv[1])
path2frames   = sys.argv[2]
vidName       = sys.argv[3]
framesExt     = sys.argv[4]
nDigits       = int(sys.argv[5])
path2features = sys.argv[6]

tabTot = np.zeros((68, 3, nimg))

for i in range(nimg):
    input = io.imread(path2frames+vidName+'/'+str(i+1).zfill(nDigits)+'.'+framesExt)
    preds = fa.get_landmarks(input)
    if preds != None:
        tabTot[:,:,i] = preds[0]
    else:
        tabTot[:,:,i] = np.nan


savitzky_window = 17
savitzky_order = 6

for j in range(3):
    for k in range(68):
        nans, x = nan_helper(tabTot[k,j,:])
        if tabTot[k,j,~nans].size != 0:
            f2 = interp1d(x(~nans), tabTot[k,j,~nans], kind='linear',bounds_error=False)
            tabTot[k,j,:]= f2(range(nimg))
        tabTot[k,j,:] = signal.savgol_filter(tabTot[k,j,:], savitzky_window, savitzky_order)
        #tabTot[k,j,nans] = np.nan

# Forcer le visage 3D a etre de taille constante
pts1 = [0,  1,15,16]
pts2 = [27,28,29,30]

tailleGlob = np.zeros(nimg)
for p1 in pts1:
    for p2 in pts2:
        p1p2 = tabTot[p2,:,:] - tabTot[p1,:,:]
        dist = np.sqrt(np.sum(np.square(p1p2),axis=0))
        tailleGlob += dist

tailleGlob_moy = np.mean(tailleGlob)
tailleGlob_Red = tailleGlob/tailleGlob_moy

for i in range(nimg):
    tabTot[:,:,i] = tabTot[:,:,i]/tailleGlob_Red[i]

np.save(path2features+'final/'+vidName+'_3DFace_predict_raw_temp', tabTot)
