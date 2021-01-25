# Feature extractor for cslr_limsi

All scripts have to be run from the main folder (`cslr_limsi_features`).

## Main script
### **`scripts/main.sh`**
  - Runs all scripts for all videos inside `videos/`
  - Parameters:
    - `--keep_full_frames`: if you want not to delete full frames after all features are computed
    - `--keep_hand_crop_frames`: if you want not to delete hand crop frames after all features are computed
  - Outputs:
    - See detail of other scripts

## Included scripts
### **`scripts/video_to_frames.sh`**
  - Converts any video to frames, in a separate folder inside `frames/full/`
  - Parameters:
    - `videoName`
  - Outputs:
    - `frames/full/videoName/00001.jpg`
    - ...
    - `frames/full/videoName/07342.jpg` (if the video contains 7342 frames)
### **`scripts/video_to_openpose_json.sh`**
  - Converts any video to openpose data, in a separate folder inside `features/openpose/`
  - Parameters:
    - `videoName`
  - Outputs:
    - `features/openpose/videoName/keypoints_json000000000000.json`
    - ...
    - `features/openpose/videoName/keypoints_json000000007341.json` (if the video contains 7342 frames)
### **`scripts/frames_to_3DFace.sh`**
  - Converts all frames of any video to a numpy file containing the 3D coordinates of face landmarks (Adrian Bulat's FaceAlignment model). Data is centered around the mid-point between eyes, and normalized by the average distance between eyes.
  - Parameters:
    - `videoName`
  - Outputs:
    - `features/final/videoName_3DFace_predict_raw.npy`
### **`scripts/openpose_json_to_clean_numpy_and_hand_crops.sh`**
  - Cleans openpose data of any video, generate hand crop images (in a separate folder inside `frames/hand/`) and outputs several numpy files
  - Parameters:
    - `videoName`
  - Outputs:
    - `frames/hand/videoName/00001_hand1.jpg`
    - `frames/hand/videoName/00001_hand2.jpg`
    - ...
    - `frames/hand/videoName/07342_hand1.jpg`
    - `frames/hand/videoName/07342_hand2.jpg` (if the video contains 7342 frames)
    - `features/final/videoName_2DBody_OP_raw.npy`
    - `features/final/videoName_3DBody_predict_raw.npy`: 3D body estimate from model trained on LSF Mocap data, predicted from 2D openpose data
    - `features/final/videoName_2DFace_OP_raw.npy`
    - `features/final/videoName_2DHand1_OP_raw.npy`
    - `features/final/videoName_2DHand2_OP_raw.npy`
### **`scripts/3DFace_raw_to_headAngles.sh`**
  - Uses 3DFace data of any video to generate a numpy file containing the 3 Euler angles for the rotation of the head
  - Parameters:
    - `videoName`
  - Outputs:
    - `features/final/videoName_headAngles.npy`
### **`scripts/hand_crops_to_HS_probabilities.sh`**
  - Computes Koller's model probabilities for 61 hand shapes, for each frame and each hand of a given video
  - Parameters:
    - `videoName`
  - Outputs:
    - `features/final/videoName_HShand1.npy`
    - `features/final/videoName_HShand2.npy`
