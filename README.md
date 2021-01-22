# Feature extractor for cslr_limsi

## Included scripts
- `scripts/video_to_frames.sh`: converts any video to frames, in a separate folder inside `frames/full/`
- `scripts/video_to_openpose_json.sh`: converts any video to openpose data, in a separate folder inside `features/openpose/`
- `scripts/frames_to_3DFace.sh`: converts all frames of any video to a numpy file (`videoName_3DFace_raw.npy`) containing the 3D coordinates of face landmarks (Adrian Bulat's FaceAlignment model). Data is centered around the mid-point between eyes, and normalized by the average distance between eyes
- `scripts/openpose_json_to_clean_numpy.sh`: cleans openpose data of any video, generate hand crop images (in a separate folder inside `frames/hand/`) and outputs several numpy files:
  - `videoName_2DBody_raw.npy`
  - `videoName_3DBody_raw.npy`: 3D body estimate from model trained on LSF Mocap data, predicted from 2D openpose data
  - `videoName_2DFace_raw.npy`
  - `videoName_2DHand1_raw.npy`
  - `videoName_2DHand2_raw.npy`
- `scripts/3DFace_raw_to_headAngles.sh`: uses 3DFace data of any video to generate a numpy file (`videoName_headAngles.npy`) containing the 3 Euler angles for the rotation of the head
