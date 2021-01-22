# Feature extractor for cslr_limsi

## Included scripts
- `scripts/video_to_frames.sh`: converts any video to frames, in a separate folder inside `frames/full/`
- `scripts/video_to_openpose_json.sh`: converts any video to openpose data, in a separate folder inside `features/openpose/`
- `scripts/frames_to_3DFace.sh`: converts all frames of any video to a numpy file (`videoName_3DFace_raw.npy`) containing the 3D coordinates of face landmarks (Adrian Bulat's FaceAlignment model)
- `scripts/openpose_json_and_3Dface_to_clean_numpy.sh`: cleans openpose data of any video, generate hand crop images and outputs several numpy files:
  - `videoName_2Draw.npy`
  - `videoName_3Draw.npy`
- `scripts/3DFace_to_headAngles.sh`: uses 3DFace data of any video to generate a numpy file (`videoName_headAngles.npy`) containing the 3 Euler angles for the rotation of the head
