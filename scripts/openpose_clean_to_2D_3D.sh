#!/bin/bash

echo "Getting coherent 2D/3D data for body/face/hands from openpose cleaned file, prediction model and 3D face estimation"

CLEAR='\033[0m'
RED='\033[0;31m'

function usage() {
  if [ -n "$1" ]; then
    echo -e "${RED}👉 $1${CLEAR}\n";
  fi
  echo "Usage: $0 [-v vidName] [--handOP] [--faceOP] [--body3D] [--face3D]"
  echo "  -v, --vidName            Video name without extension"
  echo "  --handOP                 OpenPose computed on hands too"
  echo "  --faceOP                 OpenPose computed on face too"
  echo "  --body3D                 3D Body computed too"
  echo "  --face3D                 3D Face computed too"
  echo ""
  echo "Example: $0 -v test_video_1 --handOP --faceOP --body3D --face3D "
  exit 1
}

# default params values
HANDOP=false
FACEOP=false
BODY3D=false
FACE3D=false

# parse params
while [[ "$#" > 0 ]]; do case $1 in
  -v|--vidName) VIDNAME="$2"; shift;shift;;
  --handOP) HANDOP=true; shift;;
  --faceOP) FACEOP=true; shift;;
  --body3D) BODY3D=true; shift;;
  --face3D) FACE3D=true; shift;;
  *) usage "Unknown parameter passed: $1"; shift; shift;;
esac; done

# verify params
if [ -z "$VIDNAME" ]; then usage "Video name is not set"; fi;

path2vid=`cat scripts/paths/path_to_videos.txt`
path2features=`cat scripts/paths/path_to_features.txt`
path2utils=`cat scripts/paths/path_to_utils.txt`
path2frames=`cat scripts/paths/path_to_frames.txt`
path2handFrames=`cat scripts/paths/path_to_hand_frames.txt`

vEnv=`cat scripts/virtual_env_names/vEnv_for_2D_3D.txt`

nImg=$(ls "${path2frames}${VIDNAME}/" | wc -l)

conda activate ${vEnv}
python "${path2utils}openpose_clean_to_2D_3D.py" ${nImg} ${VIDNAME} ${path2features} ${HANDOP} ${FACEOP} ${BODY3D} ${FACE3D}
conda deactivate

if [[ "$FACE3D" = true ]]; then rm "${path2features}final/${VIDNAME}_3DFace_predict_raw_temp.npy"; fi;
