#!/bin/bash

echo "Cleaning and assembling openpose json files to a unique numpy array"

CLEAR='\033[0m'
RED='\033[0;31m'

function usage() {
  if [ -n "$1" ]; then
    echo -e "${RED}👉 $1${CLEAR}\n";
  fi
  echo "Usage: $0 [-v vidName] [-h, --handOP] [-f, --faceOP]"
  echo "  -v, --vidName            Video name without extension"
  echo "  -h, --hand               OpenPose computed on hands too"
  echo "  -f, --face               OpenPose computed on face too"
  echo ""
  echo "Example: $0 -v test_video_1 -h -f"
  exit 1
}

# default params values
HAND=false
FACE=false

# parse params
while [[ "$#" > 0 ]]; do case $1 in
  -v|--vidName) VIDNAME="$2"; shift;shift;;
  -h|--handOP) HAND=true; shift;;
  -f|--faceOP) FACE=true; shift;;
  *) usage "Unknown parameter passed: $1"; shift; shift;;
esac; done

# verify params
if [ -z "$VIDNAME" ]; then usage "Video name is not set"; fi;

path2features=`cat scripts/paths/path_to_features.txt`
path2utils=`cat scripts/paths/path_to_utils.txt`
path2frames=`cat scripts/paths/path_to_frames.txt`

vEnv=`cat scripts/virtual_env_names/vEnv_for_clean_numpy.txt`

nImg=$(ls "${path2frames}${VIDNAME}/" | wc -l)

if [[ "$HAND" = true ]] && [[ "$FACE" = true ]]; then
    typeData="pfh"
elif [[ "$HAND" = true ]]; then
    typeData="ph"
elif [[ "$FACE" = true ]]; then
    typeData="pf"
else
    typeData="p"
fi

source activate ${vEnv}
python "${path2utils}openpose_json_to_clean_data.py" ${nImg} ${VIDNAME} ${path2features} ${typeData}
source deactivate
