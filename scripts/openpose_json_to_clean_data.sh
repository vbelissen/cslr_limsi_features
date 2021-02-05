#!/bin/bash

CLEAR='\033[0m'
RED='\033[0;31m'

function usage() {
  if [ -n "$1" ]; then
    echo -e "${RED}👉 $1${CLEAR}\n";
  fi
  echo "Usage: $0 [-v vidName]"
  echo "  -v, --vidName            Video name without extension"
  echo ""
  echo "Example: $0 -v test_video_1"
  exit 1
}


# parse params
while [[ "$#" > 0 ]]; do case $1 in
  -v|--vidName) VIDNAME="$2"; shift;shift;;
  *) usage "Unknown parameter passed: $1"; shift; shift;;
esac; done

# verify params
if [ -z "$VIDNAME" ]; then usage "Video name is not set"; fi;

path2features=`cat scripts/paths/path_to_features.txt`
path2utils=`cat scripts/paths/path_to_utils.txt`
path2frames=`cat scripts/paths/path_to_frames.txt`

vEnv=`cat scripts/virtual_env_names/vEnv_for_clean_numpy.txt`

nImg=$(ls "${path2frames}${VIDNAME}/" | wc -l)

source activate ${vEnv}
python "${path2utils}openpose_json_to_clean_data.py" ${nImg} ${VIDNAME} ${path2features}
source deactivate
