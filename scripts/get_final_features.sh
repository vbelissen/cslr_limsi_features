#!/bin/bash

CLEAR='\033[0m'
RED='\033[0;31m'

function usage() {
  if [ -n "$1" ]; then
    echo -e "${RED}👉 $1${CLEAR}\n";
  fi
  echo "Usage: $0 [-v vidName] [--fps] [--load3D] [--hs]"
  echo "  -v, --vidName            Video name without extension"
  echo "  --fps                    Framerate per second"
  echo "  --load3D                 3D Body and Face computed too"
  echo "  --hs                     Hand Shapes computed (Koller caffe model)"
  echo ""
  echo "Example: $0 -v test_video_1 --fps 25 --load3D --hs"
  exit 1
}

# default params values
LOAD3D=0
HS=0

# parse params
while [[ "$#" > 0 ]]; do case $1 in
  -v|--vidName) VIDNAME="$2"; shift;shift;;
  --fps) FPS="$2"; shift;shift;;
  --load3D) LOAD3D=1; shift;;
  --hs) HS=1; shift;;
  *) usage "Unknown parameter passed: $1"; shift; shift;;
esac; done

if [[ "$LOAD3D" = 1 ]]; then echo "(3D features)"; else echo "(2D features)"; fi;

# verify params
if [ -z "$VIDNAME" ]; then usage "Video name is not set"; fi;
if [ -z "$FPS" ]; then usage "Framerate is not set"; fi;

path2vid=`cat scripts/paths/path_to_videos.txt`
path2features=`cat scripts/paths/path_to_features.txt`
path2utils=`cat scripts/paths/path_to_utils.txt`
path2frames=`cat scripts/paths/path_to_frames.txt`

vEnv=`cat scripts/virtual_env_names/vEnv_for_final_features.txt`

nImg=$(ls "${path2frames}${VIDNAME}/" | wc -l)

source activate ${vEnv}
python "${path2utils}final_features.py" ${nImg} ${VIDNAME} ${FPS} ${path2features} ${LOAD3D} ${HS}
source deactivate
