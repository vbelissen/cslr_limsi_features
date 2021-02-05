#!/bin/bash

CLEAR='\033[0m'
RED='\033[0;31m'

function usage() {
  if [ -n "$1" ]; then
    echo -e "${RED}👉 $1${CLEAR}\n";
  fi
  echo "Usage: $0  [--body3D] [--face3D] [--hs]"
  echo "  --body3D      3D Body computed too"
  echo "  --face3D      3D Face computed too"
  echo "  --hs          Hand Shapes (Koller caffe model)"
  echo ""
  echo "Example: $0 --body3D --face3D --hs"
  exit 1
}

# default params values
BODY3D=0
FACE3D=0
HS=0

BODY3D_STRING=""
FACE3D_STRING=""
HS_STRING=""

# parse params
while [[ "$#" > 0 ]]; do case $1 in
--body3D) BODY3D=1; shift;;
--face3D) FACE3D=1; shift;;
--hs) HS=1; shift;;
  *) usage "Unknown parameter passed: $1"; shift; shift;;
esac; done

if [[ "$BODY3D" = 1 ]] && [[ "$FACE3D" = 1 ]]; then
  LOAD3D=1
else
  LOAD3D=0
fi

path2vid=`cat scripts/paths/path_to_videos.txt`
path2features=`cat scripts/paths/path_to_features.txt`
path2utils=`cat scripts/paths/path_to_utils.txt`
path2frames=`cat scripts/paths/path_to_frames.txt`

vEnv=`cat scripts/virtual_env_names/vEnv_for_final_features.txt`

nImg=$(ls "${path2frames}${VIDNAME}/" | wc -l)

source activate ${vEnv}
echo "Normalizing bodyFace_2D_raw_hands_None data"
python "${path2utils}normalize_features.py" ${path2vid} ${path2features} "bodyFace_2D_raw_hands_None"
echo "Normalizing bodyFace_2D_raw_hands_OP data"
python "${path2utils}normalize_features.py" ${path2vid} ${path2features} "bodyFace_2D_raw_hands_OP"
echo "Normalizing bodyFace_2D_features_hands_None data"
python "${path2utils}normalize_features.py" ${path2vid} ${path2features} "bodyFace_2D_features_hands_None"
echo "Normalizing bodyFace_2D_features_hands_OP data"
python "${path2utils}normalize_features.py" ${path2vid} ${path2features} "bodyFace_2D_features_hands_OP"
if [[ "$HS" = 1 ]]; then
  echo "Normalizing bodyFace_2D_raw_hands_HS data"
  python "${path2utils}normalize_features.py" ${path2vid} ${path2features} "bodyFace_2D_raw_hands_HS"
  echo "Normalizing bodyFace_2D_raw_hands_OP_HS data"
  python "${path2utils}normalize_features.py" ${path2vid} ${path2features} "bodyFace_2D_raw_hands_OP_HS"
  echo "Normalizing bodyFace_2D_features_hands_HS data"
  python "${path2utils}normalize_features.py" ${path2vid} ${path2features} "bodyFace_2D_features_hands_HS"
  echo "Normalizing bodyFace_2D_features_hands_OP_HS data"
  python "${path2utils}normalize_features.py" ${path2vid} ${path2features} "bodyFace_2D_features_hands_OP_HS"
fi
if [[ "$LOAD3D" = 1 ]]; then
  echo "Normalizing bodyFace_3D_raw_hands_None data"
  python "${path2utils}normalize_features.py" ${path2vid} ${path2features} "bodyFace_3D_raw_hands_None"
  echo "Normalizing bodyFace_3D_raw_hands_OP data"
  python "${path2utils}normalize_features.py" ${path2vid} ${path2features} "bodyFace_3D_raw_hands_OP"
  echo "Normalizing bodyFace_3D_features_hands_None data"
  python "${path2utils}normalize_features.py" ${path2vid} ${path2features} "bodyFace_3D_features_hands_None"
  echo "Normalizing bodyFace_3D_features_hands_OP data"
  python "${path2utils}normalize_features.py" ${path2vid} ${path2features} "bodyFace_3D_features_hands_OP"
  if [[ "$HS" = 1 ]]; then
    echo "Normalizing bodyFace_3D_raw_hands_HS data"
    python "${path2utils}normalize_features.py" ${path2vid} ${path2features} "bodyFace_3D_raw_hands_HS"
    echo "Normalizing bodyFace_3D_raw_hands_OP_HS data"
    python "${path2utils}normalize_features.py" ${path2vid} ${path2features} "bodyFace_3D_raw_hands_OP_HS"
    echo "Normalizing bodyFace_3D_features_hands_HS data"
    python "${path2utils}normalize_features.py" ${path2vid} ${path2features} "bodyFace_3D_features_hands_HS"
    echo "Normalizing bodyFace_3D_features_hands_OP_HS data"
    python "${path2utils}normalize_features.py" ${path2vid} ${path2features} "bodyFace_3D_features_hands_OP_HS"
  fi
fi
source deactivate
