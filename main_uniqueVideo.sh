#!/bin/bash

echo "Main script"

CLEAR='\033[0m'
RED='\033[0;31m'

function usage() {
  if [ -n "$1" ]; then
    echo -e "${RED}👉 $1${CLEAR}\n";
  fi
  echo "Usage: $0 [-v vidName] [--vidExt] [--framesExt] [-n nDigits] [--body3D] [--face3D] [--hs] [--keep_full_frames] [--keep_hand_crop_frames] [--keep_openpose_json] [--keep_temporary_features]"
  echo "  -v, --vidName             Video name without extension"
  echo "  --vidExt                  Video file extension"
  echo "  --framesExt               Frame files extension"
  echo "  -n, --nDigits             Number of digits for frame numbering"
  echo "  --body3D                  3D Body computed too"
  echo "  --face3D                  3D Face computed too"
  echo "  --hs                      Hand Shapes (Koller caffe model)"
  echo "  --keep_full_frames        For not deleting full frames         (optional, default=0)"
  echo "  --keep_hand_crop_frames   For not deleting hand crop frames    (optional, default=0)"
  echo "  --keep_openpose_json      For not deleting openpose json files (optional, default=0)"
  echo "  --keep_temporary_features For not deleting temporary features (optional, default=0)"
  echo ""
  echo "Example: $0 -v test_video_1 --vidExt mp4 --framesExt jpg -n 5 --body3D --face3D --hs --keep_full_frames --keep_hand_crop_frames --keep_openpose_json --keep_temporary_features"
  exit 1
}

# default params values
BODY3D=false
FACE3D=false
HS=false
KEEP_FULL_FRAMES=false
KEEP_HAND_CROP_FRAMES=false
KEEP_OPENPOSE_JSON=false
KEEP_TEMPORARY_FEATURES=false

BODY3D_STRING=""
FACE3D_STRING=""
HS_STRING=""

# parse params
while [[ "$#" > 0 ]]; do case $1 in
  -v|--vidName) VIDNAME="$2"; shift;shift;;
  --vidExt) VIDEXT="$2"; shift;shift;;
  --framesExt) FRAMESEXT="$2"; shift;shift;;
  -n|--nDigits) NDIGITS="$2"; shift;shift;;
  --body3D) BODY3D=true; shift;;
  --face3D) FACE3D=true; shift;;
  --hs) HS=true; shift;;
  --keep_full_frames) KEEP_FULL_FRAMES=true; shift;;
  --keep_hand_crop_frames) KEEP_HAND_CROP_FRAMES=true; shift;;
  --keep_openpose_json) KEEP_OPENPOSE_JSON=true; shift;;
  --keep_temporary_features) KEEP_TEMPORARY_FEATURES=true; shift;;
  *) usage "Unknown parameter passed: $1"; shift; shift;;
esac; done

if [[ "$BODY3D" = true ]]; then BODY3D_STRING=" --body3D"; fi;
if [[ "$FACE3D" = true ]]; then FACE3D_STRING=" --face3D"; fi;
if [[ "$HS" = true ]]; then HS_STRING=" --hs"; fi;

# verify params
if [ -z "$VIDNAME" ]; then usage "Video name is not set"; fi;
if [ -z "$VIDEXT" ]; then usage "Video extension is not set."; fi;
if [ -z "$FRAMESEXT" ]; then usage "Frames extension is not set."; fi;
if [ -z "$NDIGITS" ]; then usage "Number of digits for frame numbering is not set."; fi;

if [[ "$BODY3D" = true ]] && [[ "$FACE3D" = true ]]; then
  LOAD3D=true
else
  LOAD3D=false
fi

path2frames=`cat scripts/paths/path_to_frames.txt`
path2handFrames=`cat scripts/paths/path_to_hand_frames.txt`
path2openpose=`cat scripts/paths/path_to_openpose.txt`
path2features=`cat scripts/paths/path_to_features.txt`

./scripts/video_to_frames.sh -v ${VIDNAME} --vidExt ${VIDEXT} --framesExt ${FRAMESEXT} -n ${NDIGITS}
./scripts/video_to_openpose.sh -v ${VIDNAME} --vidExt ${VIDEXT}
if [[ "$FACE3D" = true ]]; then ./scripts/frames_to_3DFace_temp.sh -v ${VIDNAME} --framesExt ${FRAMESEXT} -n ${NDIGITS}; fi;
./scripts/openpose_json_to_clean_data.sh -v ${VIDNAME}
./scripts/openpose_clean_to_hand_crops.sh -v ${VIDNAME} --framesExt ${FRAMESEXT} -n ${NDIGITS}
./scripts/openpose_clean_to_2D_3D.sh -v ${VIDNAME}${BODY3D_STRING}${FACE3D_STRING}
if [[ "$HS" = true ]]; then ./scripts/hand_crops_to_HS_probabilities.sh -v ${VIDNAME} -n ${NDIGITS}; fi;
./scripts/get_final_features.sh -v ${VIDNAME}${HS_STRING}
if [[ "$LOAD3D" = true ]]; then ./scripts/get_final_features.sh -v ${VIDNAME} --load3D${HS_STRING}; fi;

if [[ "$KEEP_FULL_FRAMES" = false ]]; then rm -rf ${path2frames}${VIDNAME}; fi;
if [[ "$KEEP_HAND_CROP_FRAMES" = false ]]; then rm -rf ${path2handFrames}${VIDNAME}; fi;
if [[ "$KEEP_OPENPOSE_JSON" = false ]]; then rm -rf ${path2openpose}${VIDNAME}; fi;
if [[ "$KEEP_TEMPORARY_FEATURES" = false ]]; then rm "${path2features}temp/${VIDNAME}"*; fi;
