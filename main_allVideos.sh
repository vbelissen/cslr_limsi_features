#!/bin/bash

echo "*****************************************************************"
echo "-----------------------------------------------------------------"
echo ""
echo "   MAIN SCRIPT - GETTING ALL FEATURES FOR ALL VIDEOS IN FOLDER"
echo ""
echo "-----------------------------------------------------------------"
echo "*****************************************************************"
echo ""

CLEAR='\033[0m'
RED='\033[0;31m'

function usage() {
  if [ -n "$1" ]; then
    echo -e "${RED}👉 $1${CLEAR}\n";
  fi
  echo "Usage: $0 [--fps] [--framesExt] [-n nDigits] [--body3D] [--face3D] [--hs] [--keep_full_frames] [--keep_hand_crop_frames] [--keep_openpose_json] [--keep_temporary_features]"
  echo "  --fps                     Framerate per second"
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
  echo "Example: $0 --fps --framesExt jpg -n 5 --body3D --face3D --hs --keep_full_frames --keep_hand_crop_frames --keep_openpose_json --keep_temporary_features"
  exit 1
}

# default params values
BODY3D=0
FACE3D=0
HS=0
KEEP_FULL_FRAMES=0
KEEP_HAND_CROP_FRAMES=0
KEEP_OPENPOSE_JSON=0
KEEP_TEMPORARY_FEATURES=0

BODY3D_STRING=""
FACE3D_STRING=""
HS_STRING=""
KEEP_FULL_FRAMES_STRING=""
KEEP_HAND_CROP_FRAMES_STRING=""
KEEP_OPENPOSE_JSON_STRING=""
KEEP_TEMPORARY_FEATURES_STRING=""

# parse params
while [[ "$#" > 0 ]]; do case $1 in
  --fps) FPS="$2"; shift;shift;;
  --framesExt) FRAMESEXT="$2"; shift;shift;;
  -n|--nDigits) NDIGITS="$2"; shift;shift;;
  --body3D) BODY3D=1; shift;;
  --face3D) FACE3D=1; shift;;
  --hs) HS=1; shift;;
  --keep_full_frames) KEEP_FULL_FRAMES=1; shift;;
  --keep_hand_crop_frames) KEEP_HAND_CROP_FRAMES=1; shift;;
  --keep_openpose_json) KEEP_OPENPOSE_JSON=1; shift;;
  --keep_temporary_features) KEEP_TEMPORARY_FEATURES=1; shift;;
  *) usage "Unknown parameter passed: $1"; shift; shift;;
esac; done

if [[ "$BODY3D" = 1 ]]; then BODY3D_STRING=" --body3D"; fi;
if [[ "$FACE3D" = 1 ]]; then FACE3D_STRING=" --face3D"; fi;
if [[ "$HS" = 1 ]]; then HS_STRING=" --hs"; fi;
if [[ "$KEEP_FULL_FRAMES" = 1 ]]; then KEEP_FULL_FRAMES_STRING=" --keep_full_frames"; fi;
if [[ "$KEEP_HAND_CROP_FRAMES" = 1 ]]; then KEEP_HAND_CROP_FRAMES_STRING=" --keep_hand_crop_frames"; fi;
if [[ "$KEEP_OPENPOSE_JSON" = 1 ]]; then KEEP_OPENPOSE_JSON_STRING=" --keep_openpose_json"; fi;
if [[ "$KEEP_TEMPORARY_FEATURES" = 1 ]]; then KEEP_TEMPORARY_FEATURES_STRING=" --keep_temporary_features"; fi;

# verify params
if [ -z "$FPS" ]; then usage "Framerate is not set"; fi;
if [ -z "$FRAMESEXT" ]; then usage "Frames extension is not set."; fi;
if [ -z "$NDIGITS" ]; then usage "Number of digits for frame numbering is not set."; fi;

path2vid=`cat scripts/paths/path_to_videos.txt`
yourfilenames=`ls ${path2vid}`
for file in ${yourfilenames}; do
    filename=$(basename -- "$file")
    extension="${filename##*.}"
    filename="${filename%.*}"
    echo ""
    echo "*****************************************************************"
    echo ""
    echo "      Current video: $filename"
    echo ""
    ./main_uniqueVideo.sh -v ${filename} --vidExt ${extension} --fps ${FPS} --framesExt ${FRAMESEXT} -n ${NDIGITS} ${BODY3D_STRING}${FACE3D_STRING}${HS_STRING}${KEEP_FULL_FRAMES_STRING}${KEEP_HAND_CROP_FRAMES_STRING}${KEEP_OPENPOSE_JSON_STRING}${KEEP_TEMPORARY_FEATURES_STRING}
done

echo "*****************************************************************"
echo "-----------------------------------------------------------------"
echo ""
echo "   MAIN SCRIPT, FINAL STEP - GETTING NORMALIZED FEATURES"
echo ""
echo "-----------------------------------------------------------------"
echo "*****************************************************************"
echo ""

if [[ "$BODY3D" = 1 ]] && [[ "$FACE3D" = 1 ]]; then
  LOAD3D_STRING=" --load3D"
else
  LOAD3D_STRING=0
fi

./scripts/get_normalized_features.sh ${LOAD3D_STRING}${HS_STRING}
