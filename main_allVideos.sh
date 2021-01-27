#!/bin/bash

CLEAR='\033[0m'
RED='\033[0;31m'

function usage() {
  if [ -n "$1" ]; then
    echo -e "${RED}👉 $1${CLEAR}\n";
  fi
  echo "Usage: $0 [--framesExt] [-n nDigits] [--handOP] [--faceOP] [--body3D] [--face3D] [--hs] [--keep_full_frames] [--keep_hand_crop_frames]"
  echo "  --framesExt              Frame files extension"
  echo "  -n, --nDigits            Number of digits for frame numbering"
  echo "  --handOP                 OpenPose computed on hands too"
  echo "  --faceOP                 OpenPose computed on face too"
  echo "  --body3D                 3D Body computed too"
  echo "  --face3D                 3D Face computed too"
  echo "  --hs                     Hand Shapes (Koller caffe model)"
  echo "  --keep_full_frames       For not deleting full frames         (optional, default=0)"
  echo "  --keep_hand_crop_frames  For not deleting hand crop frames    (optional, default=0)"
  echo ""
  echo "Example: $0 --framesExt jpg -n 5 --handOP --faceOP --body3D --face3D --hs --keep_full_frames --keep_hand_crop_frames"
  exit 1
}

# default params values
HANDOP=false
FACEOP=false
BODY3D=false
FACE3D=false
HS=false
KEEP_FULL_FRAMES=false
KEEP_HAND_CROP_FRAMES=false

HANDOP_STRING=""
FACEOP_STRING=""
BODY3D_STRING=""
FACE3D_STRING=""
HS_STRING=""
KEEP_FULL_FRAMES_STRING=""
KEEP_HAND_CROP_FRAMES_STRING=""

if [[ "$HANDOP" = true ]]; then HANDOP_STRING=" --handOP"; fi;
if [[ "$FACEOP" = true ]]; then FACEOP_STRING=" --faceOP"; fi;
if [[ "$BODY3D" = true ]]; then BODY3D_STRING=" --body3D"; fi;
if [[ "$FACE3D" = true ]]; then FACE3D_STRING=" --face3D"; fi;
if [[ "$HS" = true ]]; then HS_STRING=" --hs"; fi;
if [[ "$KEEP_FULL_FRAMES" = true ]]; then KEEP_FULL_FRAMES_STRING=" --keep_full_frames"; fi;
if [[ "$KEEP_HAND_CROP_FRAMES" = true ]]; then KEEP_HAND_CROP_FRAMES_STRING=" --keep_hand_crop_frames"; fi;


# parse params
while [[ "$#" > 0 ]]; do case $1 in
  --framesExt) FRAMESEXT="$2"; shift;shift;;
  -n|--nDigits) NDIGITS="$2"; shift;shift;;
  --handOP) HANDOP=true; shift;;
  --faceOP) FACEOP=true; shift;;
  --body3D) BODY3D=true; shift;;
  --face3D) FACE3D=true; shift;;
  --hs) HS=true; shift;;
  --keep_full_frames) KEEP_FULL_FRAMES=true; shift;;
  --keep_hand_crop_frames) KEEP_HAND_CROP_FRAMES=true; shift;;
  *) usage "Unknown parameter passed: $1"; shift; shift;;
esac; done

# verify params
if [ -z "$FRAMESEXT" ]; then usage "Frames extension is not set."; fi;
if [ -z "$NDIGITS" ]; then usage "Number of digits for frame numbering is not set."; fi;

path2vid=`cat scripts/paths/path_to_videos.txt`
yourfilenames=`ls ${path2vid}`
for file in ${yourfilenames}; do
    filename=$(basename -- "$file")
    extension="${filename##*.}"
    filename="${filename%.*}"
    echo $filename
    ./main_uniqueVideo.sh -v ${filename} --vidExt ${extension} --framesExt ${FRAMESEXT} -n ${NDIGITS} ${HANDOP_STRING}${FACEOP_STRING}${BODY3D_STRING}${FACE3D_STRING}${HS_STRING}${KEEP_FULL_FRAMES_STRING}${KEEP_HAND_CROP_FRAMES_STRING}
done
