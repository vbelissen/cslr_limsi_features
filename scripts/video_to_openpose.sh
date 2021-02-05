#!/bin/bash

echo "Getting openpose json data from video"

CLEAR='\033[0m'
RED='\033[0;31m'

function usage() {
  if [ -n "$1" ]; then
    echo -e "${RED}👉 $1${CLEAR}\n";
  fi
  echo "Usage: $0 [-v vidName] [--vidExt]"
  echo "  -v, --vidName            Video name without extension"
  echo "  --vidExt                 Video file extension"
#  echo "  --keep_full_frames       For not deleting full frames         (optional, default=0)"
#  echo "  --keep_hand_crop_frames  For not deleting hand crop frames    (optional, default=0)"
  echo ""
  echo "Example: $0 -v test_video_1 --vidExt mp4"
  exit 1
}


# parse params
while [[ "$#" > 0 ]]; do case $1 in
  -v|--vidName) VIDNAME="$2"; shift;shift;;
  --vidExt) VIDEXT="$2"; shift;shift;;
  *) usage "Unknown parameter passed: $1"; shift; shift;;
esac; done

# verify params
if [ -z "$VIDNAME" ]; then usage "Video name is not set"; fi;
if [ -z "$VIDEXT" ]; then usage "Video extension is not set."; fi;
#if [ -z "$HAND" ]; then usage "Hand computation is not set."; fi;
#if [ -z "$FACE" ]; then usage "Face computation is not set."; fi;

path2vid=`cat scripts/paths/path_to_videos.txt`
path2features=`cat scripts/paths/path_to_features.txt`
path2openPose=`cat scripts/paths/path_to_openpose.txt`

mkdir "${path2features}openpose/json/${VIDNAME}"
cd "${path2openPose}"
./build/examples/openpose/openpose.bin --video "${path2vid}${VIDNAME}.${VIDEXT}" --write_keypoint_json "${path2features}openpose/json/${VIDNAME}" --hand --hand_scale_number 3 --hand_scale_range 0.4 --face --no_display
