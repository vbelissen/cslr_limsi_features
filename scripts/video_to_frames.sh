#!/bin/bash

CLEAR='\033[0m'
RED='\033[0;31m'

function usage() {
  if [ -n "$1" ]; then
    echo -e "${RED}👉 $1${CLEAR}\n";
  fi
  echo "Usage: $0 [-v vidName] [--vidExt] [--framesExt] [-n nDigits]"
  echo "  -v, --vidName            Video name without extension"
  echo "  --vidExt                 Video file extension"
  echo "  --framesExt              Frame files extension"
  echo "  -n, --nDigits            Number of digits for frame numbering"
#  echo "  --keep_full_frames       For not deleting full frames         (optional, default=0)"
#  echo "  --keep_hand_crop_frames  For not deleting hand crop frames    (optional, default=0)"
  echo ""
  echo "Example: $0 -v test_video_1 --vidExt mp4 --framesExt jpg -n 5"
  exit 1
}

# parse params
while [[ "$#" > 0 ]]; do case $1 in
  -v|--vidName) VIDNAME="$2"; shift;shift;;
  --vidExt) VIDEXT="$2"; shift;shift;;
  --framesExt) FRAMESEXT="$2"; shift;shift;;
  -n|--nDigits) NDIGITS="$2"; shift;shift;;
  #--keep_full_frames) KEEP_FULL_FRAMES=1; shift;shift;;
  #--keep_hand_crop_frames) KEEP_HAND_CROP_FRAMES=1;shift;shift;;
  *) usage "Unknown parameter passed: $1"; shift; shift;;
esac; done

# verify params
if [ -z "$VIDNAME" ]; then usage "Video name is not set"; fi;
if [ -z "$VIDEXT" ]; then usage "Video extension is not set."; fi;
if [ -z "$FRAMESEXT" ]; then usage "Frames extension is not set."; fi;
if [ -z "$NDIGITS" ]; then usage "Number of digits for frame numbering is not set."; fi;


path2vid=`cat scripts/paths/path_to_videos.txt`
path2frames=`cat scripts/paths/path_to_frames.txt`

mkdir "${path2frames}${VIDNAME}"
ffmpeg -loglevel panic -y -i "${path2vid}${VIDNAME}.${VIDEXT}" -qscale:v 2 "${path2frames}${VIDNAME}/%0${NDIGITS}d.${FRAMESEXT}"
