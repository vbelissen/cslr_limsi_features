#!/bin/bash

vidName="test_video_1"
vidExt="mp4"
path2vid="/Users/belissen/Code/cslr_limsi_features/videos/"
path2frames="/Users/belissen/Code/cslr_limsi_features/frames/full/"
nbZeros=5
framesExt="jpg"

mkdir "${path2frames}${vidName}"
ffmpeg -loglevel panic -y -i "${path2vid}${vidName}.${vidExt}" -qscale:v 2 "${path2frames}${vidName}/%0${nbZeros}d.jpg"
