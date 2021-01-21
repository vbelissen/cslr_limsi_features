#!/bin/bash

vidName="test_video_1"
vidExt="mp4"
path2vid="/people/belissen/Code/cslr_limsi_features/videos/"
path2features="/people/belissen/Code/cslr_limsi_features/features/openpose/"
path2openPose="/people/belissen/openpose/"
hand=true
face=false


mkdir "${path2features}${vidName}"
if [[ "$hand" = true && "$face" = true ]]; then
    "${path2openPose}build/examples/openpose/openpose.bin" --video "${path2vid}${vidName}.${vidExt}" --write_keypoint_json "${path2features}${vidName}" --hand --hand_scale_number 3 --hand_scale_range 0.4 --face --no_display
elif [ "$hand" = true ]; then
    "${path2openPose}build/examples/openpose/openpose.bin" --video "${path2vid}${vidName}.${vidExt}" --write_keypoint_json "${path2features}${vidName}" --hand --hand_scale_number 3 --hand_scale_range 0.4 --no_display
elif [ "$face" = true ]; then
    "${path2openPose}build/examples/openpose/openpose.bin" --video "${path2vid}${vidName}.${vidExt}" --write_keypoint_json "${path2features}${vidName}" --face --no_display
else
    "${path2openPose}build/examples/openpose/openpose.bin" --video "${path2vid}${vidName}.${vidExt}" --write_keypoint_json "${path2features}${vidName}" --no_display
fi
