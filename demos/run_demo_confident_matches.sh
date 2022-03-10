#!/bin/bash
set -e
# set -x

# Use webcam 0 as input source.
#input=0
# or use a pre-recorded video given the path.
input=/home/sunjiaming/Downloads/scannet_test/$scene_name.mp4

# Choose model and pre_trained_model
model=PDCNet_plus
pre_trained_model=megadepth
global_optim_iter=3
local_optim_iter=7
save_dir=evaluation
confident_mask_type=proba_interval_1_above_50

# Optionally assign the GPU ID.
# export CUDA_VISIBLE_DEVICES=0


echo "Running demo.."

python demos/demo_confident_matches.py --save_video --model $model --pre_trained_model $pre_trained_model --optim_iter $global_optim_iter \
--path_to_pre_trained_models $path_to_pre_trained_models \
--local_optim_iter $local_optim_iter --confident_mask_type $confident_mask_type --input $input --visualize_warp \
--save_dir $save_dir --no_display  PDCNet  --multi_stage_type  h


# to show the uncertain regions zerod out instead
python demos/demo_confident_matches.py --mask_uncertain_regions  --save_video --model $model
--pre_trained_model $pre_trained_model --optim_iter $global_optim_iter \
--path_to_pre_trained_models $path_to_pre_trained_models \
--local_optim_iter $local_optim_iter --confident_mask_type $confident_mask_type --input $input --visualize_warp \
--save_dir $save_dir --no_display  PDCNet  --multi_stage_type  h

# Then convert them to a video.
# ffmpeg -framerate 15 -pattern_type glob -i '*.png' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4