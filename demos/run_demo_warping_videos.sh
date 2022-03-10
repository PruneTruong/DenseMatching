#!/bin/bash
set -e
# set -x

# Define an image sequence you want to use
# as an example, you can use this DAVIS video

echo "Downloading DAVIS camel video (as an example here)"
gdown https://drive.google.com/uc?id=1Xb0i58NBaxyXaxpahdLnnGN3RFQBPOcN
unzip camel_davis.zip -d .
rm camel_davis.zip

# Define input and name of the sequence
input=camel
name_of_sequence=camel
start=5  # here I want to start from frame 5, end is computed automatically and middle frame as well.
# you could also set manually your start, middle and end frames

# Choose model and pre_trained_model
model=PDCNet_plus
pre_trained_model=megadepth
global_optim_iter=3
local_optim_iter=7
path_to_pre_trained_models=/cluster/work/cvl/truongp/DenseMatching/pre_trained_models/

confident_mask_type=proba_interval_1_above_50

# define where to save the outputs
save_dir=/cluster/scratch/truongp/DAVIS_demo/    #evaluation

echo "Running demo.."
python demos/demo_warping_videos.py --save_video --mask_uncertain_regions \
--start $start --model $model --pre_trained_model $pre_trained_model \
--local_optim_iter $local_optim_iter --optim_iter $global_optim_iter \
--path_to_pre_trained_models $path_to_pre_trained_models \
--data_dir $input  --name_of_sequence $name_of_sequence --save_dir $save_dir \
--confident_mask_type  $confident_mask_type  PDCNet  --multi_stage_type  h



