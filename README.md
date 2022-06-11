# Dense Matching

A general dense matching library based on PyTorch.

For any questions, issues or recommendations, please contact Prune at prune.truong@vision.ee.ethz.ch

<br />


## Updates

06/03/2022: We found that significantly better performance and reduced training time are obtained when 
**initializing with bilinear interpolation weights the weights of the transposed convolutions** used to upsample 
the predicted flow fields between the different pyramid levels. We have integrated this initialization as the default. We might provide updated pre-trained weights as well.
Alternatively, one can directly simply use bilinear interpolation for upsampling with similar 
(maybe a bit better) performance, which is also now an option proposed. 


## Highlights

Libraries for implementing, training and evaluating dense matching networks. It includes
* Common dense matching **validation datasets** for **geometric matching** (MegaDepth, RobotCar, ETH3D, HPatches), 
**optical flow** (KITTI, Sintel) and **semantic matching** (TSS, PF-Pascal, PF-Willow, Spair). 
* Scripts to **analyse** network performance and obtain standard performance scores for matching and pose estimation.
* General building blocks, including deep networks, optimization, feature extraction and utilities.
* **General training framework** for training dense matching networks with
    * Common training datasets for matching networks.
    * Functions to generate random image pairs and their corresponding ground-truth flow, as well as to add 
    moving objects and modify the flow accordingly. 
    * Functions for data sampling, processing etc.
    * And much more...

* **Official implementation** of GLU-Net (CVPR 2020), GLU-Net-GOCor (NeurIPS 2020), PWC-Net-GOCor (NeurIPS 2020), 
PDC-Net (CVPR 2021), WarpC models (ICCV 2021), PWarpC models (CVPR 2022) including trained models and respective results.

<br />

## Dense Matching Networks 

The repo contains the implementation of the following matching models. 
We provide pre-trained model weights, data preparation, evaluation commands, and results for each dataset and method. 

### [6] PWarpC: Probabilistic Warp Consistency for Weakly-Supervised Semantic Correspondences. (CVPR 2022)
Authors: [Prune Truong](https://prunetruong.com/), [Martin Danelljan](https://martin-danelljan.github.io/), 
[Fisher Yu](https://www.yf.io/), Luc Van Gool<br />

\[[Paper](https://arxiv.org/abs/2203.04279)\]
\[[Website](https://prunetruong.com/research/pwarpc)\]
\[[Poster](https://drive.google.com/file/d/1lP5E3BNqdKJL1q-YsQ-C7rOwkcb5S63W/view?usp=sharing)\]
\[[Video](https://www.youtube.com/watch?v=I2KtnvI8xZU)\]

![alt text](/images/pwarpc_banner.png)

We propose Probabilistic Warp Consistency, a weakly-supervised learning objective for semantic matching. 
Our approach directly supervises the dense matching scores predicted by the network, encoded as a conditional 
probability distribution. We first construct an image triplet by applying a known warp to one of the images in 
a pair depicting different instances of the same object class. Our probabilistic learning objectives are then 
derived using the constraints arising from the resulting image triplet. We further account for occlusion and 
background clutter present in real image pairs by extending our probabilistic output space with a learnable 
unmatched state. To supervise it, we design an objective between image pairs depicting different object classes. 
We validate our method by applying it to four recent semantic matching architectures. Our weakly-supervised approach 
sets a new state-of-the-art on four challenging semantic matching benchmarks. Lastly, we demonstrate that our 
objective also brings substantial improvements in the strongly-supervised regime, when combined with keypoint annotations. 



### [5] PDC-Net+: Enhanced Probabilistic Dense Correspondence Network. (Preprint)
Authors: [Prune Truong](https://prunetruong.com/), [Martin Danelljan](https://martin-danelljan.github.io/), Radu Timofte, Luc Van Gool <br />

\[[Paper](https://arxiv.org/abs/2109.13912)\]
\[[Website](https://prunetruong.com/research/pdcnet+)\]



### [4] WarpC: Warp Consistency for Unsupervised Learning of Dense Correspondences. (ICCV 2021 - ORAL)
Authors: [Prune Truong](https://prunetruong.com/), [Martin Danelljan](https://martin-danelljan.github.io/), 
[Fisher Yu](https://www.yf.io/), Luc Van Gool<br />

\[[Paper](https://arxiv.org/abs/2104.03308)\]
\[[Website](https://prunetruong.com/research/warpc)\]
\[[Poster](https://drive.google.com/file/d/1PCXkjxvVsjHAbYzsBtgKWLO1uE6oGP6p/view?usp=sharing)\]
\[[Slides](https://drive.google.com/file/d/1mVpLBW55nlNJZBsvxkBCti9_KhH1r9V_/view?usp=sharing)\]
\[[Video](https://www.youtube.com/watch?v=IsMotj7-peA)\]

Warp Consistency Graph            |  Results 
:-------------------------:|:-------------------------:
![alt text](/images/warpc_thumbnail.png) |  ![](/images/warpc_banner.png) 




The key challenge in learning dense correspondences lies in the lack of ground-truth matches for real image pairs. 
While photometric consistency losses provide unsupervised alternatives, they struggle with large appearance changes, 
which are ubiquitous in geometric and semantic matching tasks. Moreover, methods relying on synthetic training pairs 
often suffer from poor generalisation to real data.
We propose Warp Consistency, an unsupervised learning objective for dense correspondence regression. 
Our objective is effective even in settings with large appearance and view-point changes. Given a pair of 
real images, we first construct an image triplet by applying a randomly sampled warp to one of the original images. 
We derive and analyze all flow-consistency constraints arising between the triplet. From our observations and 
empirical results, we design a general unsupervised objective employing two of the derived constraints.
We validate our warp consistency loss by training three recent dense correspondence networks for the geometric and 
semantic matching tasks. Our approach sets a new state-of-the-art on several challenging benchmarks, including MegaDepth, 
RobotCar and TSS. 



### [3] PDC-Net: Learning Accurate Correspondences and When to Trust Them. (CVPR 2021 - ORAL)
Authors: [Prune Truong](https://prunetruong.com/), [Martin Danelljan](https://martin-danelljan.github.io/), Luc Van Gool, Radu Timofte<br />

\[[Paper](https://arxiv.org/abs/2101.01710)\]
\[[Website](https://prunetruong.com/research/pdcnet)\]
\[[Poster](https://drive.google.com/file/d/18ya__AdEIgZyix8dXuRpJ15tdrpbMUsB/view?usp=sharing)\]
\[[Slides](https://drive.google.com/file/d/1zUQmpmVp6WSa_psuI3KFvKVrNyJE-beG/view?usp=sharing)\]
\[[Video](https://youtu.be/bX0rEaSf88o)\]

![alt text](/images/PDCNet.png)


Dense flow estimation is often inaccurate in the case of large displacements or homogeneous regions. For most 
applications and down-stream tasks, such as pose estimation, image manipulation, or 3D reconstruction, it is 
crucial to know **when and where** to trust the estimated matches. 
In this work, we aim to estimate a dense flow field relating two images, coupled with a robust pixel-wise 
confidence map indicating the reliability and accuracy of the prediction. We develop a flexible probabilistic 
approach that jointly learns the flow prediction and its uncertainty. In particular, we parametrize the predictive 
distribution as a constrained mixture model, ensuring better modelling of both accurate flow predictions and outliers. 
Moreover, we develop an architecture and training strategy tailored for robust and generalizable uncertainty 
prediction in the context of self-supervised training. 


### [2] GOCor: Bringing Globally Optimized Correspondence Volumes into Your Neural Network. (NeurIPS 2020)
Authors: [Prune Truong](https://prunetruong.com/) *, [Martin Danelljan](https://martin-danelljan.github.io/) *, Luc Van Gool, Radu Timofte<br />

\[[Paper](https://arxiv.org/abs/2009.07823)\]
\[[Website](https://prunetruong.com/research/gocor)\]
\[[Video](https://www.youtube.com/watch?v=V22MyFChBCs)\]


The feature correlation layer serves as a key neural network module in numerous computer vision problems that
involve dense correspondences between image pairs. It predicts a correspondence volume by evaluating dense scalar products 
between feature vectors extracted from pairs of locations in two images. However, this point-to-point feature comparison 
is insufficient when disambiguating multiple similar regions in an image, severely affecting the performance of 
the end task. 
**This work proposes GOCor, a fully differentiable dense matching module, acting as a direct replacement to 
the feature correlation layer.** The correspondence volume generated by our module is the result of an internal 
optimization procedure that explicitly accounts for similar regions in the scene. Moreover, our approach is 
capable of effectively learning spatial matching priors to resolve further matching ambiguities. 


![alt text](/images/corr_diff_iteration.jpg)




### [1] GLU-Net: Global-Local Universal Network for dense flow and correspondences (CVPR 2020 - ORAL).
Authors: [Prune Truong](https://prunetruong.com/), [Martin Danelljan](https://martin-danelljan.github.io/) and Radu Timofte <br />
\[[Paper](https://arxiv.org/abs/1912.05524)\]
\[[Website](https://prunetruong.com/research/glu-net)\]
\[[Poster](https://drive.google.com/file/d/1pS_OMZ83EG-oalD-30vDa3Ru49GWi-Ky/view?usp=sharing)\]
\[[Oral Video](https://www.youtube.com/watch?v=xB2gNx8f8Xc&feature=emb_title)\]
\[[Teaser Video](https://www.youtube.com/watch?v=s5OUdkM9QLo)\]

![alt text](/images/glunet.png)


<br />
<br />

## Pre-trained weights

The pre-trained models can be found in the [model zoo](https://github.com/PruneTruong/DenseMatching/blob/main/MODEL_ZOO.md)


<br />

## Table of Content

1. [Installation](#Installation)
2. [Test on your own image pairs!](#test)
3. [Overview](#overview)
4. [Benchmarks and results](#Results)
    1. [Correspondence evaluation](#correspondence_eval)
        1. [MegaDepth](#megadepth)
        2. [RobotCar](#robotcar)
        3. [ETH3D](#eth3d)
        4. [HPatches](#hpatches)
        5. [KITTI](#kitti)
        6. [Sintel](#sintel)
        7. [TSS](#tss)
        8. [PF-Pascal](#pfpascal)
        9. [PF-Willow](#pfwillow)
        10. [Spair-71k](#spair)
        11. [Caltech-101](#caltech)
    2. [Pose estimation](#pose_estimation)
        1. [YFCC100M](#yfcc)
        2. [ScanNet](#scannet)
    3. [Sparse evaluation on HPatches](#sparse_hp)
5. [Training](#training)
6. [Acknowledgement](#acknowledgement)
7. [Changelog](#changelog)


<br />

## 1. Installation <a name="Installation"></a>

Inference runs for torch version >= 1.0

* Create and activate conda environment with Python 3.x

```bash
conda create -n dense_matching_env python=3.7
conda activate dense_matching_env
```

* Install all dependencies (except for cupy, see below) by running the following command:
```bash
pip install numpy opencv-python torch torchvision matplotlib imageio jpeg4py scipy pandas tqdm gdown pycocotools timm
```

**Note**: CUDA is required to run the code. Indeed, the correlation layer is implemented in CUDA using CuPy, 
which is why CuPy is a required dependency. It can be installed using pip install cupy or alternatively using one of the 
provided binary packages as outlined in the CuPy repository. The code was developed using Python 3.7 & PyTorch 1.0 & CUDA 9.0, 
which is why I installed cupy for cuda90. For another CUDA version, change accordingly. 

```bash
pip install cupy-cuda90 --no-cache-dir 
```


* This repo includes [GOCor](https://arxiv.org/abs/2009.07823) as git submodule. 
You need to pull submodules with 
```bash
git submodule update --init --recursive
git submodule update --recursive --remote
```

* Create admin/local.py by running the following command and update the paths to the dataset. 
We provide an example admin/local_example.py where all datasets are stored in data/. 
```bash
python -c "from admin.environment import create_default_local_file; create_default_local_file()"
```

* **Download pre-trained model weights** with the command ```bash assets/download_pre_trained_models.sh```. See more in [model zoo](https://github.com/PruneTruong/DenseMatching/blob/main/MODEL_ZOO.md)

<br />

## 2. Test on your own image pairs!  <a name="Test"></a>

Possible model choices are : 
* SFNet, PWarpCSFNet_WS, PWarpCSFNet_SS, NCNet, PWarpCNCNet_WS, PWarpCNCNet_SS, CATs, PWarpCCATs_SS, CATs_ft_features, 
 PWarpCCATs_ft_features_SS
* WarpCGLUNet, GLUNet_star, WarpCSemanticGLUNet, 
* PDCNet_plus, PDCNet, GLUNet_GOCor_star, 
* SemanticGLUNet, GLUNet, GLUNet_GOCor, PWCNet, PWCNet_GOCor

Possible pre-trained model choices are: static, dynamic, chairs_things, chairs_things_ft_sintel, megadepth, 
megadepth_stage1, pfpascal, spair

<br />

<details>
  <summary><b>Note on PDCNet and PDC-Net+ inference options</b></summary>

PDC-Net and PDC-Net+ have multiple inference alternative options. 
if model is PDC-Net, add options:  
* --confidence_map_R, for computation of the confidence map p_r, default is 1.0
* --multi_stage_type in 
    * 'D' (or 'direct')
    * 'H' (or 'homography_from_quarter_resolution_uncertainty')
    * 'MS' (or 'multiscale_homo_from_quarter_resolution_uncertainty')
* --ransac_thresh, used for homography and multiscale multi-stages type, default is 1.0
* --mask_type, for thresholding the estimated confidence map and using the confident matches for internal homography estimation, for 
homography and multiscale multi-stage types, default is proba_interval_1_above_5
* --homography_visibility_mask, default is True
* --scaling_factors', used for multi-scale, default are \[0.5, 0.6, 0.88, 1, 1.33, 1.66, 2\]

Use direct ('D') when image pairs only show limited view-point changes (for example consecutive images of a video, 
like in the optical flow task). For larger view-point changes, use homography ('H') or multi-scale ('MS'). 


For example, to run PDC-Net or PDC-Net+ with homography, add at the end of the command
```bash
PDCNet --multi_stage_type H --mask_type proba_interval_1_above_10
```

</details>


### Test on a specific image pair 


You can test the networks on a pair of images using test_models.py and the provided trained model weights. 
You must first choose the model and pre-trained weights to use. 
The inputs are the paths to the query and reference images. 
The images are then passed to the network which outputs the corresponding flow field relating the reference to the query image. 
The query is then warped according to the estimated flow, and a figure is saved. 

<br />

For this pair of MegaDepth images (provided to check that the code is working properly) and using **PDCNet** (MS) 
trained on the megadepth dataset, the output is:

```bash
python test_models.py --model PDCNet --pre_trained_model megadepth --path_query_image images/piazza_san_marco_0.jpg --path_reference_image images/piazza_san_marco_1.jpg --save_dir evaluation/ PDCNet --multi_stage_type MS --mask_type proba_interval_1_above_10
```
additional optional arguments: --pre_trained_models_dir (default is pre_trained_models/)
![alt text](/images/Warped_query_image_PDCNet_megadepth.png)


<br />

Using **GLU-Net-GOCor** trained on the dynamic dataset, the output for this image pair of eth3d is:

```bash
python test_models.py --model GLUNet_GOCor --pre_trained_model dynamic --path_query_image images/eth3d_query.png --path_reference_image images/eth3d_reference.png --save_dir evaluation/
```
![alt text](/images/eth3d_warped_query_image_GLUNet_GOCor_dynamic.png)

<br />

For baseline **GLU-Net**, the output is instead:

```bash
python test_models.py --model GLUNet --pre_trained_model dynamic --path_query_image images/eth3d_query.png --path_reference_image images/eth3d_reference.png --save_dir evaluation/

```
![alt text](/images/eth3d_warped_query_image_GLUNet_dynamic.png)


<br />

And for **PWC-Net-GOCor** and baseline **PWC-Net**:


```bash
python test_models.py --model PWCNet_GOCor --pre_trained_model chairs_things --path_query_image images/kitti2015_query.png --path_reference_image images/kitti2015_reference.png --save_dir evaluation/
```

![alt text](/images/kitti2015_warped_query_image_PWCNet_GOCor_chairs_things.png)

<br />

```bash
python test_models.py --model PWCNet --pre_trained_model chairs_things --path_query_image images/kitti2015_query.png --path_reference_image images/kitti2015_reference.png --save_dir evaluation/
```
![alt text](/images/kitti2015_warped_query_image_PWCNet_chairs_things.png)

<br />


## Demos with videos 

* **demos/demo_single_pair.ipynb**: Play around with our models on different image pairs, compute the flow field 
relating an image pair and visualize the warped images and confident matches. 

* **demos/demo_pose_estimation_and_reconstruction.ipynb**:  Play around with our models on different image pairs 
(with intrinsic camera parameters known), compute the flow field and confidence map, then the relative pose.   

* Run the **online demo with a webcam or video** to reproduce the result shown in the GIF above. 
We compute the flow field between the target (middle) and the source (left). We plot the 1000 top confident matches as well. 
The warped source is represented on the right (and should resemble the middle). 
Only the regions for which the matches were predicted as confident are visible. 


![alt text](/images/scannet.gif)

We modify the utils code from [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork), so you need to adhere to 
their license in order to use that. Then run:
```bash
bash demo/run_demo_confident_matches.sh
``` 


* From a video, **warp each frame of the video (left) to the middle frame** like in the GIF below. The warped frame is 
represented on the right (and should resemble the middle). 
Only the regions for which the matches were predicted as confident are visible. 

![alt text](/images/camel_5_90_mid_47_84.gif)

We modify the utils code from [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork), so you need to adhere to 
their license in order to use that. Then run:
```bash
bash demo/run_demo_warping_videos.sh
``` 



* More to come!


## 3. Overview  <a name="overview"></a>

The framework consists of the following sub-modules.

* training: 
    * actors: Contains the actor classes for different trainings. The actor class is responsible for passing the input 
    data through the network and calculating losses. 
    Here are also pre-processing classes, that process batch tensor inputs to the desired inputs needed for training the network. 
    * trainers: The main class which runs the training.
    * losses: Contain the loss classes 
* train_settings: Contains settings files, specifying the training of a network.
* admin: Includes functions for loading networks, tensorboard etc. and also contains environment settings.
* datasets: Contains integration of a number of datasets. Additionally, it includes modules to generate 
synthetic image pairs and their corresponding ground-truth flow as well as to add independently moving objects 
and modify the flow accordingly. 
* utils_data: Contains functions for processing data, e.g. loading images, data augmentations, sampling frames.
* utils_flow: Contains functions for working with flow fields, e.g. converting to mapping, warping an array according 
to a flow, as well as visualization tools. 
* third_party: External libraries needed for training. Added as submodules.
* models: Contains different layers and network definitions.
* validation: Contains functions to evaluate and analyze the performance of the networks in terms of predicted 
flow and uncertainty. 



## 4. Benchmark and results  <a name="Results"></a>

All paths to the datasets must be provided in file admin/local.py. 
We provide an example admin/local_example.py where all datasets are stored in data/. 
You need to update the paths of admin/local.py before running the evaluation. 



<details>
  <summary><b>Note on PDCNet and PDCNet+ inference options</b></summary>

PDC-Net and PDC-Net+ has multiple inference alternative options. 
if model if PDC-Net, add options:  
* --confidence_map_R, for computation of the confidence map p_r, default is 1.0
* --multi_stage_type in 
    * 'D' (or 'direct')
    * 'H' (or 'homography_from_quarter_resolution_uncertainty')
    * 'MS' (or 'multiscale_homo_from_quarter_resolution_uncertainty')
* --ransac_thresh, used for homography and multiscale multi-stages type, default is 1.0
* --mask_type, for thresholding the estimated confidence map and using the confident matches for internal homography estimation, for 
homography and multiscale multi-stage types, default is proba_interval_1_above_5
* --homography_visibility_mask, default is True
* --scaling_factors', used for multi-scale, default are \[0.5, 0.6, 0.88, 1, 1.33, 1.66, 2\]

For example, to run PDC-Net or PDC-Net+ with homography, add at the end of the command
```bash
PDCNet --multi_stage_type H --mask_type proba_interval_1_above_10
```

</details>


<details>
  <summary><b>Note on reproducibility</b></summary>

Results using PDC-Net with multi-stage (homography_from_quarter_resolution_uncertainty, H) or multi-scale 
(multiscale_homo_from_quarter_resolution_uncertainty, MS) employ RANSAC internally. Therefore results may vary a bit but should remain within 1-2 %.
For pose estimation, we also compute the pose with RANSAC, which leads to some variability in the results.  
  
</details>

### 4.1. Correspondence evaluation <a name="correspondence_eval"></a>

Metrics are computed with, 
```bash
python -u eval_matching.py --dataset dataset_name --model model_name --pre_trained_models pre_trained_model_name --optim_iter optim_step  --local_optim_iter local_optim_iter --save_dir path_to_save_dir 
```

Optional argument:
--path_to_pre_trained_models: 
   * default is pre_trained_models/
   * if it is a path to a directory: it is the path to the directory containing the model weights, the path to the model weights will be 
    path_to_pre_trained_models + model_name + '_' + pre_trained_model_name
   * if it is a path to a checkpoint directly, it is used as the path to the model weights directly, and pre_trained_model_name is only used
    as the name to save the metrics. 


<details>
  <summary><b>MegaDepth</b><a name="megadepth"></a></summary>


**Data preparation**: We use the test set provided in [RANSAC-Flow](https://github.com/XiSHEN0220/RANSAC-Flow/tree/master/evaluation/evalCorr). 
It is composed of 1600 pairs and  also includes a csv file ('test1600Pairs.csv') containing 
the name of image pairs to evaluate and the corresponding ground-truth correspondences. 
Download everything with 
```bash
bash assets/download_megadepth_test.sh
```

The resulting file structure is the following
```bash
megadepth_test_set/
└── MegaDepth/
    └── Test/
        └── test1600Pairs/  
        └── test1600Pairs.csv
```

<br /><br />
**Evaluation**: After updating the path of 'megadepth' and 'megadepth_csv' in admin/local.py, evaluation is run with
```bash
python eval_matching.py --dataset megadepth --model PDCNet --pre_trained_models megadepth --optim_iter 3 --local_optim_iter 7 --save_dir path_to_save_dir PDCNet --multi_stage_type MS
```


Similar results should be obtained: 
| Model          | Pre-trained model type      | PCK-1 (%) | PCK-3 (%) | PCK-5 (%) | 
|----------------|-----------------------------|-------|-------|-------|
| GLU-Net  (this repo)      | static  (CityScape-DPED-ADE)        | 29.51 | 50.67 | 56.12 | 
| GLU-Net  (this repo)      | dynamic           | 21.59 | 52.27 | 61.91 | 
| GLU-Net  (paper)      | dynamic       | 21.58 | 52.18 | 61.78 | 
| GLU-Net-GOCor  (this repo) | static (CitySCape-DPED-ADE) | 32.24 | 52.51 | 58.90 | 
| GLU-Net-GOCor (this repo) | dynamic  | 37.23 | **61.25** | **68.17** | 
| GLU-Net-GOCor (paper) | dynamic      | **37.28**| 61.18 | 68.08 | 
|----------------|-----------------------------|-------|-------|-------|
| GLU-Net-GOCor* (paper) | megadepth                   | 57.77 | 78.61 | 82.24 | 
| PDC-Net  (D)   (this repo)  | megadepth                   | 68.97 | 84.03 | 85.68 | 
| PDC-Net  (H)   (paper)   | megadepth                   | 70.75 | 86.51 | 88.00 | 
| PDC-Net (MS)  (paper) | megadepth                   | 71.81 | 89.36 | 91.18 | 
| PDC-Net+ (D) (paper)  | megadepth |  72.41 | 86.70 | 88.13  |
| PDC-Net+ (H) (paper)  | megadepth |  73.92  |  89.21 |  90.48  |
| PDC-Net+ (MS) (paper)  | megadepth |  **74.51**  |  **90.69**  | **92.10** |
|----------------|-----------------------------|-------|-------|-------|
| GLU-Net*  (paper)   | megadepth  |  38.50 | 54.66 | 59.60 | 
| GLU-Net*  (this repo)   | megadepth  |  38.62 | 54.70 |  59.76 | 
| WarpC-GLU-Net (paper) | megadepth | 50.61 | 73.80 | 78.61 | 
| WarpC-GLU-Net (this repo) | megadepth | **50.77** | **73.91** | **78.73** | 
</details>



<details>
  <summary><b>RobotCar <a name="robotcar"></b></a></summary>
  
**Data preparation**: Images can be downloaded from the 
[Visual Localization Challenge](https://www.visuallocalization.net/datasets/) (at the bottom of the site), 
or more precisely [here](https://www.dropbox.com/sh/ql8t2us433v8jej/AAB0wfFXs0CLPqSiyq0ukaKva/ROBOTCAR?dl=0&subfolder_nav_tracking=1). 
The CSV file with the ground-truth correspondences can be downloaded from [here](https://drive.google.com/file/d/16mZLUKsjceAt1RTW1KLckX0uCR3O4x5Q/view). 
The file structure should be the following: 

```bash
RobotCar
├── img/
└── test6511.csv
```

<br /><br />
**Evaluation**: After updating the path of 'robotcar' and 'robotcar_csv' in admin/local.py, evaluation is run with
```bash
python eval_matching.py --dataset robotcar --model PDCNet --pre_trained_models megadepth --optim_iter 3 --local_optim_iter 7 --save_dir path_to_save_dir PDCNet --multi_stage_type MS
```
 
Similar results should be obtained: 
 | Model          | Pre-trained model type      | PCK-1 (%) | PCK-3 (%) | PCK-5 (%) |
|----------------|-----------------------------|-------|-------|-------|
| GLU-Net     (paper)   | static (CityScape-DPED-ADE)          | 2.30  | 17.15 | 33.87 |
| GLU-Net-GOCor  (paper) | static | **2.31**  | **17.62** | **35.18** |
| GLU-Net-GOCor  (paper) | dynamic                     | 2.10  | 16.07 | 31.66 |
|----------------|-----------------------------|-------|-------|-------|
| GLU-Net-GOCor* (paper) | megadepth                   | 2.33  | 17.21 | 33.67 |
| PDC-Net  (H)    (paper)   | megadepth                   | 2.54  | 18.97 | 36.37 |
| PDC-Net (MS)   (paper) | megadepth                   | 2.58  | 18.87 | 36.19 |
| PDC-Net+ (D)  (paper)  | megadepth |  2.57  | **19.12** | **36.71**  |
| PDC-Net+ (H)  (paper)  | megadepth |  2.56 | 19.00 | 36.56 |
| PDC-Net+ (MS)  (paper)  | megadepth |  **2.63** | 19.01 | 36.57 |
|----------------|-----------------------------|-------|-------|-------|
| GLU-Net* (paper) | megadepth | 2.36 | 17.18 | 33.28 |
| WarpC-GLU-Net (paper) | megadepth | **2.51** | **18.59** | **35.92** | 
</details>



<details>
  <summary><b>ETH3D <a name="eth3d"></a></b></summary>
  
**Data preparation**: execute 'bash assets/download_ETH3D.sh' from our [GLU-Net repo](https://github.com/PruneTruong/GLU-Net). 
It does the following: 
- Create your root directory ETH3D/, create two sub-directories multiview_testing/ and multiview_training/
- Download the "Low rew multi-view, training data, all distorted images" [here](https://www.eth3d.net/data/multi_view_training_rig.7z) and unzip them in multiview_training/
- Download the "Low rew multi-view, testing data, all undistorted images" [here](https://www.eth3d.net/data/multi_view_test_rig_undistorted.7z) and unzip them in multiview_testing/
- We directly provide correspondences for pairs of images taken at different intervals. There is one bundle file for each dataset and each rate of interval, for example "lakeside_every_5_rate_of_3". 
This means that we sampled the source images every 5 images and the target image is taken at a particular rate from each source image. Download all these files [here](https://drive.google.com/file/d/1Okqs5QYetgVu_HERS88DuvsABGak08iN/view?usp=sharing) and unzip them. 

As illustration, your root ETH3D directory should be organised as follows:
<pre>
/ETH3D/
       multiview_testing/
                        lakeside/
                        sand_box/
                        storage_room/
                        storage_room_2/
                        tunnel/
       multiview_training/
                        delivery_area/
                        electro/
                        forest/
                        playground/
                        terrains/
        info_ETH3D_files/
</pre>
The organisation of your directories is important, since the bundle files contain the relative paths to the images, from the ETH3D root folder. 

<br /><br />
**Evaluation**: for each interval rate (3,5,7,9,11,13,15), we compute the metrics for each of the sub-datasets 
(lakeside, delivery area and so on). The final metrics are the average over all datasets for each rate. 
After updating the path of 'eth3d' in admin/local.py, evaluation is run with
```bash
python eval_matching.py --dataset robotcar --model PDCNet --pre_trained_models megadepth --optim_iter 3 --local_optim_iter 7 --save_dir path_to_save_dir PDCNet --multi_stage_type D
```

<br />
AEPE for different rates of intervals between image pairs.

| Method        | Pre-trained model type | rate=3 | rate=5 | rate=7 | rate=9 | rate=11 | rate=13 | rate=15 |
|---------------|------------------------|--------|--------|--------|--------|---------|---------|---------|
| LiteFlowNet   | chairs-things          | **1.66**   | 2.58   | 6.05   | 12.95  | 29.67   | 52.41   | 74.96   |
| PWC-Net       | chairs-things          | 1.75   | 2.10   | 3.21   | 5.59   | 14.35   | 27.49   | 43.41   |
| PWC-Net-GOCor | chairs-things          |  1.70      |  **1.98**      |  **2.58**      | **4.22**      |  **10.32**       |  **21.07**       |  **38.12**       |
|---------------|------------------------|--------|--------|--------|--------|---------|---------|---------|
| DGC-Net       |                        | 2.49   | 3.28   | 4.18   | 5.35   | 6.78    | 9.02    | 12.23   |
| GLU-Net       | static                 | 1.98   | 2.54   | 3.49   | 4.24   | 5.61    | 7.55    | 10.78   |
| GLU-Net       | dynamic                | 2.01   | 2.46   | 2.98   | 3.51   | 4.30    | 6.11    | 9.08    |
| GLU-Net-GOCor | dynamic                | **1.93**   | **2.28**   | **2.64**   | **3.01**   | **3.62**    | **4.79**    | **7.80**    |
|---------------|------------------------|--------|--------|--------|--------|---------|---------|---------|
| GLU-Net-GOCor*| megadepth              |  1.68 | 1.92 | 2.18 | 2.43 |  2.89 | 3.31 |  4.27 |
| PDC-Net  (D) (paper) | megadepth       | 1.60   | 1.79   | 2.03   |  2.26  |  2.58   | 2.92    | 3.69    |
| PDC-Net  (H)  | megadepth              | 1.58   | 1.77   | 1.98   |  2.24  |  2.56   | 2.91    | 3.73    |
| PDC-Net  (MS)  | megadepth             |  1.60 |  1.79  |  2.00  | 2.26   | 2.57   | 2.90    |  3.56   |
| PDC-Net+  (H) (paper) | megadepth  | **1.56** | **1.74** | **1.96** | **2.18** | **2.48** |  **2.73** | **3.24** |
| PDC-Net+  (MS) (paper) | megadepth  | 1.58 | 1.76 | 1.96 | 2.16 | 2.49 | **2.73** | **3.24** |


PCK-1 for different rates of intervals between image pairs: 

Note that the PCKs are computed **per image**, and 
then averaged per sequence. The final metrics is the average over all sequences. It corresponds to the results 
'_per_image' in the outputted metric file. 
Note that this is not the metrics used in the [PDC-Net paper](https://arxiv.org/abs/2101.01710), where the PCKs are c
omputed **per sequence** instead, using the PDC-Net direct approach (corresponds to results '-per-rate' 
in outputted metric file). 


| Method        | Pre-trained model type | rate=3 | rate=5 | rate=7 | rate=9 | rate=11 | rate=13 | rate=15 |
|---------------|------------------------|--------|--------|--------|--------|---------|---------|---------|
| LiteFlowNet   | chairs-things          | **61.63**   | **56.55**   | **49.83**   | **42.00**  | 33.14   | 26.46   | 21.22   |
| PWC-Net       | chairs-things          | 58.50   | 52.02   | 44.86   | 37.41   | 30.36   | 24.75   | 19.89   |
| PWC-Net-GOCor | chairs-things          | 58.93   | 53.10   |  46.91  |  40.93  |  **34.58**   |  **29.25**  |  **24.59**       |
|---------------|------------------------|--------|--------|--------|--------|---------|---------|---------|
| DGC-Net       |                        | 
| GLU-Net       | static                 | **50.55**   | **43.08**   | **36.98**   | 32.45   | 28.45    | 25.06    | 21.89   |
| GLU-Net       | dynamic                | 46.27  | 39.28   | 34.05   | 30.11   | 26.69    | 23.73    | 20.85    |
| GLU-Net-GOCor | dynamic                | 47.97   |41.79   | 36.81   | **33.03**   | **29.80**    | **26.93**    | **23.99**    |
|---------------|------------------------|--------|--------|--------|--------|---------|---------|---------|
| GLU-Net-GOCor*| megadepth              | 59.40 |  55.15 | 51.18 |  47.86 | 44.46 | 41.78 | 38.91 |
| PDC-Net  (D)  | megadepth              |  61.82 | 58.41  | 55.02 |  52.40 | 49.61 | 47.43 | 45.01 | 
| PDC-Net  (H)  | megadepth              |  62.63  | 59.29 | 56.09 | 53.31 | 50.69 | 48.46 | 46.17 | 
| PDC-Net  (MS) | megadepth              |  62.29 | 59.14   | 55.87 | 53.23 | 50.59 | 48.45 | 46.17 |
| PDC-Net+  (H) | megadepth              | **63.12** | **59.93** | **56.81** | **54.12** | **51.59** | **49.55** | **47.32** |
| PDC-Net+  (MS) | megadepth              | 62.95 | 59.76 | 56.64 | 54.02 | 51.50 | 49.38 | 47.24 | 



PCK-5 for different rates of intervals between image pairs: 

| Method        | Pre-trained model type | rate=3 | rate=5 | rate=7 | rate=9 | rate=11 | rate=13 | rate=15 |
|---------------|------------------------|--------|--------|--------|--------|---------|---------|---------|
| LiteFlowNet   | chairs-things          | **92.79**   | 90.70   | 86.29   | 78.50  | 66.07   | 55.05   | 46.29   |
| PWC-Net       | chairs-things          | 92.64  | 90.82   | 87.32   | 81.80   | 72.95   | 64.07   | 55.47   |
| PWC-Net-GOCor | chairs-things          | 92.81      |  **91.45**      |  **88.96**      |  **85.53**      |  **79.44**       |  **72.06**       |  **64.92**       |
|---------------|------------------------|--------|--------|--------|--------|---------|---------|---------|
| DGC-Net       |                        | 88.50   | 83.25   | 78.32  | 73.74   | 69.23   | 64.28    | 58.66  |
| GLU-Net       | static                 | 91.22  | 87.91   | 84.23   |  80.74  | 76.84    | 72.35    | 67.77   |
| GLU-Net       | dynamic                | 91.45   | 88.57   | 85.64   | 83.10   | 80.12    | 76.66    | 73.02    |
| GLU-Net-GOCor | dynamic                | **92.08**   | **89.87**   | **87.77**   | **85.88**   | **83.69**    | **81.12**    | **77.90**    |
|---------------|------------------------|--------|--------|--------|--------|---------|---------|---------|
| GLU-Net-GOCor*| megadepth              |  93.03 |92.13 | 91.04 | 90.19 | 88.98 |  87.81 |  85.93 | 
| PDC-Net  (D) (paper) | megadepth       | 93.47 | 92.72 | 91.84 | 91.15 | 90.23 | 89.45 | 88.10 | 
| PDC-Net  (H)  | megadepth              | 93.50 | 92.71  | 91.93 | 91.16 | 90.35 | 89.52 | 88.32 | 
| PDC-Net  (MS)  | megadepth             |  93.47 | 92.69 | 91.85 | 91.15 | 90.33 | 89.55 | 88.43 | 
| PDC-Net+  (H)  | megadepth             |  **93.54** | 92.78 |  **92.04** | 91.30 | **90.60** | 89.9 | **89.03** |
| PDC-Net+  (MS)  | megadepth             | 93.50 |  **92.79** | **92.04** | **91.35** | **90.60** | **89.97** | 88.97 | 



</details>


<details>
  <summary><b>HPatches <a name="hpatches"></a></b></summary>

**Data preparation**: Download the data with 
```bash
bash assets/download_hpatches.sh
```
The corresponding csv files for each viewpoint ID with the path to the images and the homography parameters 
relating the pairs are listed in assets/. 


<br /><br />
**Evaluation**: After updating the path of 'hp' in admin/local.py, evaluation is run with
```bash
python eval_matching.py --dataset hp --model GLUNet_GOCor --pre_trained_models static --optim_iter 3 --local_optim_iter 7 --save_dir path_to_save_dir
```
Similar results should be obtained:
|                             | Pre-trained model type | AEPE  | PCK-1 (\%)  | PCK-3 (%) | PCK-5 (\%)  |
|-----------------------------|------------------------|-------|-------------|-----------|-------------|
| DGC-Net \[Melekhov2019\] |                        | 33.26 | 12.00       |           | 58.06       |
| GLU-Net           (this repo)          | static                 | 25.05 | 39.57       |   71.45        | 78.60       |
| GLU-Net           (paper)          | static                 | 25.05 | 39.55       |     -      | 78.54       |
| GLU-Net-GOCor          (this repo)     | static                 | **20.16** | 41.49       |   74.12        | **81.46**       |
| GLU-Net-GOCor          (paper)     | static                 | **20.16** | **41.55**       |    -       | 81.43       |
|---------------|------------------------|--------|--------|--------|--------|---------|---------|
| PDCNet (D)         (this repo)     | megadepth                 | 19.40 | 43.94       |    78.51       | 85.81       |
| PDCNet (H)         (this repo)     | megadepth                 | **17.51** |  **48.69**   |   **82.71**       | **89.44**       |



</details>



<details>
  <summary><b>KITTI<a name="kitti"></a></b></summary>

**Data preparation**: Both KITTI-2012 and 2015 datasets are available [here](http://www.cvlibs.net/datasets/kitti/eval_flow.php)

<br /> 

**Evaluation**: After updating the path of 'kitti2012' and 'kitti2015' in admin/local.py, evaluation is run with
```bash
python eval_matching.py --dataset kitti2015 --model PDCNet --pre_trained_models megadepth --optim_iter 3 --local_optim_iter 7 PDCNet --multi_stage_type direct
```

Similar results should be obtained:
|                |                         | KITTI-2012 |             | KITTI-2015 |           |
|----------------|-------------------------|------------|-------------|------------|-----------|
| Models         | Pre-trained model type  | AEPE       | F1   (%)    | AEPE       | F1  (%)   |
| PWC-Net-GOCor    (this repo)    | chairs-things                | 4.12      | 19.58       |  10.33     | 31.23     |
| PWC-Net-GOCor    (paper)    | chairs-things                | 4.12      | 19.31       |  10.33     | 30.53     |
| PWC-Net-GOCor    (this repo)    | chairs-things ft sintel          |    2.60  |  9.69     | 7.64       | 21.36     |
| PWC-Net-GOCor    (paper)    | chairs-things ft sintel          |    **2.60**  |  **9.67**     | **7.64**       | **20.93**     |
|----------------|-------------------------|------------|-------------|------------|-----------|
| GLU-Net    (this repo)    | static                | 3.33      | 18.91       | 9.79       | 37.77     |
| GLU-Net    (this repo)    | dynamic                 | 3.12     | 19.73       | 7.59       | 33.92     |
| GLU-Net    (paper)    | dynamic                 | 3.14       | 19.76       | 7.49       | 33.83     |
| GLU-Net-GOCor  (this repo) | dynamic                 | **2.62**       | **15.17**       | **6.63**       | 27.58     |
| GLU-Net-GOCor  (paper) | dynamic                 | 2.68       | 15.43       | 6.68       | **27.57**     |
|----------------|-------------------------|------------|-------------|------------|-----------|
| GLU-Net-GOCor* (paper) | megadepth               | 2.26       | 9.89        | 5.53       | 18.27     |
| PDC-Net **(D)**    (paper and this repo) | megadepth               | 2.08       | 7.98        | 5.22       | 15.13     |
| PDC-Net (H)    (this repo) | megadepth               | 2.16       | 8.19        | 5.31      | 15.23     |
| PDC-Net (MS)    (this repo) | megadepth               | 2.16       | 8.13        | 5.40      | 15.33     |
| PDC-Net+ **(D)**    (paper) | megadepth    | **1.76** | **6.60** | **4.53** | **12.62**  |

</details>


<details>
  <summary><b>Sintel <a name="sintel"></a></b></summary>

**Data preparation**: Download the data with 
```bash
bash assets/download_sintel.sh
```


**Evaluation**: After updating the path of 'sintel' in admin/local.py, evaluation is run with

```bash
python eval_matching.py --dataset sintel --model PDCNet --pre_trained_models megadepth --optim_iter 3 --local_optim_iter 7 --save_dir path_to_save_dir PDCNet --multi_stage_type direct
```

Similar results should be obtained:
|               | Pre-trained model type         | AEPE   | PCK-1 / dataset (\%) | PCK-5 / dataset  (\%)  | AEPE   | PCK-1  / dataset (\%) | PCK-5 / dataset  (\%)  |
|---------------|---------------------|--------|-------------|--------------|--------|-------------|--------------|
| PWC-Net-GOCor (this repo) | chairs-things          | 2.38   | 82.18       | 94.14        | 3.70   | 77.36       | 91.20        |
| PWC-Net-GOCor (paper) | chairs-things          | 2.38   | 82.17       | 94.13        | 3.70   | 77.34       | 91.20        |
| PWC-Net-GOCor (paper) | chairs-things ft sintel  | (1.74) | (87.93)     | (95.54)      | (2.28) | (84.15)     | (93.71)      |
|---------------|--------------------|--------|-------------|--------------|--------|-------------|--------------|
| GLU-Net   (this repo)     | dynamic                | 4.24   | 62.21       | 88.47        | 5.49   | 58.10       | 85.16        |
| GLU-Net    (paper)     | dynamic               | 4.25   | 62.08       | 88.40        | 5.50   | 57.85       | 85.10        |
| GLU-Net-GOCor (this repo) | dynamic                | **3.77**   | 67.11       | **90.47**        | **4.85**   | 63.36       | **87.76**        |
| GLU-Net-GOCor (paper) | dynamic                | 3.80   | **67.12**       | 90.41        | 4.90   | **63.38**       | 87.69        |
|---------------|-----------------|--------|-------------|--------------|--------|-------------|--------------|
| GLU-Net-GOCor* (paper) | megadepth         | **3.12** | 80.00 | 92.68 | **4.46** | 73.10 | 88.94 | 
| PDC-Net (D)   (this repo)     |  megadepth     | 3.30  | **85.06**      | **93.38**       | 4.48  | **78.07**       | **90.07**        |
| PDC-Net (H)   (this repo)     |  megadepth     | 3.38   | 84.95       | 93.35        | 4.50   | 77.62       | 90.07      |
| PDC-Net (MS)   (this repo)     |  megadepth     | 3.40   | 84.85       | 93.33        | 4.54   | 77.41       | 90.06      |
</details>


<details>
  <summary><b>TSS  <a name="tss"></a></b></summary>

**Data preparation**: To download the images, run:
```bash
bash assets/download_tss.sh
```
 
<br />

**Evaluation**: After updating the path of 'tss' in admin/local.py, evaluation is run with
 ```bash
python eval_matching.py --dataset TSS --model GLUNet_GOCor --pre_trained_models static --optim_iter 3 --local_optim_iter 7 --flipping_condition True --save_dir path_to_save_dir
```
Similar results should be obtained:
| Model          | Pre-trained model type      | FGD3Car | JODS | PASCAL | All  |
|--------------------------------|--------|---------|------|--------|------|
| Semantic-GLU-Net \[1\] |   Static     | 94.4    | 75.5 | 78.3   | 82.8 |
| GLU-Net (our repo)                | Static | 93.2    | 73.69 | 71.1   | 79.33 |
| GLU-Net (paper)                | Static | 93.2    | 73.3 | 71.1   | 79.2 |
| GLU-Net-GOCor (our repo, GOCor iter=3, 3)          | Static | 94.6   | 77.9 | 77.7   | 83.4 |
| GLU-Net-GOCor (our repo, GOCor iter=3, 7)          | Static | 94.6    | 77.6 | 77.1   | 83.1 |
| GLU-Net-GOCor (paper)          | Static | 94.6    | 77.9 | 77.7   | 83.4 |
| Semantic-GLU-Net  \[4\]  |  pfpascal | 95.3 | 82.2 | 78.2 | 
| WarpC-SemanticGLU-Net  | pfpascal |  **97.1** | **84.7** | **79.7** | **87.2** |

</details>

<details>
  <summary><b>PF-Pascal <a name="pfpascal"></a></b></summary>

**Data preparation**: To download the images, run:
```bash
bash assets/download_pf_pascal.sh
```
 
<br />

**Evaluation**: After updating the path of 'PFPascal' in admin/local.py, evaluation is run with
 ```bash
python eval_matching.py --dataset PFPascal --model WarpCSemanticGLUNet --pre_trained_models pfpascal --flipping_condition False --save_dir path_to_save_dir
```
Similar results should be obtained:
| Model          | Pre-trained model type     |  alpha=0.05 | alpha=0.1  
|--------------------------------|--------|---------|---------|
| Semantic-GLU-Net  \[1\]  |  static (paper) | 46.0 | 70.6 | 
| Semantic-GLU-Net  \[1\]  |  static (this repo) | 45.3 |  70.3 | 
| Semantic-GLU-Net  \[4\]  (this repo)  |  pfpascal | 48.4  | 72.4 |
| WarpC-SemanticGLU-Net  \[4\] (paper)  | pfpascal |   62.1 | **81.7** |
| WarpC-SemanticGLU-Net  \[4\]  (this repo) | pfpascal  |   **62.7** |  **81.7** | 

</details>

<details>
  <summary><b>PF-Willow <a name="pfwillow"></a></b></summary>

**Data preparation**: To download the images, run:
```bash
bash assets/download_pf_willow.sh
```
 
<br />

**Evaluation**: After updating the path of 'PFWillow' in admin/local.py, evaluation is run with
 ```bash
python eval_matching.py --dataset PFWillow --model WarpCSemanticGLUNet --pre_trained_models pfpascal --flipping_condition False --save_dir path_to_save_dir
```
Similar results should be obtained:
| Model          | Pre-trained model type     |  alpha=0.05  |  alpha=0.1  
|--------------------------------|--------|---------|---------|
| Semantic-GLU-Net  \[1\]  (paper) |  static | 36.4 | 63.8 |
| Semantic-GLU-Net  \[1\]  (this repo) |  static | 36.2 | 63.7 |
| Semantic-GLU-Net  \[4\]  |  pfpascal | 39.7 | 67.6 |
| WarpC-SemanticGLU-Net  \[4\] (paper)  | pfpascal |  **49.0** | 75.1 | 
| WarpC-SemanticGLU-Net  \[4\] (this repo)  | pfpascal |  48.9 | **75.2** | 

</details>

<details>
  <summary><b>Spair-71k <a name="spair"></a></b></summary>

**Data preparation**: To download the images, run:
```bash
bash assets/download_spair.sh
```
 
<br />

**Evaluation**: After updating the path of 'spair' in admin/local.py, evaluation is run with
 ```bash
python eval_matching.py --dataset spair --model WarpCSemanticGLUNet --pre_trained_models pfpascal  --flipping_condition False --save_dir path_to_save_dir
```
Similar results should be obtained:
| Model          | Pre-trained model type     |  alpha=0.1  
|--------------------------------|--------|---------|
| Semantic-GLU-Net  \[1\]  |  static |  15.1
| Semantic-GLU-Net  \[4\]  |  pfpascal | 16.5 |
| WarpC-SemanticGLU-Net   | spair |   23.5 |
| WarpC-SemanticGLU-Net  \[4\] | pfpascal |   **23.8** |
</details>


<details>
  <summary><b>Caltech-101 <a name="caltech"></a></b></summary>

**Data preparation**: To download the images, run:
```bash
bash assets/download_caltech.sh
```
 
<br />

**Evaluation**: After updating the path of 'spair' in admin/local.py, evaluation is run with
 ```bash
python eval_matching.py --dataset caltech --model WarpCSemanticGLUNet --pre_trained_models pfpascal  --flipping_condition False --save_dir path_to_save_dir
```
</details>





### 4.2 Pose estimation <a name="pose_estimation"></a>


Metrics are computed with
```bash
python -u eval_pose_estimation.py --dataset dataset_name --model model_name --pre_trained_models pre_trained_model_name --optim_iter optim_step  --local_optim_iter local_optim_iter --estimate_at_quarter_reso True --mask_type_for_pose_estimation proba_interval_1_above_10 --save_dir path_to_save_dir 
```


<details>
  <summary><b>YFCC100M  <a name="yfcc"></a></b></summary>
  
**Data preparation**: The groundtruth for YFCC is provided the file assets/yfcc_test_pairs_with_gt_original.txt (from [SuperGlue repo](https://github.com/magicleap/SuperGluePretrainedNetwork)). 
Images can be downloaded from the [OANet repo](https://github.com/zjhthu/OANet) and moved to the desired location
```bash
bash assets/download_yfcc.sh
```
File structure should be 
```bash
YFCC
└──  images/
       ├── buckingham_palace/
       ├── notre_dame_front_facade/
       ├── reichstag/
       └── sacre_coeur/
```

  
<br /><br />
**Evaluation**: After updating the path 'yfcc' in admin/local.py, compute metrics on YFCC100M with PDC-Net homography (H) using the command:

```bash
python -u eval_pose_estimation.py --dataset YFCC --model PDCNet --pre_trained_models megadepth --optim_iter 3  --local_optim_iter 7 --estimate_at_quarter_reso True --mask_type_for_pose_estimation proba_interval_1_above_10 --save_dir path_to_save_dir PDCNet --multi_stage_type H --mask_type proba_interval_1_above_10 
```

You should get similar metrics (not exactly the same because of RANSAC):
  
|              | mAP @5 | mAP @10 | mAP @20 | Run-time (s) |
|--------------|--------|---------|---------|--------------|
| PDC-Net (D)  | 60.52  | 70.91   | 80.30   | 0.         |
| PDC-Net (H)  | 63.90  | 73.00   | 81.22   | 0.74         |
| PDC-Net (MS) | 65.18  | 74.21   | 82.42   | 2.55         |
| PDC-Net+ (D) |  63.93 | 73.81 | 82.74 |
| PDC-Net+ (H) | **67.35** | **76.56** | **84.56** | 0.74 | 

</details>


<details>
  <summary><b>ScanNet <a name="scanNet"></a></b></summary>
  
**Data preparation**:  The images of the ScanNet test set (100 scenes, scene0707_00 to scene0806_00) are provided 
[here](https://drive.google.com/file/d/19o07SOWpv_DQcIbjb87BAKBHNCcsr4Ax/view?usp=sharing). 
They were extracted from [ScanNet github repo](https://github.com/ScanNet/ScanNet) and processed. 
We use the groundtruth provided by in the [SuperGlue repo](https://github.com/magicleap/SuperGluePretrainedNetwork) 
provided here in the file assets/scannet_test_pairs_with_gt.txt. 


<br /><br />
**Evaluation**: After updating the path 'scannet_test' in admin/local.py, compute metrics on ScanNet with PDC-Net homography (H) using the command:
```bash
python -u eval_pose_estimation.py --dataset scannet --model PDCNet --pre_trained_models megadepth --optim_iter 3  --local_optim_iter 7 --estimate_at_quarter_reso True --mask_type_for_pose_estimation proba_interval_1_above_10 --save_dir path_to_save_dir PDCNet --multi_stage_type H --mask_type proba_interval_1_above_10 
```


You should get similar metrics (not exactly the same because of RANSAC):
  
|              | mAP @5 | mAP @10 | mAP @20 |
|--------------|--------|---------|---------|
| PDC-Net (D)  | 39.93  | 50.17   | 60.87   |
| PDC-Net (H)  | 42.87  | 53.07   | 63.25   |
| PDC-Net (MS) | 42.40  | 52.83   | 63.13   | 
| PDC-Net+ (D) |  42.93 | 53.13 | 63.95 |
| PDC-Net+ (H) |  **45.66** | **56.67** | **67.07** |


</details>



### 4.3 Sparse evaluation on HPatches <a name="sparse_hp"></a>


We provide the link to the cache results [here](https://drive.google.com/drive/folders/1gphUcvBXO12EsqskdMlH3CsLxHPLtIqL?usp=sharing) 
for the sparse evaluation on HPatches. Check [PDC-Net+](https://arxiv.org/abs/2109.13912) for more details. 
    

## 5. Training <a name="Training"></a>

### Quick Start

The installation should have generated a local configuration file "admin/local.py". 
In case the file was not generated, run 
```python -c "from admin.environment import create_default_local_file; create_default_local_file()"```to generate it. 
Next, set the paths to the training workspace, i.e. the directory where the model weights and checkpoints will be saved. 
Also set the paths to the datasets you want to use (and which should be downloaded beforehand, see below). 
If all the dependencies have been correctly installed, you can train a network using the run_training.py script 
in the correct conda environment.

```bash
conda activate dense_matching_env
python run_training.py train_module train_name
```

Here, train_module is the sub-module inside train_settings and train_name is the name of the train setting file to be used.

For example, you can train using the included default train_PDCNet_stage1 settings by running:
```bash
python run_training.py PDCNet train_PDCNet_stage1
```

### Training datasets downloading <a name="scanNet"></a>

<details>
  <summary><b>DPED-CityScape-ADE </b></summary>

This is the same image pairs used in [GLU-Net repo](https://github.com/PruneTruong/GLU-Net). 
For the training, we use a combination of the DPED, CityScapes and ADE-20K datasets. 
The DPED training dataset is composed of only approximately 5000 sets of images taken by four different cameras. 
We use the images from two cameras, resulting in around  10,000 images. 
CityScapes additionally adds about 23,000 images. 
We complement with a random sample of ADE-20K images with a minimum resolution of 750 x 750. 
It results in 40.000 original images, used to create pairs of training images by applying geometric transformations to them. 
The path to the original images as well as the geometric transformation parameters are given in the csv files
'assets/csv_files/homo_aff_tps_train_DPED_CityScape_ADE.csv' and 'assets/csv_files/homo_aff_tps_test_DPED_CityScape_ADE.csv'.

1. Download the original images

* Download the [DPED dataset](http://people.ee.ethz.ch/~ihnatova/) (54 GB) ==> images are created in original_images/
* Download the [CityScapes dataset](https://www.cityscapes-dataset.com/)
    - download 'leftImg8bit_trainvaltest.zip' (11GB, left 8-bit images - train, val, and test sets', 5000 images) ==> images are created in CityScape/
    - download leftImg8bit_trainextra.zip (44GB, left 8-bit images - trainextra set, 19998 images) ==> images are created in CityScape_extra/

* Download the [ADE-20K dataset](https://drive.google.com/file/d/19r7dsYraHsNGI1ViZi4VwCfQywdODCDU/view?usp=sharing) (3.8 GB, 20.210 images) ==> images are created in ADE20K_2016_07_26/


Put all the datasets in the same directory. 
As illustration, your root training directory should be organised as follows:
```bash
training_datasets/
    ├── original_images/
    ├── CityScape/
    ├── CityScape_extra/
    └── ADE20K_2016_07_26/
```

2. Save the synthetic image pairs and flows to disk                
During training, from this set of original images, the pairs of synthetic images could be created on the fly at each epoch. 
However, this dataset generation takes time and since no augmentation is applied at each epoch, one can also create the dataset in advance
and save it to disk. During training, the image pairs composing the training datasets are then just loaded from the disk 
before passing through the network, which is a lot faster. 
To generate the training dataset and save it to disk: 

```bash
python assets/save_training_dataset_to_disk.py --image_data_path /directory/to/original/training_datasets/ 
--csv_path assets/homo_aff_tps_train_DPED_CityScape_ADE.csv --save_dir /path/to/save_dir --plot True
```    
It will create the images pairs and corresponding flow fields in save_dir/images and save_dir/flow respectively.

3. Add the paths in admin/local.py as 'training_cad_520' and 'validation_cad_520'

</details>

<details>
  <summary><b>COCO </b></summary>
  
This is useful for adding moving objects. 
Download the images along with annotations from [here](http://cocodataset.org/#download). The root folder should be
organized as follows. The add the paths in admin/local.py as 'coco'. 
```bash
coco_root
    └── annotations
        └── instances_train2014.json
    └──images
        └── train2014
```
</details>


<details>
  <summary><b> MegaDepth </b></summary>
  
We use the reconstructions provided in the [D2-Net repo](https://github.com/mihaidusmanu/d2-net).
You can download the undistorted reconstructions and aggregated scene information folder directly 
[here - Google Drive](https://drive.google.com/drive/folders/1hxpOsqOZefdrba_BqnW490XpNX_LgXPB). 

File structure should be the following:
```bash
MegaDepth
├── Undistorted_Sfm
└── scene_info
```

Them add the paths in admin/local.py as 'megadepth_training'. 

</details>




### Training scripts 

The framework currently contains the training code for the following matching networks. 
The setting files can be used train the networks, or to know the exact training details.


<details>
  <summary><b>Probabilistic Warp Consistency (PWarpC) <a name="warpc"></a></b></summary>
 
* **PWarpC.train_weakly_supervised_PWarpC_SFNet_pfpascal**: The default settings used to train the 
weakly-supervised PWarpC-SF-Net on PF-Pascal. 

* **PWarpC.train_weakly_supervised_PWarpC_SFNet_spair_from_pfpascal**: The default settings used to train the 
weakly-supervised PWarpC-SF-Net on SPair. More precisely, the network is first trained on PF-Pascal (above) and 
further finetuned on SPair-71K. 

* **PWarpC.train_strongly_supervised_PWarpC_SFNet_pfpascal**: The default settings used to train the strongly-supervised
PWarpC-SF-Net on PF-Pascal. 

* **PWarpC.train_strongly_supervised_PWarpC_SFNet_spair_from_pfpascal**: The default settings used to train the strongly-supervised
PWarpC-SF-Net on Spair-71K. 

* The rest to come

</details>




<details>
  <summary><b>Warp Consistency (WarpC) <a name="warpc"></a></b></summary>
  
* **WarpC.train_WarpC_GLUNet_stage1**: The default settings used for first stage network training without visibility mask. 
We train on real image pairs of the MegaDepth dataset. 

* **WarpC.train_WarpC_GLUNet_stage2**: We further finetune the network trained with stage1, by including our visibility mask. 
The network corresponds to our final WarpC-GLU-Net (see [WarpC paper](https://arxiv.org/abs/2104.03308)). 

* **WarpC.train_ft_WarpCSemanticGLUNet**: The default settings used for training the final WarpC-SemanticGLU-Net 
(see [WarpC paper](https://arxiv.org/abs/2104.03308)). 
We finetune the original SemanticGLUNet (trained on the static/CAD synthetic data) on PF-Pascal using Warp Consistency. 


</details>



<details>
  <summary><b>PDC-Net and PDC-Net+<a name="pdcnet"></a></b></summary>

* **PDCNet.train_PDCNet_plus_stage1**: The default settings used for first stage network training with fixed backbone weights. 
We train first on synthetically generated image pairs from the DPED, CityScape and ADE dataset (pre-computed and saved), 
on which we add MULTIPLE independently moving objects and perturbations. We also train by applying our object reprojection mask. 

* **PDCNet.train_PDCNet_plus_stage2**: The default settings used for training the final PDC-Net+ model (see [PDC-Net+ paper](https://arxiv.org/abs/2109.13912)). 
This setting fine-tunes all layers in the model trained using PDCNet_stage1 (including the feature backbone). As training
dataset, we use a combination of the same dataset than in stage 1 as well as image pairs from the MegaDepth dataset 
and their sparse ground-truth correspondence data. We also apply the reprojection mask. 



* **PDCNet.train_PDCNet_stage1**: The default settings used for first stage network training with fixed backbone weights. 
We initialize the backbone VGG-16 with pre-trained ImageNet weights. We train first on synthetically generated image 
pairs from the DPED, CityScape and ADE dataset (pre-computed and saved), on which we add independently moving objects and perturbations. 

* **PDCNet.train_PDCNet_stage2**: The default settings used for training the final PDC-Net model (see [PDC-Net paper](https://arxiv.org/abs/2101.01710)). 
This setting fine-tunes all layers in the model trained using PDCNet_stage1 (including the feature backbone). As training
dataset, we use a combination of the same dataset than in stage 1 as well as image pairs from the MegaDepth dataset 
and their sparse ground-truth correspondence data. 

* **PDCNet.train_GLUNet_GOCor_star_stage1**: Same settings than for PDCNet_stage1, with different model (non probabilistic baseline). 
The loss is changed accordingly to the L1 loss instead of the negative log likelihood loss. 

* **PDCNet.train_GLUNet_GOCor_star_stage2**: The default settings used for training the final GLU-Net-GOCor* 
(see [PDCNet paper](https://arxiv.org/abs/2101.01710)). 

</details>



<details>
  <summary><b>Example training with randomly generated data <a name="glunet"></a></b></summary>

* **GLUNet.train_GLUNet_with_synthetically_generated_data**: This is a simple example of how to generate random transformations
on the fly, and to apply them to original images, in order to create training image pairs and their corresponding 
ground-truth flow. Here, the random transformations are applied to MegaDepth images. On the created image pairs and 
ground-truth flows, we additionally add a randomly moving object. 


</details>
 
 
 

<details>
  <summary><b>GLU-Net <a name="glunet"></a></b></summary>

* **GLUNet.train_GLUNet_static**: The default settings used training the final GLU-Net (of the paper
 [GLU-Net](https://arxiv.org/abs/1912.05524)).  
We fix the  backbone weights and initialize the backbone VGG-16 with pre-trained ImageNet weights. 
We train on synthetically generated image pairs from the DPED, CityScape and ADE dataset (pre-computed and saved),
which is later ([GOCor paper](https://arxiv.org/abs/2009.07823)) referred to as 'static' dataset. 

* **GLUNet.train_GLUNet_dynamic**: The default settings used training the final GLU-Net trained on the dynamic 
dataset (of the paper [GOCor](https://arxiv.org/abs/2009.07823)).  
We fix the  backbone weights and initialize the backbone VGG-16 with pre-trained ImageNet weights. 
We train on synthetically generated image pairs from the DPED, CityScape and ADE dataset (pre-computed and saved), 
on which we add one independently moving object. 
This dataset is referred to as 'dynamic' dataset in [GOCor paper](https://arxiv.org/abs/2009.07823). 

* **GLUNet.train_GLUNet_GOCor_static**: The default settings used training the final GLU-Net-GOCor 
(of the paper [GOCor](https://arxiv.org/abs/2009.07823)).  
We fix the  backbone weights and initialize the backbone VGG-16 with pre-trained ImageNet weights. 
We train on synthetically generated image pairs from the DPED, CityScape and ADE dataset (pre-computed and saved),
which is later ([GOCor paper](https://arxiv.org/abs/2009.07823)) referred to as 'static' dataset. 

* **GLUNet.train_GLUNet_GOCor_dynamic**: The default settings used training the final GLU-Net-GOCor trained on the dynamic 
dataset (of the paper [GOCor](https://arxiv.org/abs/2009.07823)).  
We fix the  backbone weights and initialize the backbone VGG-16 with pre-trained ImageNet weights. 
We train on synthetically generated image pairs from the DPED, CityScape and ADE dataset (pre-computed and saved), 
on which we add one independently moving object. 
This dataset is referred to as 'dynamic' dataset in [GOCor paper](https://arxiv.org/abs/2009.07823). 

</details>


### Training your own networks

To train a custom network using the toolkit, the following components need to be specified in the train settings. 
For reference, see [train_GLUNet_static.py](https://github.com/PruneTruong/DenseMatching/blob/main/train_settings/GLUNet/train_GLUNet_static.py).

* Datasets: The datasets to be used for training. A number of standard matching datasets are already available in 
the datasets module. The dataset class can be passed a processing function, which should perform the necessary 
processing of the data before batching it, e.g. data augmentations and conversion to tensors.
* Dataloader: Determines how to sample the batches. Can use specific samplers. 
* Network: The network module to be trained.
* BatchPreprocessingModule: The pre-processing module that takes the batch and will transform it to the inputs 
required for training the network. Depends on the different networks and training strategies. 
* Objective: The training objective.
* Actor: The trainer passes the training batch to the actor who is responsible for passing the data through the 
network correctly, and calculating the training loss. The batch preprocessing is also done within the actor class. 
* Optimizer: Optimizer to be used, e.g. Adam.
* Scheduler: Scheduler to be used. 
* Trainer: The main class which runs the epochs and saves checkpoints.



## 6. Acknowledgement <a name="Acknowledgement"></a>

We borrow code from public projects, such as [pytracking](https://github.com/visionml/pytracking), [GLU-Net](https://github.com/PruneTruong/GLU-Net), 
[DGC-Net](https://github.com/AaltoVision/DGC-Net), [PWC-Net](https://github.com/NVlabs/PWC-Net), 
[NC-Net](https://github.com/ignacio-rocco/ncnet), [Flow-Net-Pytorch](https://github.com/ClementPinard/FlowNetPytorch), 
[RAFT](https://github.com/princeton-vl/RAFT), [CATs](https://github.com/SunghwanHong/Cost-Aggregation-transformers)...

## 7. ChangeLog <a name="changelog"></a>

* 06/21: Added evaluation code
* 07/21: Added training code and more options for evaluation
* 08/21: Fixed memory leak in mixture dataset + added other sampling for megadepth dataset
* 10/21: Added pre-trained models of WarpC 
* 12/21: Added training code for WarpC and PDC-Net+, + randomly generated data + Caltech evaluation, + pre-trained models of PDC-Net+ + demo on notebook
* 02/22: Small modifications 
* 03/22: Major refactoring, added video demos, code for PWarpC, default initialization of deconv to bilinear weights. 
