from pathlib import Path
import argparse
import numpy as np
import torch
from tqdm import tqdm
import os
import json

from admin.stats import merge_dictionaries
import admin.settings as ws_settings
from utils_flow.img_processing_utils import pad_to_same_shape
from model_selection import select_model
from validation.utils import matches_from_flow
from models.inference_utils import estimate_mask
from utils_flow.flow_and_mapping_operations import convert_flow_to_mapping
from validation.utils import (compute_pose_error, compute_epipolar_error,
                              estimate_pose, pose_auc, read_image,
                              rotate_intrinsics, rotate_pose_inplane,
                              scale_intrinsics)
from validation.test_parser import define_model_parser, boolean_string

torch.set_grad_enabled(False)


def main(args, settings):
    min_size = args.minSize

    save_dir = os.path.join(args.save_dir, 'mask_for_pose_est_' + args.mask_type_for_pose_estimation)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if args.network_type == 'PDCNet' and ('d' not in args.multi_stage_type.lower()):
        # add sub-possibility with mask threshold of internal multi-stage alignment
        save_dir = os.path.join(save_dir, 'mask_for_multi_stage_align_' + args.mask_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    save_dict = {}
    for pre_trained_model_type in args.pre_trained_models:
        print(pre_trained_model_type)
        network, estimate_uncertainty = select_model(
            args.model, pre_trained_model_type, args, args.optim_iter, local_optim_iter,
            path_to_pre_trained_models=args.path_to_pre_trained_models)

        pbar = tqdm(enumerate(pairs), total=len(pairs))
        pose_errors = []
        percentage_of_correct_points = []
        output = {}
        run_time = 0
        for i, pair in pbar:
            name0, name1 = pair[:2]
            # If a rotation integer is provided (e.g. from EXIF data), use it:
            if len(pair) >= 5:
                rot0, rot1 = int(pair[2]), int(pair[3])
            else:
                rot0, rot1 = 0, 0

            # Load the image pair.
            # scales here is ratio between new load_size and old load_size
            # for now we resize keeping aspect ratio
            image0, scales0 = read_image(
                input_dir / name0, device, rot0, resize_float=False, min_size=min_size, resize=args.resize)
            image1, scales1 = read_image(
                input_dir / name1, device, rot1, resize_float=False, min_size=min_size, resize=args.resize)
            if image0 is None or image1 is None:
                print('Problem reading image pair: {} {}'.format(
                    input_dir/name0, input_dir/name1))
                exit(1)

            Ish, Isw, _ = image0.shape
            Ith, Itw, _ = image1.shape
            Is_numpy, It_numpy = pad_to_same_shape(image0, image1)
            Is = torch.Tensor(Is_numpy).permute(2, 0, 1).unsqueeze(0)
            It = torch.Tensor(It_numpy).permute(2, 0, 1).unsqueeze(0)

            if args.estimate_at_quarter_reso:
                size_of_flow_padded = (It_numpy.shape[0] // 4, It_numpy.shape[1] // 4)
                size_of_flow = (Ith // 4, Itw // 4)
                size_of_source = (Ish // 4, Isw // 4)
                # scaling first in horizontal direction, then vertical
                scaling = 4.0
                scaling_for_keypoints = np.float32([Ith, Itw])[::-1] / np.float32(size_of_flow)[::-1]
            else:
                size_of_flow_padded = It_numpy.shape[:2]
                size_of_flow = image1.shape[:2]
                size_of_source = (Ish, Isw)
                scaling = 1.0
                scaling_for_keypoints = 1.0

            mask_padded = torch.ones(size_of_flow_padded).unsqueeze(0).byte().to(device)
            mask_padded = mask_padded.bool() if float(torch.__version__[:3]) >= 1.1 else mask_padded.byte()

            # scaling defines the final outputted shape by the network.
            if not estimate_uncertainty:
                flow_estimated_padded = network.estimate_flow(Is, It, scaling=1.0/scaling)
            else:
                flow_estimated_padded,  uncertainty_est_padded = network.estimate_flow_and_confidence_map(Is, It,
                                                                                                          scaling=1.0/scaling)
                if 'warping_mask' in list(uncertainty_est_padded.keys()):
                    # get mask from internal multi stage alignment, if it took place
                    mask_padded = uncertainty_est_padded['warping_mask'] * mask_padded

                # get the mask according to uncertainty estimation
                mask_padded = estimate_mask(args.mask_type_for_pose_estimation, uncertainty_est_padded) \
                            * mask_padded

            # remove the padding
            flow = flow_estimated_padded[:, :, :size_of_flow[0], :size_of_flow[1]]
            mask = mask_padded[:, :size_of_flow[0], :size_of_flow[1]]
            mapping_estimated = convert_flow_to_mapping(flow)
            # remove point that lead to outside the source image
            mask = mask & mapping_estimated[:, 0].ge(0) & mapping_estimated[:, 1].ge(0) & \
                mapping_estimated[:, 0].le(size_of_source[1] - 1) & mapping_estimated[:, 1].le(size_of_source[0] - 1)

            mkpts0, mkpts1 = matches_from_flow(flow, mask, scaling=scaling_for_keypoints)
            # Estimate the pose and compute the pose error.
            assert len(pair) == 38, 'Pair does not have ground truth info'
            K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
            K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
            T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

            # Scale the intrinsics to resized image.
            K0 = scale_intrinsics(K0, scales0)
            K1 = scale_intrinsics(K1, scales1)

            # Update the intrinsics + extrinsics if EXIF rotation was found.

            if rot0 != 0 or rot1 != 0:
                print('pair {}, update intrinsic'.format(i))
                cam0_T_w = np.eye(4)
                cam1_T_w = T_0to1  # rotation from camera 0 to camera 1
                if rot0 != 0:
                    K0 = rotate_intrinsics(K0, image0.shape, rot0)
                    cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
                if rot1 != 0:
                    K1 = rotate_intrinsics(K1, image1.shape, rot1)
                    cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
                cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
                T_0to1 = cam1_T_cam0  # rotation from camera 0 to camera 1, ie from source to target

            epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
            correct = epi_errs < 5e-4
            num_correct = np.sum(correct)
            precision = np.mean(correct) if len(correct) > 0 else 0

            thresh = 1.  # In pixels relative to resized image load_size.
            ret = estimate_pose(mkpts0, mkpts1, K0, K1, ransac=args.ransac, thresh=thresh)
            # estimates relative pose from camera 0/source to camera 1/target

            # corresponds to Rotation from camera 0 to camera 1
            if ret is None:
                err_t, err_R = np.inf, np.inf
            else:
                R, t, inliers = ret
                err_t, err_R = compute_pose_error(T_0to1, R, t)

            # Write the evaluation results to disk.
            out_eval = {'error_t': err_t,
                        'error_R': err_R,
                        'num_correct': num_correct,
                        'percentage_of_correct': precision,
                        'epipolar_errors': epi_errs}

            pose_error = np.maximum(out_eval['error_t'], out_eval['error_R'])
            pose_errors.append(pose_error)
            percentage_of_correct_points.append(out_eval['percentage_of_correct'])

        # compute the average !
        thresholds = [5, 10, 20]
        aucs = pose_auc(pose_errors, thresholds)
        aucs = [100. * yy for yy in aucs]
        print('Evaluation Results (mean over {} pairs):'.format(len(pairs)))
        print('AUC@5\t AUC@10\t AUC@20\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t '.format(aucs[0], aucs[1], aucs[2]))

        Acc5 = np.sum(np.array(pose_errors) < 5.0) / float(len(pose_errors))
        Acc10 = np.sum(np.array(pose_errors) < 10.0) / float(len(pose_errors))
        Acc15 = np.sum(np.array(pose_errors) < 15.0) / float(len(pose_errors))
        Acc20 = np.sum(np.array(pose_errors) < 20.0) / float(len(pose_errors))
        mAP10 = np.mean([Acc5, Acc10])
        mAP20 = np.mean([Acc5, Acc10, Acc15, Acc20])

        print('mAP@5: {}'.format(Acc5))
        print('mAP@10: {}'.format(mAP10))
        print('mAP@20: {}'.format(mAP20))

        output['AUC@5'] = aucs[0]
        output['AUC@10'] = aucs[1]
        output['AUC@20'] = aucs[2]
        output['Acc@5'] = Acc5
        output['Acc@10'] = Acc10
        output['Acc@15'] = Acc15
        output['Acc@20'] = Acc20

        output['mAP@5'] = Acc5
        output['mAP@10'] = mAP10
        output['mAP@20'] = mAP20
        output['percentage_of_correct'] = np.mean(percentage_of_correct_points)
        output['run_time'] = run_time

        save_dict[pre_trained_model_type] = output

    name_to_save = args.model
    if 'gocor' in args.model.lower() or 'PDCNet' in args.model:
        name_save_metrics = 'metrics_{}_iter_{}_{}'.format(name_to_save, args.optim_iter, local_optim_iter)
    else:
        name_save_metrics = 'metrics_{}'.format(name_to_save)

    path_file = '{}/{}.txt'.format(save_dir, name_save_metrics)
    if os.path.exists(path_file):
        with open(path_file, 'r') as outfile:
            save_dict_existing = json.load(outfile)
        save_dict = merge_dictionaries([save_dict_existing, save_dict])

    with open(path_file, 'w') as outfile:
        json.dump(save_dict, outfile, ensure_ascii=False, separators=(',', ':'))
        print('written to file ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pose estimation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, help='Dataset name', required=True)

    define_model_parser(parser)
    parser.add_argument('--pre_trained_models', nargs='+', required=True,
                        help='Names of the pre trained models')

    parser.add_argument('--minSize', type=int, default=480,
                        help='Resize images so that the minimum load_size is this argument, '
                             'while keeping the original aspect ratio')
    parser.add_argument('--resize', type=int, nargs='+', default=[640, 480],
                        help='Resize the input image before running inference. If two numbers, resize to the exact '
                             'dimensions, if one number, resize the max dimension, if -1, do not resize')
    # estimation parameter
    parser.add_argument('--estimate_at_quarter_reso', default=True, type=boolean_string,
                        help='estimate the relative pose from the flow field at quarter reso ? default is True')
    parser.add_argument('--mask_type_for_pose_estimation', default='proba_interval_1_above_10', type=str,
                        help='mask type for pose_estimation from uncertainty prediction')

    # ransac
    parser.add_argument('--ransac', default=True, type=boolean_string,
                        help="using ransac to filter the outlier or not (default True )")
    parser.add_argument('--ransac_thresh', type=float, default=1.0, help='threshold used for RANSAC')
    parser.add_argument('--save_dir', type=str, default='evaluation/',
                        help='path to directory to save the results')
    args = parser.parse_args()
    local_optim_iter = int(args.local_optim_iter) if args.local_optim_iter else args.optim_iter
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running inference on device \"{}\"'.format(device))

    # Create the output directories if they do not exist already.
    settings = ws_settings.Settings()

    if args.dataset == 'YFCC':
        input_dir = Path(settings.env.yfcc)
        input_pairs = 'assets/yfcc_test_pairs_with_gt_original.txt'
    elif args.dataset == 'scannet':
        input_dir = Path(settings.env.scannet_test)
        input_pairs = 'assets/scannet_test_pairs_with_gt.txt'
    else:
        raise NotImplementedError

    with open(input_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]

    if not all([len(p) == 38 for p in pairs]):
        raise ValueError(
            'All pairs should have ground truth info for evaluation.'
            'File \"{}\" needs 38 valid entries per row'.format(args.input_pairs))

    print('Looking for data in directory \"{}\"'.format(input_dir))
    main(args, settings)
