import torch
import sys
import numpy as np
import importlib
import imageio
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from packaging import version

env_path = os.path.join(os.path.dirname(__file__), '../')
if env_path not in sys.path:
    sys.path.append(env_path)

from model_selection import select_model
from utils_data.io import writeFlow, writeMask, load_flo
from utils_flow.visualization_utils import draw_matches, horizontal_combine_images, draw_keypoints
from validation.utils import (resize_image, matches_from_flow, assign_flow_to_keypoints, get_mutual_matches)
from .compute_matches_at_sparse_keypoints_from_pair_list import (get_image_pair_info_, filter_matches_pydegensac,
                                                                 filter_matches_magsac)


torch.set_grad_enabled(False)


def names_to_pair_simlocmatch(img_fname0, img_fname1):
    key = '{}-{}'.format(img_fname0, img_fname1)  # must always be in that specific order
    return key


def get_matching_keypoints_coordinates(flow, mask, data_source, data_target, cfg, path_to_save, name_image,
                                       plot=False, confidence_map=None):
    """
    From flow and mask relating the target to the source, get the coordinates of the matching keypoints in both images.
    Args:
        flow: torch tensor of size b, 2, h, w
        mask: torch tensor of size b, h, w
        data_source: dict with keys 'image_original', 'size_original', 'image_resized', 'size_resized',
                     'image_resized_padded', 'size_resized_padded', 'image_resized_padded_torch'
        data_target: dict with keys 'image_original', 'size_original', 'image_resized', 'size_resized',
                     'image_resized_padded', 'size_resized_padded', 'image_resized_padded_torch'
        cfg: config, default is
            cfg = {'estimate_at_quarter_resolution': True,
                   'apply_ransac_both_ways': False, 'ransac_type': 'magsac', 'ransac_inlier_ratio': 1.0,
                   'min_nbr_matches': 30, 'min_nbr_matches_after_ransac': 15, 'scaling_kp': 2
           }
        path_to_save:
        name_image:
        confidence_map: torch tensor of size b, h, w

    Returns:
        p_source: Nx2, keypoint coordinates in source image
        p_target: Nx2, keypoint coordinates in target image
        scores: N, confidence of the match
    """

    hA, wA = data_source['size_original']
    hB, wB = data_target['size_original']

    Ish, Isw = data_source['size_resized']
    Ith, Itw = data_target['size_resized']

    if cfg.estimate_at_quarter_resolution:
        scaling = 4.0
        # if the images were not dividable by 4.0, scaling might not be exactly 4.0
        # that was the actual output of the network
        size_of_flow_padded = [h // scaling for h in data_target['size_resized_padded']]
        scaling_for_keypoints = np.float32(data_target['size_resized_padded'])[::-1] / \
                                np.float32(size_of_flow_padded)[::-1]
    else:
        scaling_for_keypoints = [1.0, 1.0]

    p_source, p_target = matches_from_flow(flow, mask, scaling=1.0)
    if confidence_map is not None:
        scores = confidence_map.squeeze()[mask.squeeze()].cpu().numpy()

    # rescale to original resolution
    p_source[:, 1] = p_source[:, 1] / float(Ish) * float(hA) * scaling_for_keypoints[1]
    p_source[:, 0] = p_source[:, 0] / float(Isw) * float(wA) * scaling_for_keypoints[0]
    p_target[:, 1] = p_target[:, 1] / float(Ith) * float(hB) * scaling_for_keypoints[1]
    p_target[:, 0] = p_target[:, 0] / float(Itw) * float(wB) * scaling_for_keypoints[0]

    '''
    p_source[:, 0] = p_source[:, 0] + .5 / float(Isw) * float(wA) * scaling_for_keypoints - .5
    p_source[:, 1] = p_source[:, 1] + .5 / float(Ish) * float(hA) * scaling_for_keypoints - .5
    p_target[:, 0] = p_target[:, 0] + .5 / float(Itw) * float(wB) * scaling_for_keypoints - .5
    p_target[:, 1] = p_target[:, 1] + .5 / float(Ith) * float(hB) * scaling_for_keypoints - .5
    '''

    if p_source.shape[0] > cfg.min_nbr_matches:
        # TODO: something for the confidence here
        if cfg.apply_ransac_both_ways and cfg.ransac_type == 'magsac':
            p_source, p_target = filter_matches_magsac(src_pts=p_source.reshape(-1, 2), dst_pts=p_target.reshape(-1, 2),
                                           min_number_of_points=cfg.min_nbr_matches_after_ransac,
                                           thresh=cfg.ransac_inlier_ratio)
        elif cfg.apply_ransac_both_ways and cfg.ransac_type == 'degensac':
            p_source, p_target = filter_matches_pydegensac(src_pts=p_source.reshape(-1, 2), dst_pts=p_target.reshape(-1, 2),
                                               min_number_of_points=cfg.min_nbr_matches_after_ransac,
                                               thresh=cfg.ransac_inlier_ratio)
    else:
        if confidence_map is not None:
            # overwrites keypoints when less than minimum number of matches
            return np.empty([0, 2], dtype=np.float32), np.empty([0, 2], dtype=np.float32),  \
                   np.empty([0], dtype=np.float32)
        else:
            # overwrites keypoints when less than minimum number of matches
            return np.empty([0, 2], dtype=np.float32), np.empty([0, 2], dtype=np.float32)

    if plot:
        Is_original = data_source['image_original']
        It_original = data_target['image_original']
        image_1 = np.uint8(Is_original)
        image_2 = np.uint8(It_original)
        n = p_source.shape[0]
        if n > 0:
            image_1 = draw_keypoints(image_1, p_source)
            image_2 = draw_keypoints(image_2, p_target)
            image_matches_gt = np.clip(
                draw_matches(image_2, image_1, p_target[int(n // 2):int(n // 2) + 2].astype(np.int32),
                             p_source[int(n // 2):int(n // 2) + 2].astype(np.int32)).astype(np.uint8), 0,
                255)
        else:
            image_matches_gt = horizontal_combine_images(image_2, image_1)
        fig, axis = plt.subplots(1, 1, figsize=(20, 20))
        axis.imshow(image_matches_gt)
        axis.set_title("image matches = {}".format(n))
        fig.savefig('{}/{}.png'.format(path_to_save, name_image),
                    bbox_inches='tight')
        plt.close(fig)
    if confidence_map is not None:
        return p_source, p_target, scores
    else:
        return p_source, p_target,


def get_matching_keypoints_for_an_image_pair(network, image0_original, image1_original, cfg, segNet=None,
                                             plot=False, save_plots_dir=None, name_of_pair_for_flow=None):
    """
    Get the coordinates of the matching keypoints between a pair of images.
    Args:
        network:
        image0_original: numpy array of shape HxWx3
        image1_original: numpy array of shape H_xW_x3
        cfg: config
            cfg = {'estimate_at_quarter_resolution': True,
           'apply_ransac_both_ways': False, 'ransac_type': 'magsac', 'ransac_inlier_ratio': 1.0,
           'min_nbr_matches': 30, 'min_nbr_matches_after_ransac': 15}
        segNet: segmentation backbone. default is None
        plot:
        save_plots_dir:
        name_of_pair_for_flow:

    Returns:
        p_source: Nx2, keypoint coordinates in source image
        p_target: Nx2, keypoint coordinates in target image
        scores: N, confidence of the match
    """
    data = get_image_pair_info_(source=image0_original, target=image1_original, cfg=cfg)

    flow_from_1_to_0, confidence_map_from_1_to_0, mask_from_1_to_0 = network.perform_matching(
        data['source'], data['target'], cfg, segNet=segNet)

    # match the other way
    if cfg.compute_matching_both_ways:
        flow_from_0_to_1, confidence_map_from_0_to_1, mask_from_0_to_1 = network.perform_matching(
            data['target'], data['source'], cfg, segNet=segNet)

    keypoints_0_flow_1_to_0, keypoints_1_flow_1_to_0, scores_from_1_to_0 = get_matching_keypoints_coordinates(
        flow_from_1_to_0, mask_from_1_to_0, data_source=data['source'],
        data_target=data['target'], cfg=cfg, path_to_save=save_plots_dir,
        name_image='{}_forward'.format(name_of_pair_for_flow), plot=plot,
        confidence_map=confidence_map_from_1_to_0)

    if cfg.compute_matching_both_ways:
        keypoints_1_flow_0_to_1, keypoints_0_flow_0_to_1, scores_from_0_to_1 = get_matching_keypoints_coordinates(
            flow_from_0_to_1, mask_from_0_to_1, data_source=data['target'],
            data_target=data['source'], cfg=cfg, path_to_save=save_plots_dir,
            name_image='{}_backward'.format(name_of_pair_for_flow), plot=plot,
            confidence_map=confidence_map_from_0_to_1)

    if cfg.compute_matching_both_ways:
        if cfg.final_matches_type == 'max':
            # for now select matches with the maximum number
            N_0_1 = keypoints_0_flow_1_to_0.shape[0]
            N_1_0 = keypoints_0_flow_0_to_1.shape[0]
            if N_0_1 > N_1_0:
                p_source = keypoints_0_flow_1_to_0
                p_target = keypoints_1_flow_1_to_0
                scores = scores_from_1_to_0

            else:
                p_source = keypoints_0_flow_0_to_1
                p_target = keypoints_1_flow_0_to_1
                scores = scores_from_0_to_1
        elif cfg.final_matches_type == 'reunion':
            # reunite matches
            p_source = np.concatenate((keypoints_0_flow_1_to_0, keypoints_0_flow_0_to_1), axis=0)
            p_target = np.concatenate((keypoints_1_flow_1_to_0, keypoints_1_flow_0_to_1), axis=0)
            scores = np.concatenate((scores_from_1_to_0.reshape(-1, 1), scores_from_0_to_1.reshape(-1, 1)), axis=0)
        elif cfg.final_matches_type == 'mutual':
            acceptable_error = cfg.threshold_mutual_matching

        else:
            raise NotImplementedError
    else:
        p_source = keypoints_0_flow_1_to_0
        p_target = keypoints_1_flow_1_to_0
        scores = scores_from_1_to_0

    if plot:
        Is_original = data['source']['image_original']
        It_original = data['target']['image_original']
        image_1 = np.uint8(Is_original)
        image_2 = np.uint8(It_original)
        n = p_source.shape[0]
        if n > 0:
            image_1 = draw_keypoints(image_1, p_source)
            image_2 = draw_keypoints(image_2, p_target)
            image_matches_gt = np.clip(
                draw_matches(image_2, image_1, p_target[int(n // 2):int(n // 2) + 10].astype(np.int32),
                             p_source[int(n // 2):int(n // 2) + 10].astype(np.int32)).astype(np.uint8), 0,
                255)
        else:
            image_matches_gt = horizontal_combine_images(image_2, image_1)
        fig, axis = plt.subplots(1, 1, figsize=(20, 20))
        axis.imshow(image_matches_gt)
        axis.set_title("image matches = {}".format(n))
        fig.savefig('{}/{}.png'.format(save_plots_dir, 'final_{}'.format(name_of_pair_for_flow)),
                    bbox_inches='tight')
        plt.close(fig)

    return p_source, p_target, scores


def retrieve_matching_keypoints_from_pair_list(args, cfg, pair_names, images_dir, name_to_pair_function,
                                               matches_dict={}, save_flow_dir=None, save_plots_dir=None):
    """
    Retrieves matching keypoints between image pairs, for all image pairs provided in the list of image pairs.
    Returns the coordinates of the keypoints for each image pairs.

    Args:
        args:
        cfg: config, check default_cfg
        pair_names: list of pair names
        images_dir:
        name_to_pair_function: function to convert image pair names to key for matches h5 file
        matches_dict: dictionary containing matches
        save_flow_dir:
        save_plots_dir:

    Returns:
        matches_dict: dictionary containing matches, where there is a key for each image pair,
                      defined by the name_to_pair_function.
                      for each pair, Nx4 for N matches, contains the keypoint coordinate in the source image,
                      followed by the keypoint coordinate in the target image
    """
    if not args.local_optim_iter:
        local_optim_iter = args.optim_iter
    else:
        local_optim_iter = int(args.local_optim_iter)

    if save_plots_dir is None:
        save_plots_dir = os.path.join(args.save_dir, '{}_{}'.format(args.model, args.pre_trained_model))
    if not os.path.isdir(save_plots_dir) and args.plot:
        os.makedirs(save_plots_dir)

    if save_flow_dir is None:
        save_flow_dir = os.path.join(args.save_dir, 'flows')
    if args.save_flow:
        os.makedirs(save_flow_dir, exist_ok=True)
    if args.load_flow:
        if not os.path.exists(save_flow_dir):
            raise ValueError('The flow path that you indicated does not exist {}'.format(save_flow_dir))

    segNet = None
    if cfg.use_segnet:
        segNet = SegNet(os.path.join(args.segnet_pretrained_dir, 'ade20k_resnet50dilated_encoder.pth'),
                        os.path.join(args.segnet_pretrained_dir, 'ade20k_resnet50dilated_decoder.pth'), 1, segFg=False)

    network, estimate_uncertainty = select_model(
        args.model, args.pre_trained_model, args, args.optim_iter, local_optim_iter,
        path_to_pre_trained_models=args.path_to_pre_trained_models)

    pbar = tqdm(enumerate(pair_names), total=len(pair_names))
    for i, pair in pbar:
        # read the images and feed them to the model
        src_fn = os.path.join(images_dir, pair.split(' ')[0])
        tgt_fn = os.path.join(images_dir, pair.split(' ')[1])
        img_fname0, img_fname1 = pair.split(' ')
        expr_module = importlib.import_module('validation.compute_matches_at_sparse_keypoints_from_pair_list')
        name_of_pair = getattr(expr_module, name_to_pair_function)(img_fname0, img_fname1)
        name_of_pair_for_flow = name_of_pair.replace('/', '-').replace(' ', '--')

        if name_of_pair in list(matches_dict.keys()):
            continue

        if not args.overwrite and args.save_flow and \
                os.path.exists(
                    os.path.join(save_flow_dir, '{}-forward-flow.flo'.format(name_of_pair_for_flow))) and \
                os.path.exists(
                    os.path.join(save_flow_dir, '{}-forward-mask.png'.format(name_of_pair_for_flow))) and \
                os.path.exists(
                    os.path.join(save_flow_dir, '{}-backward-flow.flo'.format(name_of_pair_for_flow))) and \
                os.path.exists(
                    os.path.join(save_flow_dir, '{}-backward-mask.png'.format(name_of_pair_for_flow))):
            continue

        image0_original = imageio.imread(src_fn)
        image1_original = imageio.imread(tgt_fn)

        data = get_image_pair_info_(source=image0_original, target=image1_original, cfg=cfg)

        confidence_map_from_0_to_1, confidence_map_from_1_to_0 = None, None
        if args.load_flow:
            flow_from_1_to_0 = load_flo(os.path.join(save_flow_dir,
                                                     '{}-forward-flow.flo'.format(name_of_pair_for_flow)))
            mask_from_1_to_0 = imageio.imread(os.path.join(save_flow_dir,
                                                           '{}-forward-mask.png'
                                                           .format(name_of_pair_for_flow))).astype(np.uint8)
            mask_from_1_to_0 = mask_from_1_to_0 / 255
            flow_from_1_to_0 = torch.from_numpy(flow_from_1_to_0).unsqueeze(0).permute(0, 3, 1,
                                                                                       2).cuda().float()
            mask_from_1_to_0 = torch.from_numpy(mask_from_1_to_0).unsqueeze(
                0).cuda().float()
            mask_from_1_to_0 = mask_from_1_to_0.bool() if version.parse(torch.__version__) >= version.parse("1.1") \
                else mask_from_1_to_0.byte()

        else:
            flow_from_1_to_0, confidence_map_from_1_to_0, mask_from_1_to_0 = network.perform_matching(
                data['source'], data['target'], cfg, segNet=segNet)

        if args.save_flow:
            writeFlow(flow_from_1_to_0.permute(0, 2, 3, 1).squeeze().cpu().numpy(),
                      '{}-forward-flow.flo'.format(name_of_pair_for_flow), save_flow_dir)
            writeMask(mask_from_1_to_0.squeeze().cpu().numpy(),
                      '{}-forward-mask.png'.format(name_of_pair_for_flow), save_flow_dir)

        # match the other way
        if args.compute_matching_both_ways:
            if args.load_flow:
                flow_from_0_to_1 = load_flo(os.path.join(save_flow_dir,
                                                         '{}-backward-flow.flo'.format(name_of_pair_for_flow)))
                mask_from_0_to_1 = imageio.imread(os.path.join(save_flow_dir,
                                                               '{}-backward-mask.png'.format(name_of_pair_for_flow))).astype(
                    np.uint8)
                mask_from_0_to_1 = mask_from_0_to_1 / 255
                flow_from_0_to_1 = torch.from_numpy(flow_from_0_to_1).unsqueeze(0).permute(0, 3, 1,
                                                                                           2).cuda().float()
                mask_from_0_to_1 = torch.from_numpy(mask_from_0_to_1).unsqueeze(
                    0).cuda().float()
                mask_from_0_to_1 = mask_from_0_to_1.bool() if version.parse(torch.__version__) >= version.parse("1.1")\
                    else mask_from_0_to_1.byte()
            else:
                flow_from_0_to_1, confidence_map_from_0_to_1, mask_from_0_to_1 = network.perform_matching(
                    data['target'], data['source'], cfg, segNet=segNet)

            if args.save_flow:
                writeFlow(flow_from_0_to_1.permute(0, 2, 3, 1).squeeze().cpu().numpy(),
                          '{}-backward-flow.flo'.format(name_of_pair_for_flow), save_flow_dir)
                writeMask(mask_from_0_to_1.squeeze().cpu().numpy(),
                          '{}-backward-mask.png'.format(name_of_pair_for_flow), save_flow_dir)

        if not args.save_flow:
            keypoints_0_flow_1_to_0, keypoints_1_flow_1_to_0 = get_matching_keypoints_coordinates(
                    flow_from_1_to_0, mask_from_1_to_0, data_source=data['source'],
                    data_target=data['target'], cfg=cfg, path_to_save=save_plots_dir,
                    name_image='{}_forward'.format(name_of_pair_for_flow), plot=args.plot,
                    confidence_map=confidence_map_from_1_to_0)

            if args.compute_matching_both_ways:
                keypoints_1_flow_0_to_1, keypoints_0_flow_0_to_1 = get_matching_keypoints_coordinates(
                    flow_from_0_to_1, mask_from_0_to_1, data_source=data['target'],
                    data_target=data['source'], cfg=cfg, path_to_save=save_plots_dir,
                    name_image='{}_backward'.format(name_of_pair_for_flow), plot=args.plot,
                    confidence_map=confidence_map_from_0_to_1)

            if args.compute_matching_both_ways:
                if cfg.final_matches_type == 'max':
                    # for now select matches with the maximum number
                    N_0_1 = keypoints_0_flow_1_to_0.shape[0]
                    N_1_0 = keypoints_0_flow_0_to_1.shape[0]
                    if N_0_1 > N_1_0:
                        p_source = keypoints_0_flow_1_to_0
                        p_target = keypoints_1_flow_1_to_0

                    else:
                        p_source = keypoints_0_flow_0_to_1
                        p_target = keypoints_1_flow_0_to_1
                elif cfg.final_matches_type == 'reunion':
                    # reunite matches
                    p_source = np.concatenate((keypoints_0_flow_1_to_0, keypoints_0_flow_0_to_1), axis=0)
                    p_target = np.concatenate((keypoints_1_flow_1_to_0, keypoints_1_flow_0_to_1), axis=0)
                elif cfg.final_matches_type == 'mutual':
                    acceptable_error = args.threshold_mutual_matching
                    matches = get_mutual_matches(matches_0_1, matches_1_0, keypoints_A, keypoints_B, acceptable_error)

                else:
                    raise NotImplementedError
            else:
                p_source = keypoints_0_flow_1_to_0
                p_target = keypoints_1_flow_1_to_0

            # print('\nbefore: {} matches'.format(matches.shape[0]))

            if cfg.ransac_type == 'magsac' and p_source.shape[0] > cfg.min_nbr_matches_after_ransac:
                p_source, p_target = filter_matches_magsac(src_pts=p_source.reshape(-1, 2),
                                                         dst_pts=p_target.reshape(-1, 2),
                                                         min_number_of_points=cfg.min_nbr_matches_after_ransac,
                                                         thresh=cfg.ransac_inlier_thresh)
            elif cfg.ransac_type == 'degensac' and p_source.shape[0] > cfg.min_nbr_matches_after_ransac:
                p_source, p_target = filter_matches_pydegensac(src_pts=p_source.reshape(-1, 2),
                                                             dst_pts=p_target.reshape(-1, 2),
                                                             min_number_of_points=cfg.min_nbr_matches_after_ransac,
                                                             thresh=cfg.ransac_inlier_thresh)

            output = np.concatenate([p_source, p_target], axis=1)  # Nx4
            # output = remove_duplicates(output).reshape(-1, 4)
            # here outputs are Nx4, source_pts and target_pts
            # Handling cases with no matching points found

            matches_dict[name_of_pair] = output

            if args.plot:
                Is_original = data['source']['image_original']
                It_original = data['target']['image_original']
                image_1 = np.uint8(Is_original)
                image_2 = np.uint8(It_original)

                n = p_source.shape[0]
                if n > 0:
                    image_1 = draw_keypoints(image_1, p_source)
                    image_2 = draw_keypoints(image_2, p_target)
                    image_matches_gt = np.clip(
                        draw_matches(image_2, image_1, p_target[int(n // 2):int(n // 2) + 10].astype(np.int32),
                                     p_source[int(n // 2):int(n // 2) + 10].astype(np.int32)).astype(np.uint8), 0,
                        255)
                else:
                    image_matches_gt = horizontal_combine_images(image_2, image_1)
                fig, axis = plt.subplots(1, 1, figsize=(20, 20))
                axis.imshow(image_matches_gt)
                axis.set_title("image matches = {}".format(n))
                fig.savefig('{}/{}.png'.format(save_plots_dir, 'final_{}'.format(name_of_pair_for_flow)),
                            bbox_inches='tight')
                plt.close(fig)
    return matches_dict
