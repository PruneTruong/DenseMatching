import torch
import sys
import numpy as np
import argparse
from packaging import version
import h5py
import importlib
import imageio
import cv2
import os
from tqdm import tqdm
import pydegensac
from collections import OrderedDict
torch.set_grad_enabled(False)
from pathlib import Path
from matplotlib import pyplot as plt

env_path = os.path.join(os.path.dirname(__file__), '../')
if env_path not in sys.path:
    sys.path.append(env_path)

from model_selection import select_model
from utils_flow.img_processing_utils import pad_to_same_shape
from utils_data.io import writeFlow, writeMask, load_flo
from utils_flow.visualization_utils import draw_matches, horizontal_combine_images, draw_keypoints
from validation.utils import (resize_image, matches_from_flow, assign_flow_to_keypoints, get_mutual_matches)
from validation.test_parser import boolean_string, define_model_parser
from models.external.superpoint.modified_superpoint import SUPERPOINTS


# ----------------- NAMING FUNCTIONS --------------------------------
def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))


def names_to_pair_imc(img_fname0, img_fname1):
    name0 = os.path.splitext(os.path.basename(img_fname0))[0]
    name1 = os.path.splitext(os.path.basename(img_fname1))[0]
    key = '{}-{}'.format(name0, name1)  # must always be in that specific order
    return key


def name_to_keypoint_imc(img_fname):
    return os.path.splitext(os.path.basename(img_fname))[0]


# ------------------ KEYPOINTS EXTRACTION FUNCTIONS ---------------------------------------
def get_grid_keypoints(data, cfg):
    hA, wA = data['size_original']
    scaling_kp = cfg.scaling_kp

    size_of_keypoints_s = np.int32([hA // scaling_kp, wA // scaling_kp])

    # creates all the keypoint from the original image ==> thats in resolution resized // 4 or not,
    # at each keypoitn location.
    # dense grip at pixel location
    XA, YA = np.meshgrid(np.linspace(0, size_of_keypoints_s[1] - 1, size_of_keypoints_s[1]),
                         np.linspace(0, size_of_keypoints_s[0] - 1, size_of_keypoints_s[0]))

    YA = YA.flatten()
    XA = XA.flatten()

    # put them in dimension of original image scaling
    YA = YA / float(size_of_keypoints_s[0]) * float(hA)
    XA = XA / float(size_of_keypoints_s[1]) * float(wA)

    XA = np.round(XA.reshape(-1, 1), 2)
    YA = np.round(YA.reshape(-1, 1), 2)

    # give keypoint.
    keypoints_A = np.concatenate((XA, YA), axis=1)
    return keypoints_A


class KeypointExtractor:
    """
    Class responsible for extracting keypoints from an image.
    """
    def __init__(self, cfg):
        self.extractor_name = cfg.keypoint_extractor
        if cfg.keypoint_extractor == 'superpoint':
            dict_sup = {'path_weights': '/cluster/work/cvl/truongp/deep_correspondence_search/pre_trained_'
                                        'models/superpoint_v1.pth', 'cuda': True,
                        'nms_dist': cfg.keypoint_nms,
                        'conf_thresh': 0.005}
            self.extractor_model = SUPERPOINTS(dict_sup)
        elif cfg.keypoint_extractor == 'dense_grid':
            print('Dense Grid')
        else:
            raise NotImplementedError
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_keypoints(self, image, cfg):
        if self.extractor_name == 'dense_grid':
            kp = get_grid_keypoints({'size_original': image.shape[:2]}, cfg)
        else:
            h, w = image.shape[:2]
            image_resized, scales0 = resize_image(
                image, self.device, 0.0, min_size=cfg.min_size,
                keep_original_image_when_smaller_reso=cfg.keep_original_image_when_smaller_reso)
            h_resized, w_resized = image_resized.shape[:2]
            # extract  keypoints at resized resolution
            kp, _ = self.extractor_model.find_and_describe_keypoints(image_resized)

            # rescale them to original resolution
            kp[:, 0] *= float(w) / float(w_resized)
            kp[:, 1] *= float(h) / float(h_resized)
        return kp


def extract_keypoints_and_save_h5(keypoint_extractor_module, images_dir, image_names, export_dir, name, cfg):
    feature_path = Path(export_dir, name + '_keypoints.h5')
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    feature_file = h5py.File(str(feature_path), 'a')

    print('Compute keypoints over a grid')
    pbar = tqdm(enumerate(image_names), total=len(image_names))
    for i, image_name in pbar:
        image = imageio.imread(os.path.join(images_dir, image_name))
        kp = keypoint_extractor_module.get_keypoints(image, cfg=cfg)
        grp = feature_file.create_group(image_name)
        grp.create_dataset('keypoints', data=kp)
    feature_file.close()


def extract_keypoints_from_image_list(keypoint_extractor_module, images_dir, image_names, cfg,
                                      name_to_keypoint_function=None):
    kp_dict = {}
    print('Compute keypoints over a grid')
    pbar = tqdm(enumerate(image_names), total=len(image_names))
    for i, image_name in pbar:
        image = imageio.imread(os.path.join(images_dir, image_name))
        kp = keypoint_extractor_module.get_keypoints(image, cfg=cfg)

        name_of_keypoint = image_name
        if name_to_keypoint_function is not None:
            expr_module = importlib.import_module('validation.compute_matches_at_sparse_keypoints_from_pair_list')
            name_of_keypoint = getattr(expr_module, name_to_keypoint_function)(image_name)
        kp_dict[name_of_keypoint] = kp
    return kp_dict


# ------------------------ UTILS AND RANSAC -----------------------------------------------
def get_image_list(root):
    globs = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.ppm']

    paths = []
    for g in globs:
        paths += list(Path(root).glob('**/'+g))
    if len(paths) == 0:
        raise ValueError(f'Could not find any image in root: {root}.')
    paths = [str(i.relative_to(root)) for i in paths]
    print(f'Found {len(paths)} images in root {root}.')
    return paths


def filter_matches_magsac(src_pts, dst_pts, matches=None, max_iters=25000, min_number_of_points=15, thresh=1.0):
    F, inliers_mask = cv2.findFundamentalMat(
        src_pts,
        dst_pts,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=thresh,
        confidence=0.99999,
        maxIters=max_iters)

    if matches is not None:
        dim_matches = matches.shape[1]  # index_source, index_target, can also contain score
        # return matches filtered
        if np.array(inliers_mask).sum() < min_number_of_points:
            return np.empty([0, dim_matches], dtype=np.int32)

        inliers_mask = np.array(inliers_mask).astype(bool).reshape(-1)
        return matches[inliers_mask]

    else:
        if np.array(inliers_mask).sum() < min_number_of_points:
            return np.empty([0, 2], dtype=np.int32), np.empty([0, 2], dtype=np.int32)

        inliers_mask = np.array(inliers_mask).astype(bool).reshape(-1)
        return src_pts[inliers_mask], dst_pts[inliers_mask]


def filter_matches_pydegensac(src_pts, dst_pts, matches=None, max_iters=100000, min_number_of_points=15, thresh=1.0):
    F, inliers_mask = pydegensac.findFundamentalMatrix(src_pts, dst_pts, thresh, 0.999999, max_iters)

    if matches is not None:
        dim_matches = matches.shape[1]  # index_source, index_target, can also contain score
        # return filtered matches
        if np.array(inliers_mask).sum() < min_number_of_points:
            return np.empty([0, dim_matches], dtype=np.int32)  # index_source, index_target
        inliers_mask = np.array(inliers_mask).astype(bool).reshape(-1)
        return matches[inliers_mask]
    else:
        if np.array(inliers_mask).sum() < min_number_of_points:
            return np.empty([0, 2], dtype=np.int32), np.empty([0, 2], dtype=np.int32)

        inliers_mask = np.array(inliers_mask).astype(bool).reshape(-1)
        return src_pts[inliers_mask], dst_pts[inliers_mask]


def remove_duplicates(matches):
    """
    Args:
        matches: Nx2 or Nx3, contains index of keypoints

    Returns:
        matches
    """
    final_matches = []
    kps_final_matches = []
    if matches.shape[0] == 0:
        return matches
    else:
        for i in matches.tolist():
            # take only coordinates, ignore score if there is one
            i_ = i[:2]
            if i not in kps_final_matches:
                kps_final_matches.append(i_)
                final_matches.append(i)
        matches = np.array(final_matches)
        return matches


# ----------------------- PREPROCESSING -------------------------------------------------------------
def get_image_pair_info_(source, target, cfg):
    """
    Resize and process the images as required in config.
    Args:
        source: numpy array HxWx3
        target: numpy array HxWx3
        cfg: config, must contain fields 'keep_original_image_when_smaller_reso' and 'min_size'
    Returns:
        data: dictionary with fields 'source' and 'target'. Each is a dictionary with fields
        'image_original', 'size_original', 'image_resized', 'size_resized',
        'image_resized_padded', 'size_resized_padded', 'image_resized_padded_torch'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_resized, scales0 = resize_image(
        source, device, 0.0, min_size=cfg.min_size,
        keep_original_image_when_smaller_reso=cfg.keep_original_image_when_smaller_reso)
    target_resized, scales1 = resize_image(
        target, device, 0.0, min_size=cfg.min_size,
        keep_original_image_when_smaller_reso=cfg.keep_original_image_when_smaller_reso)

    source_numpy, target_numpy = pad_to_same_shape(source_resized, target_resized)
    source_torch = torch.Tensor(source_numpy).permute(2, 0, 1).unsqueeze(0)
    target_torch = torch.Tensor(target_numpy).permute(2, 0, 1).unsqueeze(0)

    data_source = {'image_original': source, 'size_original': source.shape[:2],
                   'image_resized': source_resized, 'size_resized': source_resized.shape[:2],
                   'image_resized_padded': source_numpy, 'size_resized_padded': source_numpy.shape[:2],
                   'image_resized_padded_torch': source_torch}
    data_target = {'image_original': target, 'size_original': target.shape[:2],
                   'image_resized': target_resized, 'size_resized': target_resized.shape[:2],
                   'image_resized_padded': target_numpy, 'size_resized_padded': target_numpy.shape[:2],
                   'image_resized_padded_torch': target_torch}
    data = {'source': data_source, 'target': data_target}
    return data


# --------------------- MATCHES FROM FLOW AND KEYPOINTS -----------------------------------------
def get_matches_at_sparse_keypoints(flow, mask, data_source, data_target, keypoints_source, keypoints_target,
                                    cfg, path_to_save, name_image, plot=False, confidence_map=None):
    """
    From flow and mask relating the target to the source, get the matches in the form of index of corresponding
    keypoints. The keypoints were previously extracted (by a sparse detector) and are scaled to original resolution.
    Args:
        flow: torch tensor of size b, 2, h, w
        mask: torch tensor of size b, h, w
        data_source: dict with keys 'image_original', 'size_original', 'image_resized', 'size_resized',
                     'image_resized_padded', 'size_resized_padded', 'image_resized_padded_torch'
        data_target: dict with keys 'image_original', 'size_original', 'image_resized', 'size_resized',
                     'image_resized_padded', 'size_resized_padded', 'image_resized_padded_torch'
        cfg: config, default is
            cfg = {'min_dist_to_sparse_keypoint': 2.0, 'apply_ransac_both_ways': False, 'ransac_type': 'magsac',
                   'ransac_inlier_ratio': 1.0,
                   'min_nbr_matches': 30, 'min_nbr_matches_after_ransac': 15, 'scaling_kp': 2
           }
        path_to_save:
        name_image:
        confidence_map: torch tensor of size b, h, w

    Returns:
        matches: if confidence_map is None:
                     Nx2 for N matches, contains the index of the corresponding keypoints in source and target
                     image respectively
                 else:
                     Nx3 for N matches, contains the index of the corresponding keypoints in source and target
                     image respectively, and the confidence score
    """
    hA, wA = data_source['size_original']
    hB, wB = data_target['size_original']

    # will need to rescale the keypoint to the size of the flow.
    if cfg.estimate_at_quarter_resolution:
        size_of_keypoints_s = [int(image_shape_ // 4) for image_shape_ in data_source['size_resized']]
        size_of_keypoints_t = [int(image_shape_ // 4) for image_shape_ in data_target['size_resized']]
    else:
        size_of_keypoints_s = data_source['size_resized']
        size_of_keypoints_t = data_target['size_resized']

    assert flow.shape[-1] == size_of_keypoints_t[1] and flow.shape[-2] == size_of_keypoints_t[0]

    # keypoints now are already scaled to original images, but to assign them to the flow, we need to rescale
    # them to flow resolution, that is resized image resolution
    keypoints_source_rescaled = keypoints_source.copy()
    keypoints_source_rescaled[:, 0] *= float(size_of_keypoints_s[1]) / float(wA)
    keypoints_source_rescaled[:, 1] *= float(size_of_keypoints_s[0]) / float(hA)

    keypoints_target_rescaled = keypoints_target.copy()
    keypoints_target_rescaled[:, 0] *= float(size_of_keypoints_t[1]) / float(wB)
    keypoints_target_rescaled[:, 1] *= float(size_of_keypoints_t[0]) / float(hB)
    keypoints_target_rescaled = np.int32(np.round(keypoints_target_rescaled))

    matches = assign_flow_to_keypoints(keypoints_source_rescaled, keypoints_target_rescaled, flow.squeeze(),
                                       mask.squeeze(), confidence_map=confidence_map,
                                       min_to_kp=cfg.min_dist_to_sparse_keypoint)

    if cfg.apply_ransac_both_ways and cfg.ransac_type == 'magsac' and \
            matches.shape[0] > cfg.min_nbr_matches_after_ransac:
        matches = filter_matches_magsac(src_pts=keypoints_source[matches[:, 0].astype(np.int32)].reshape(-1, 2),
                                        dst_pts=keypoints_target[matches[:, 1].astype(np.int32)].reshape(-1, 2),
                                        matches=matches, min_number_of_points=cfg.min_nbr_matches_after_ransac,
                                        thresh=cfg.ransac_inlier_ratio)
    elif cfg.apply_ransac_both_ways and cfg.ransac_type == 'degensac' and \
            matches.shape[0] > cfg.nbr_min_matches_after_ransac:
        matches = filter_matches_pydegensac(src_pts=keypoints_source[matches[:, 0].astype(np.int32)].reshape(-1, 2),
                                            dst_pts=keypoints_target[matches[:, 1].astype(np.int32)].reshape(-1, 2),
                                            matches=matches, min_number_of_points=cfg.min_nbr_matches_after_ransac,
                                            thresh=cfg.ransac_inlier_ratio)

    # matches here are Nx2 or Nx3
    assert matches.shape[1] == 2 or matches.shape[1] == 3

    if plot:
        Is_original = data_source['image_original']
        It_original = data_target['image_original']
        image_1 = np.uint8(Is_original)
        image_2 = np.uint8(It_original)
        n = matches.shape[0]
        if n > 0:
            kp_1 = keypoints_source[matches[:, 0].astype(np.int32)].reshape(-1, 2)
            kp_2 = keypoints_target[matches[:, 1].astype(np.int32)].reshape(-1, 2)
            image_matches_gt = np.clip(
                draw_matches(image_2, image_1, kp_2[int(n // 2):int(n // 2) + 10].astype(np.int32),
                             kp_1[int(n // 2):int(n // 2) + 10].astype(np.int32)).astype(np.uint8), 0,
                255)
        else:
            image_matches_gt = horizontal_combine_images(image_2, image_1)
        fig, axis = plt.subplots(1, 1, figsize=(20, 20))
        axis.imshow(image_matches_gt)
        axis.set_title("image matches = {}".format(n))
        fig.savefig('{}/{}.png'.format(path_to_save, name_image),
                    bbox_inches='tight')
        plt.close(fig)

    return matches


def get_matches_at_keypoints_dense_grid(flow, mask, data_source, data_target, keypoints_source, keypoints_target,
                                        cfg, path_to_save, name_image, plot=False, confidence_map=None):
    """
    From flow and mask relating the target to the source, get the matches in the form of index of corresponding
    keypoints. The keypoints were previously created densely in a grid of a specific size defined in the config.
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
        matches: if confidence_map is None:
                     Nx2 for N matches, contains the index of the corresponding keypoints in source and target
                     image respectively
                 else:
                     Nx3 for N matches, contains the index of the corresponding keypoints in source and target
                     image respectively, and the confidence score
    """

    hA, wA = data_source['size_original']
    hB, wB = data_target['size_original']

    Ish, Isw = data_source['size_resized']
    Ith, Itw = data_target['size_resized']
    scaling_kp = cfg.scaling_kp

    size_of_keypoints_s = np.int32([hA // scaling_kp, wA // scaling_kp])
    size_of_keypoints_t = np.int32([hB // scaling_kp, wB // scaling_kp])

    # get the actual keypoints from the match
    # here correct because we made sure that resized image was dividable by 4
    if cfg.estimate_at_quarter_resolution:
        scaling = 4.0
        # if the images were not dividable by 4.0, scaling might not be exactly 4.0
        # that was the actual output of the network
        size_of_flow_padded = [h // scaling for h in data_target['size_resized_padded']]
        scaling_for_keypoints = np.float32(data_target['size_resized_padded'])[::-1] / \
                            np.float32(size_of_flow_padded)[::-1]
    else:
        scaling_for_keypoints = [1.0, 1.0]

    pA, pB = matches_from_flow(flow, mask, scaling=1.0)
    if pA.shape[0] < cfg.min_nbr_matches:
        # not enough matches
        matches = np.empty([0, 3], dtype=np.float32) if confidence_map is not None else np.empty([0, 2], dtype=np.int32)
    else:

        XA_match, YA_match = pA[:, 0].copy(), pA[:, 1].copy()
        XB_match, YB_match = pB[:, 0].copy(), pB[:, 1].copy()

        # scale those to size_of_keypoints
        XA_match = XA_match * size_of_keypoints_s[1] / float(Isw) * scaling_for_keypoints[0]
        YA_match = YA_match * size_of_keypoints_s[0] / float(Ish) * scaling_for_keypoints[1]
        XB_match = XB_match * size_of_keypoints_t[1] / float(Itw) * scaling_for_keypoints[0]
        YB_match = YB_match * size_of_keypoints_t[0] / float(Ith) * scaling_for_keypoints[1]

        XA_match = np.int32(np.round(XA_match))
        YA_match = np.int32(np.round(YA_match))
        XB_match = np.int32(np.round(XB_match))
        YB_match = np.int32(np.round(YB_match))

        idx_A = (YA_match * size_of_keypoints_s[1] + XA_match).reshape(-1)
        idx_B = (YB_match * size_of_keypoints_t[1] + XB_match).reshape(-1)

        #assert (keypoints_source[idx_A, 0] - XA_match.astype(np.float32) > 0).sum() == 0 #/ float(size_of_keypoints_s[1]) * float(wA)), 2).sum())
        #assert (keypoints_source[idx_A, 1] - YA_match.astype(np.float32) > 0).sum() == 0 #/ float(size_of_keypoints_s[0]) * float(hA)), 2).sum())
        #assert (keypoints_target[idx_B, 0] - XB_match.astype(np.float32) > 0).sum() == 0 #/ float(size_of_keypoints_t[1]) * float(wB)), 2).sum())
        #assert (keypoints_target[idx_B, 1] - YB_match.astype(np.float32) > 0).sum() == 0 #/ float(size_of_keypoints_t[0]) * float(hB)), 2).sum())

        matches_list = np.concatenate((idx_A.reshape(-1, 1), idx_B.reshape(-1, 1)), axis=1)
        if confidence_map is not None:
            scores = confidence_map.squeeze()[pB[:, 1], pB[:, 0]].cpu().numpy()
            matches_list = np.concatenate((matches_list, scores.reshape(-1, 1)), axis=1)
        matches = np.asarray(matches_list)

        if cfg.apply_ransac_both_ways and cfg.ransac_type == 'magsac' and \
                matches.shape[0] > cfg.min_nbr_matches_after_ransac:
            matches = filter_matches_magsac(src_pts=keypoints_source[idx_A].reshape(-1, 2),
                                            dst_pts=keypoints_target[idx_B].reshape(-1, 2),
                                            matches=matches, min_number_of_points=cfg.min_nbr_matches_after_ransac,
                                            thresh=cfg.ransac_inlier_ratio)
        elif cfg.apply_ransac_both_ways and cfg.ransac_type == 'degensac' and \
                matches.shape[0] > cfg.nbr_min_matches_after_ransac:
            matches = filter_matches_pydegensac(src_pts=keypoints_source[idx_A].reshape(-1, 2),
                                                dst_pts=keypoints_target[idx_B].reshape(-1, 2),
                                                matches=matches, min_number_of_points=cfg.min_nbr_matches_after_ransac,
                                                thresh=cfg.ransac_inlier_ratio)

    # matches here are Nx2 or Nx3
    assert matches.shape[1] == 2 or matches.shape[1] == 3

    if plot:
        Is_original = data_source['image_original']
        It_original = data_target['image_original']
        image_1 = np.uint8(Is_original)
        image_2 = np.uint8(It_original)
        n = matches.shape[0]
        if n > 0:
            kp_1 = keypoints_source[matches[:, 0].astype(np.int32)].reshape(-1, 2)
            kp_2 = keypoints_target[matches[:, 1].astype(np.int32)].reshape(-1, 2)
            image_matches_gt = np.clip(
                draw_matches(image_2, image_1, kp_2[int(n // 2):int(n // 2) + 2].astype(np.int32),
                             kp_1[int(n // 2):int(n // 2) + 2].astype(np.int32)).astype(np.uint8), 0,
                255)
        else:
            image_matches_gt = horizontal_combine_images(image_2, image_1)
        fig, axis = plt.subplots(1, 1, figsize=(20, 20))
        axis.imshow(image_matches_gt)
        axis.set_title("image matches = {}".format(n))
        fig.savefig('{}/{}.png'.format(path_to_save, name_image),
                    bbox_inches='tight')
        plt.close(fig)

    return matches


# ------------------ FINAL FUNCTIONS FOR GETTING MATCHES FROM IMAGES AND KEYPOINTS ------------------------
def retrieve_matches_at_keypoints_locations_from_pair_list(args, cfg, pair_names, images_dir, name_to_pair_function,
                                                           name_to_keypoint_function=None, kp_dict=None,
                                                           path_to_h5_keypoints=None, key_for_keypoints=None,
                                                           matches_dict={}, save_flow_dir=None, save_plots_dir=None):
    """
    Retrieves matches between each image pair specificied in a list of image pairs, with prior keypoints extracted
    densely in a grid for each image.
    Each match has shape 1x2, it contains the index of the corresponding keypoints in source and target
    images respectively. It can also contain the confidence of the match, in that case the match is 1x3.

    Args:
        args:
        cfg: config, check default_cfg
        pair_names: list of pair names
        images_dir:
        name_to_pair_function: function to convert image pair names to key for matches h5 file
        name_to_keypoint_function: function to convert image image names to key for keypoint h5 file
        kp_dict: dictionary containing keypoints for each image
        path_to_h5_keypoints: path to h5 file containing keypoints for each imahe
        key_for_keypoints: additional keys to access keypoint in kp_dict, when applicable
        matches_dict: dictionary containing matches
        save_flow_dir:
        save_plots_dir:

    Returns:
        matches_dict: dictionary containing matches, where there is a key for each image pair,
                      defined by the name_to_pair_function.
                      for each pair, Nx2 for N matches, contains the index of the corresponding keypoints
                      in source and target image respectively.
                      If a confidence value is available, Nx3, where the third value is the confidence of the match.
    """
    if not args.local_optim_iter:
        local_optim_iter = args.optim_iter
    else:
        local_optim_iter = int(args.local_optim_iter)

    if save_plots_dir is None:
        save_plots_dir = os.path.join(args.save_dir, '{}_{}'.format(args.model, args.pre_trained_model))
    if not os.path.isdir(save_plots_dir) and args.plot:
        os.makedirs(save_plots_dir)

    if path_to_h5_keypoints is not None and not args.save_flow:
        assert os.path.exists(path_to_h5_keypoints)
        kp_dict = h5py.File(path_to_h5_keypoints, 'r')

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

        name0 = img_fname0
        name1 = img_fname1
        if name_to_keypoint_function is not None:
            name0 = getattr(expr_module, name_to_keypoint_function)(img_fname0)
            name1 = getattr(expr_module, name_to_keypoint_function)(img_fname1)

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

        if not args.save_flow:
            if key_for_keypoints is None:
                keypoints_A = kp_dict[name0].__array__()
                keypoints_B = kp_dict[name1].__array__()
            else:
                keypoints_A = kp_dict[name0][key_for_keypoints].__array__()
                keypoints_B = kp_dict[name1][key_for_keypoints].__array__()

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
                mask_from_0_to_1 = mask_from_0_to_1.bool() if version.parse(torch.__version__) >= version.parse("1.1") \
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
            if cfg.keypoint_extractor == 'dense_grid':
                # get keypoints and matches
                matches_0_1 = get_matches_at_keypoints_dense_grid(
                    flow_from_1_to_0, mask_from_1_to_0, data_source=data['source'],
                    data_target=data['target'], keypoints_source=keypoints_A,
                    keypoints_target=keypoints_B, cfg=cfg, path_to_save=save_plots_dir,
                    name_image='{}_forward'.format(name_of_pair_for_flow), plot=args.plot,
                    confidence_map=confidence_map_from_1_to_0)

                if args.compute_matching_both_ways:
                    matches_1_0 = get_matches_at_keypoints_dense_grid(
                        flow_from_0_to_1, mask_from_0_to_1, data_source=data['target'],
                        data_target=data['source'], keypoints_source=keypoints_B,
                        keypoints_target=keypoints_A, cfg=cfg, path_to_save=save_plots_dir,
                        name_image='{}_backward'.format(name_of_pair_for_flow), plot=args.plot,
                        confidence_map=confidence_map_from_0_to_1)
            else:
                matches_0_1 = get_matches_at_sparse_keypoints(
                    flow_from_1_to_0, mask_from_1_to_0, data_source=data['source'],
                    data_target=data['target'], keypoints_source=keypoints_A,
                    keypoints_target=keypoints_B, cfg=cfg, path_to_save=save_plots_dir,
                    name_image='{}_forward'.format(name_of_pair_for_flow), plot=args.plot,
                    confidence_map=confidence_map_from_1_to_0)

                if args.compute_matching_both_ways:
                    matches_1_0 = get_matches_at_sparse_keypoints(
                        flow_from_0_to_1, mask_from_0_to_1, data_source=data['target'],
                        data_target=data['source'], keypoints_source=keypoints_B,
                        keypoints_target=keypoints_A, cfg=cfg, path_to_save=save_plots_dir,
                        name_image='{}_backward'.format(name_of_pair_for_flow), plot=args.plot,
                        confidence_map=confidence_map_from_0_to_1)

            if args.compute_matching_both_ways:
                if cfg.final_matches_type == 'max':
                    # for now select matches with the maximum number
                    N_0_1 = matches_0_1.shape[0]
                    N_1_0 = matches_1_0.shape[0]
                    if N_0_1 > N_1_0:
                        matches = matches_0_1
                    else:
                        # put index of image 0 first
                        if matches_1_0.shape[1] == 3:
                            # also contains the score of the match
                            matches = matches_1_0[:, [1, 0, 2]]
                        else:
                            matches = matches_1_0[:, [1, 0]]
                elif cfg.final_matches_type == 'reunion':
                    # reunite matches, put indexes in the correct order
                    if matches_1_0.shape[1] == 3:
                        # also contains the score of the match
                        matches_1_0 = matches_1_0[:, [1, 0, 2]]
                    else:
                        matches_1_0 = matches_1_0[:, [1, 0]]
                    matches = np.concatenate((matches_0_1, matches_1_0), axis=0)
                    assert matches.shape[0] == matches_0_1.shape[0] + matches_1_0.shape[0]
                elif cfg.final_matches_type == 'mutual':
                    acceptable_error = args.threshold_mutual_matching
                    matches = get_mutual_matches(matches_0_1, matches_1_0, keypoints_A, keypoints_B, acceptable_error)

                else:
                    raise NotImplementedError
            else:
                matches = mask_from_1_to_0

            # print('\nbefore: {} matches'.format(matches.shape[0]))

            if cfg.ransac_type == 'magsac' and matches.shape[0] > cfg.min_nbr_matches_after_ransac:
                matches = filter_matches_magsac(src_pts=keypoints_A[matches[:, 0].astype(np.int32)].reshape(-1, 2),
                                                dst_pts=keypoints_B[matches[:, 1].astype(np.int32)].reshape(-1, 2),
                                                matches=matches,
                                                min_number_of_points=cfg.min_nbr_matches_after_ransac,
                                                thresh=cfg.ransac_inlier_thresh)
            elif cfg.ransac_type == 'degensac' and matches.shape[0] > cfg.min_nbr_matches_after_ransac:
                matches = filter_matches_pydegensac(src_pts=keypoints_A[matches[:, 0].astype(np.int32)].reshape(-1, 2),
                                                    dst_pts=keypoints_B[matches[:, 1].astype(np.int32)].reshape(-1, 2),
                                                    matches=matches,
                                                    min_number_of_points=cfg.min_nbr_matches_after_ransac,
                                                    thresh=cfg.ransac_inlier_thresh)

            matches = remove_duplicates(matches)  # .astype(np.int32), now there might be confidence value too
            matches_dict[name_of_pair] = matches

            if args.plot:
                Is_original = data['source']['image_original']
                It_original = data['target']['image_original']
                image_1 = np.uint8(Is_original)
                image_2 = np.uint8(It_original)

                n = matches.shape[0]
                if n > 0:
                    kp_1 = keypoints_A[matches[:, 0].astype(np.int32)].reshape(-1, 2)
                    kp_2 = keypoints_B[matches[:, 1].astype(np.int32)].reshape(-1, 2)
                    if args.keypoint_extractor != 'dense_grid':
                        image_1 = draw_keypoints(image_1, kp_1)
                        image_2 = draw_keypoints(image_2, kp_2)

                    image_matches_gt = np.clip(
                        draw_matches(image_2, image_1,
                                     kp_2[-int(n // 5):-int(n // 5) + 2].astype(np.int32),
                                     kp_1[-int(n // 5):-int(n // 5) + 2].astype(np.int32)).astype(
                            np.uint8),
                        0,
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


default_cfg = OrderedDict(
            {'keep_original_image_when_smaller_reso': False, 'min_size': 480,

           # matching
           'compute_matching_both_ways': True,
           'estimate_at_quarter_resolution': True, 'use_segnet': False,
           'segnet_pretrained_dir': '',
           'mask_type_for_pose_estimation': 'proba_interval_1_above_10',
           'apply_ransac_both_ways': False, 'ransac_type': 'magsac', 'ransac_inlier_thresh': 1.0,
           'min_nbr_matches': 30, 'min_nbr_matches_after_ransac': 15,

           'scaling_kp': 2,
           'final_matches_type': 'reunion'
    })


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_mode', type=str, default='euler',
                        help='data mode, only important for euler cluster')
    parser.add_argument('--image_pairs', type=str, default='all')
    parser.add_argument('--nchunks', type=int, default=1)
    parser.add_argument('--chunk_idx', type=int, default=0)
    parser.add_argument('--skip_up_to', type=int, default=0)

    parser.add_argument('--save_flow', default=False, type=boolean_string,
                        help='save flow')
    parser.add_argument('--overwrite', default=False, type=boolean_string,
                        help='save flow')
    parser.add_argument('--load_flow', default=False, type=boolean_string,
                        help='save flow')
    parser.add_argument('--flow_dir', type=str, default=None)
    parser.add_argument('--save_matches_files', default=True, type=boolean_string,
                        help='plot? default is False')
    parser.add_argument('--save_dir', type=str, default='evaluation/',
                        help='path to directory to save the text files and results')

    define_model_parser(parser)  # model parameters
    parser.add_argument('--pre_trained_model', type=str, default='CityScape_DPED_ADE',
                        help='which pre trained model to use')
    parser.add_argument('--use_segnet', default=False, type=boolean_string,
                        help='apply segmentation backbone as additional filtering for the matches? default is False')
    parser.add_argument('--segnet_pretrained_dir', type=str,
                        default='/scratch_net/pelle/truongp/code/pre_trained_models/SegNet',
                        help='Path to the SegNet pretrained models directory')

    parser.add_argument('--min_size', type=int, default=0, help='Resize image so that minimum dimension is min_size.')
    parser.add_argument('--keep_original_image_when_smaller_reso', default=False, type=boolean_string)
    parser.add_argument('--keypoint_extractor', type=str, default='dense_grid',
                        help='keypoint extractor name')
    parser.add_argument('--scaling_kp', type=int, default=2, help='for dense grid computation, from original resolution')
    parser.add_argument('--keypoint_nms', type=int, default=4, help='NMS used for sparse keypoints')
    parser.add_argument('--min_dist_to_sparse_keypoint', type=float, default=2, help='min_dist_to_sparse_keypoint')

    parser.add_argument('--compute_matching_both_ways', default=True, type=boolean_string,
                        help='compute_matching_both_ways? default is True')
    parser.add_argument('--estimate_at_quarter_resolution', default=True, type=boolean_string,
                        help='estimate matches from quarter resolution flow (output of network)? default is True')
    parser.add_argument('--mask_type_for_pose_estimation', type=str,
                        help='mask_type_for_pose_estimation')
    parser.add_argument('--ransac_type', default=None, choices=['None', 'magsac', 'degensac'],
                        help="using ransac to filter the outlier or not (default None )")
    parser.add_argument('--final_matches_type', default='max', choices=['max', 'reunion', 'mutual'])
    parser.add_argument('--threshold_mutual_matching', type=float, default=3.0,
                        help="threshold_mutual_matching")
    parser.add_argument('--min_nbr_matches', type=int, default=15,
                        help="minimum number of required matches")
    parser.add_argument('--min_nbr_matches_after_ransac', type=int, default=30,
                        help="minimum number of required matches after ransac")
    parser.add_argument('--ransac_inlier_thresh', type=float, default=0.5)
    parser.add_argument('--apply_ransac_both_ways', default=False, type=boolean_string,
                        help='apply_ransac_both_ways? default is False')

    parser.add_argument('--plot', default=False, type=boolean_string,
                        help='plot? default is False')
    return parser


if __name__ == "__main__":
    """
    cfg = OrderedDict(
            {'keep_original_image_when_smaller_reso': False, 'min_size': 480,

           # matching
           'compute_matching_both_ways': True, 
           'estimate_at_quarter_resolution': True, 'use_segnet': False, 
           'segnet_pretrained_dir': '', 
           'mask_type_for_pose_estimation': 'proba_interval_1_above_10',
           'apply_ransac_both_ways': False, 'ransac_type': 'magsac', 'ransac_inlier_thresh': 1.0,
           'min_nbr_matches': 30, 'min_nbr_matches_after_ransac': 15,

           'scaling_kp': 2,
           'final_matches_type': 'reunion'
    }
    """


