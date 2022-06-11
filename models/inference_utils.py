import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from packaging import version
import math
from scipy import ndimage
import cv2


from models.PDCNet.mod_uncertainty import (estimate_average_variance_of_mixture_density,
                                           estimate_probability_of_confidence_interval_of_mixture_density)
from utils_flow.flow_and_mapping_operations import unnormalize, normalize


def estimate_mask(mask_type, uncertainty_est, list_item=-1):
    """
    Estimates a mask of valid correspondences from the estimated uncertainty components.
    Args:
        mask_type: str, specifying what condition to use for the mask
        uncertainty_est: dict with uncertainty components. can have multiple fields such as 'log_var_map', 'weight_map',
                         'cyclic_consistency_error', 'inv_cyclic_consistency_error', 'p_r' (for PDCNet)

    Returns:
        mask: bool tensor with shape (b, h, w) when uncertainty components are (b, 1, h, w).
    """
    # all inputs should be b, 1, h, w
    # return b, h, w

    # choice = ['cyclic_consistency_error_below_x', 'x_percent_most_certain', 'variance_below_x',
    #           'proba_interval_z_above_x_NMS_y',  'proba_interval_z_above_x_grid_y', 'proba_interval_z_above_x']
    if 'cyclic_consistency_error_below' in mask_type:
        min_confidence = float(mask_type.split('below_', 1)[-1])
        if 'cyclic_consistency_error' not in uncertainty_est.keys():
            raise ValueError('Cyclic consistency error not computed! Check the arguments.')
        cyclic_consistency_error = uncertainty_est['cyclic_consistency_error']
        mask = cyclic_consistency_error.le(min_confidence).squeeze(1)
    elif 'percent_most_certain' in mask_type:
        # alternative mask
        p_r = uncertainty_est['p_r']
        b, _, h, w = p_r.shape
        uncert = p_r.view(b * h * w)
        quants = float(mask_type.split('_percent', 1)[0])

        # get percentiles for sampling and corresponding subsets
        thresholds = np.percentile(uncert.cpu().numpy(), quants)
        mask = uncert.le(thresholds).view(b, h, w)
    elif 'variance_below' in mask_type:
        min_confidence = float(mask_type.split('variance_below_', 1)[-1])
        if 'variance' in list(uncertainty_est.keys()):
            variance = uncertainty_est['variance']
        else:
            variance = estimate_average_variance_of_mixture_density(uncertainty_est['weight_map'],
                                                                    uncertainty_est['log_var_map'])
        mask = variance.le(min_confidence).squeeze(1)
    elif 'proba_interval' in mask_type and 'NMS' in mask_type:
        # ex 'proba_interval_1_above_10_NMS_4'
        info = (mask_type.split('above_', 1)[-1]).split('_NMS_', 1)
        min_confidence = float(info[0])
        size_of_NMS_window = float(info[1])
        R = float((mask_type.split('interval_', 1)[1]).split('_above_', 1)[0])
        if uncertainty_est['inference_parameters']['R'] == R:
            p_r = uncertainty_est['p_r']
        else:
            p_r = estimate_probability_of_confidence_interval_of_mixture_density(uncertainty_est['weight_map'],
                                                                                 uncertainty_est['log_var_map'], R=R)
        mask = torch.from_numpy(ndimage.maximum_filter(p_r.squeeze().cpu().numpy(), size=size_of_NMS_window)
                                .astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
        mask_1 = torch.ge(p_r, mask.float())
        mask_2 = mask.ge(min_confidence/100)
        mask = (mask_1 & mask_2).squeeze(1)
        # mask = torch.from_numpy(non_max_suppression(proba_confidence_interval.squeeze().cpu().numpy(), 3, 0.3)
        #                        .astype(np.uint8)).byte().unsqueeze(0).cuda()
    elif 'proba_interval' in mask_type and 'grid' in mask_type:
        # ex 'proba_interval_1_above_10_grid_4'
        info = (mask_type.split('above_', 1)[-1]).split('_grid_', 1)
        min_confidence = float(info[0])
        size_of_NMS_window = int(info[1])
        R = float((mask_type.split('interval_', 1)[1]).split('_above_', 1)[0])

        if uncertainty_est['inference_parameters']['R'] == R:
            p_r = uncertainty_est['p_r']
        else:
            p_r = estimate_probability_of_confidence_interval_of_mixture_density(uncertainty_est['weight_map'],
                                                                                 uncertainty_est['log_var_map'], R=R)
        mask_valid = p_r.ge(min_confidence/100).squeeze()

        h, w = p_r.shape[-2:]

        # make the grid
        XA, YA = torch.meshgrid(torch.arange(size_of_NMS_window, w - size_of_NMS_window, size_of_NMS_window),
                                torch.arange(size_of_NMS_window, h - size_of_NMS_window, size_of_NMS_window))

        YA = YA.flatten()
        XA = XA.flatten()
        valid_kp = mask_valid[YA, XA]
        mask = torch.zeros_like(mask_valid)
        mask[YA[valid_kp], XA[valid_kp]] = True
        mask = mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask.byte()
    elif 'proba_interval' in mask_type and ('NMS' not in mask_type or 'grid' not in mask_type):
        # ex 'proba_interval_1_above_10'
        min_confidence = float(mask_type.split('above_', 1)[-1])
        R = float((mask_type.split('interval_', 1)[1]).split('_above_', 1)[0])
        if 'p_r' in uncertainty_est.keys():
            if uncertainty_est['inference_parameters']['R'] == R:
                p_r = uncertainty_est['p_r']
            else:
                p_r = estimate_probability_of_confidence_interval_of_mixture_density(uncertainty_est['weight_map'],
                                                                                     uncertainty_est['log_var_map'],
                                                                                     R=R)
        else:
            if 'inv_cyclic_consistency_error' not in uncertainty_est.keys():
                raise ValueError('Cyclic consistency error not computed! Check the arguments.')
            p_r = uncertainty_est['inv_cyclic_consistency_error']

        mask = p_r.ge(min_confidence/100).squeeze(1)
    else:
        raise ValueError('unknown mask type, you selected {}'.format(mask_type))
    return mask


def matches_from_flow(flow, binary_mask, scaling=1.0):
    """
    Retrieves the pixel coordinates of 'good' matches in source and target images, based on provided flow field
    (relating the target to the source image) and a binary mask indicating where the flow is 'good'.
    Args:
        flow: tensor of shape B, 2, H, W (will be reshaped if it is not the case). Flow field relating the target
              to the source image, defined in the target image coordinate system.
        binary_mask: tensor of shape B, H, W. Boolean mask of correct correspondences.
        scaling: scalar or list of scalar (horizontal and then vertical direction):
                 scaling factor to apply to the retrieved pixel coordinates in both images.
        scaling: float, scaling factor to apply to the retrieved pixel coordinates in both images.

    Returns:
        pixel coordinates of 'good' matches in the source image, Nx2 (numpy array)
        pixel coordinates of 'good' matches in the target image, Nx2 (numpy array)
    """

    if flow.shape[1] != 2:
        flow = flow.permute(0, 3, 1, 2)

    B, _, hB, wB = flow.shape
    xx = torch.arange(0, wB).view(1, -1).repeat(hB, 1)
    yy = torch.arange(0, hB).view(-1, 1).repeat(1, wB)
    xx = xx.view(1, 1, hB, wB).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, hB, wB).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if flow.is_cuda:
        grid = grid.cuda()
        binary_mask = binary_mask.cuda()

    mapping = flow + grid
    mapping_x = mapping.permute(0, 2, 3, 1)[:, :, :, 0]
    mapping_y = mapping.permute(0, 2, 3, 1)[:, :, :, 1]
    grid_x = grid.permute(0, 2, 3, 1)[:, :, :, 0]
    grid_y = grid.permute(0, 2, 3, 1)[:, :, :, 1]

    pts2 = torch.cat((grid_x[binary_mask].unsqueeze(1), grid_y[binary_mask].unsqueeze(1)), dim=1)
    pts1 = torch.cat((mapping_x[binary_mask].unsqueeze(1), mapping_y[binary_mask].unsqueeze(1)), dim=1)
    # convert to mapping and then take the correspondences

    return pts1.cpu().numpy()*scaling, pts2.cpu().numpy()*scaling


def from_homography_to_pixel_wise_mapping(shape, H):
    """
    From a homography relating image I to image I', computes pixel wise mapping and pixel wise displacement
    between pixels of image I to image I'
    Args:
        shape: shape of image
        H: homography

    Returns:
        map_x mapping of each pixel of image I in the horizontal direction (given index of its future position)
        map_y mapping of each pixel of image I in the vertical direction (given index of its future position)
    """
    h_scale, w_scale=shape[:2]

    # estimate the grid
    X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                       np.linspace(0, h_scale - 1, h_scale))
    X, Y = X.flatten(), Y.flatten()
    # X is same shape as shape, with each time the horizontal index of the pixel

    # create matrix representation --> each contain horizontal coordinate, vertical and 1
    XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T

    # multiply Hinv to XYhom to find the warped grid
    XYwarpHom = np.dot(H, XYhom)
    Xwarp = XYwarpHom[0,:]/(XYwarpHom[2,:]+1e-8)
    Ywarp = XYwarpHom[1,:]/(XYwarpHom[2,:]+1e-8)

    # reshape to obtain the ground truth mapping
    map_x = Xwarp.reshape((h_scale,w_scale))
    map_y = Ywarp.reshape((h_scale,w_scale))
    return map_x.astype(np.float32), map_y.astype(np.float32)


def homography_is_accepted(H):
    """
    Criteria to decide if a homography is correct (not too squewed..)
    https://github.com/MasteringOpenCV/code/issues/11

    Args:
        H: homography to consider
    Returns:
        bool
            True: homography is correct
            False: otherwise
    """
    H /= H[2, 2]
    det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
    if det < 0:
        return False
    N1 = math.sqrt(H[0, 0]**2 + H[1, 0]**2)
    N2 = math.sqrt(H[0, 1]**2 + H[1, 1]**2)
    # N3 = math.sqrt(H[2, 0]**2 + H[2, 1]**2) # this criteria is too easy

    if N1 > 100 or N1 < 0.001:
        # print('rejected homo, N1 is {} and N2 is {}'.format(N1, N2))
        return False
    if N2 > 100 or N2 < 0.001:
        # print('rejected homo, N1 is {} and N2 is {}'.format(N1, N2))
        return False
    return True


def estimate_homography_and_correspondence_map(flow_estimated, binary_mask, original_shape, mapping_output_shape=None,
                                               scaling=1.0, min_nbr_points=0, ransac_thresh=1.0, device='cuda'):
    """
    Estimates homography relating the target image to the source image given the estimated flow and binary mask
    indicating where the flow is 'good'. Also computes the dense correspondence map corresponding to the
    estimated homography, with dimensions given by mapping_output_shape
    Args:
        flow_estimated: tensor of shape B, 2, H, W (will be reshaped if it is not the case). Flow field relating
                        the target to the source image, defined in the target image coordinate system.
        binary_mask: bool mask corresponding to valid flow vectors, shape B, H, W
        original_shape: shape of the original source and target images. The homopraghy corresponds to this shape
        mapping_output_shape: shape of returned correspondence map. If None, uses original_shape
        scaling: scalar or list of scalar (horizontal and then vertical direction):
                 scaling factor to apply to the retrieved pixel coordinates in both images.
        min_nbr_points: mininum number of matches for estimating the homography
        ransac_thresh: threshold used for ransac, when estimating the homography
        device:

    Returns:
        H: homography transform relating the target to the reference, at original shape
        mapping_from_homography_torch: corresponding dense correspondence map, at resolution mapping_output_shape.
                                       It is a torch tensor, of shape b, 2, mapping_output_shape[0],
                                       mapping_output_shape[1]

    """
    # estimates matching keypoints in both images
    mkpts0, mkpts1 = matches_from_flow(flow_estimated, binary_mask, scaling=scaling)
    # 0 is in source image, 1 is in target image

    original_shape = original_shape[:2]
    mapping_from_homography_torch = None
    H = None
    if len(mkpts1) > min_nbr_points:
        try:
            H, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, ransac_thresh, maxIters=3000)
            H_is_acceptable = homography_is_accepted(H)
            if H_is_acceptable:
                mapping_from_homography_x, mapping_from_homography_y = from_homography_to_pixel_wise_mapping(
                    original_shape, np.linalg.inv(H))
                mapping_from_homography_numpy = np.dstack((mapping_from_homography_x, mapping_from_homography_y))
                mapping_from_homography_torch = torch.from_numpy(mapping_from_homography_numpy)\
                    .unsqueeze(0).permute(0, 3, 1, 2)

                if mapping_output_shape is not None:
                    mapping_from_homography_torch = torch.nn.functional.interpolate(
                        input=normalize(mapping_from_homography_torch).to(device), size=mapping_output_shape, mode='bilinear',
                        align_corners=False)
                    mapping_from_homography_torch = unnormalize(mapping_from_homography_torch)
                '''
                if mapping_output_shape is not None:
                    mapping_from_homography_torch = torch.nn.functional.interpolate(
                        input=mapping_from_homography_torch.to(device), size=mapping_output_shape, mode='bilinear',
                        align_corners=False)
                    mapping_from_homography_torch[:, 0] *= float(mapping_output_shape[1]) / float(original_shape[1])
                    mapping_from_homography_torch[:, 1] *= float(mapping_output_shape[0]) / float(original_shape[0])
                '''
            else:
                # print('rejected a homography')
                H = None
        except:
            mapping_from_homography_torch = None
            H = None
    return H, mapping_from_homography_torch


def estimate_homography_and_inliers(flow_estimated, mask, scaling=1.0, min_nbr_points=0, ransac_thresh=1.0):
    """
    Estimates homography relating the target image to the source image given the estimated flow and binary mask
    indicating where the flow is 'good'.
    Args:
        flow_estimated: tensor of shape B, 2, H, W (will be reshaped if it is not the case). Flow field relating
                        the target to the source image, defined in the target image coordinate system.
        mask: bool mask corresponding to valid flow vectors, shape B, H, W
        scaling: scalar or list of scalar (horizontal and then vertical direction):
                 scaling factor to apply to the retrieved pixel coordinates in both images.
        min_nbr_points: mininum number of matches for estimating the homography
        ransac_thresh: threshold used for ransac, when estimating the homography

    Returns:
        H: homography transform relating the target to the reference, at original shape
        inliers_sum: number of inliers

    """
    # estimates matching keypoints
    mkpts0, mkpts1 = matches_from_flow(flow_estimated, mask, scaling=scaling)
    # 1 is in target image, while 0 is in source image
    H = None
    inliers_sum = 0
    if len(mkpts1) > min_nbr_points:
        homo, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, ransac_thresh, maxIters=3000)
        if homo is not None:
            H_is_acceptable = homography_is_accepted(homo)
            if H_is_acceptable:
                H = homo
                inliers_sum = inliers.sum()

    return H, inliers_sum
