import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
from imageio import imread
import math
from skimage.feature import peak_local_max

from utils_flow.flow_and_mapping_operations import convert_flow_to_mapping, normalize, unnormalize
from utils_data.geometric_transformation_sampling.homography_parameters_sampling import from_homography_to_pixel_wise_mapping


# --- PREPROCESSING ---
def process_resize(w, h, resize):
    assert (len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def get_new_resolution_with_minimum(minSize, I, strideNet, keep_original_image_when_smaller_reso=False):
    h, w = I.shape[:2]
    if min(h, w) < minSize and keep_original_image_when_smaller_reso:
        new_w, new_h = w // strideNet * strideNet, h // strideNet * strideNet
    else:
        ratio = min(w / float(minSize), h / float(minSize))
        new_w, new_h = round(w / ratio), round(h / ratio)
        new_w, new_h = new_w // strideNet * strideNet, new_h // strideNet * strideNet
    return new_w, new_h


def resize_image(image, device, rotation, resize_float=False, resize=None, min_size=None, strideNet=8,
                 keep_original_image_when_smaller_reso=False):

    w, h = image.shape[1], image.shape[0]
    if min_size is not None:
        # it means we need to resize the image keeping aspect ratio so that smallest side is equal to min_size
        w_new, h_new = get_new_resolution_with_minimum(min_size, image, strideNet,
                                                       keep_original_image_when_smaller_reso=keep_original_image_when_smaller_reso)
        # print('Will resize to {}x{} (WxH)'.format(w_new, h_new))
    else:
        '''
        if len(resize) == 2 and resize[1] == -1:
            resize = resize[0:1]
        if len(resize) == 2:
            print('Will resize to {}x{} (WxH)'.format(resize[0], resize[1]))
        elif len(resize) == 1 and resize[0] > 0:
            print('Will resize max dimension to {}'.format(resize[0]))
        elif len(resize) == 1:
            print('Will not resize images')
        else:
            raise ValueError('Cannot specify more than two integers for --resize')
        '''
        w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new))

    if rotation != 0:
        image = np.rot90(image, k=rotation).copy()
        # needs the copy for later to be read in torch !
        if rotation % 2:
            scales = scales[::-1]
    return image, scales


def read_image(path, device, rotation, resize_float=False, resize=None, min_size=None, strideNet=8,
               keep_original_image_when_smaller_reso=False):
    image = imread(str(path))
    if image is None:
        return None, None, None

    return resize_image(image, device, rotation, resize_float, resize=resize, min_size=min_size, strideNet=strideNet,
                        keep_original_image_when_smaller_reso=keep_original_image_when_smaller_reso)


# --- GEOMETRY ---
def estimate_pose(kpts0, kpts1, K0, K1, ransac, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    if ransac:
        E, mask = cv2.findEssentialMat(
            kpts0, kpts1, np.eye(3), threshold=norm_thresh,
            prob=conf,
            method=cv2.RANSAC)
    else:
        E, mask = cv2.findFundamentalMat(
            kpts0, kpts1,  method=cv2.FM_8POINT
        )

    ret = None
    if E is not None:
        best_num_inliers = 0

        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(
                _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


def rotate_intrinsics(K, image_shape, rot):
    """image_shape is the shape of the image after rotation"""
    assert rot <= 3
    h, w = image_shape[:2][::-1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 1:
        return np.array([[fy, 0., cy],
                         [0., fx, w-1-cx],
                         [0., 0., 1.]], dtype=K.dtype)
    elif rot == 2:
        return np.array([[fx, 0., w-1-cx],
                         [0., fy, h-1-cy],
                         [0., 0., 1.]], dtype=K.dtype)
    else:  # if rot == 3:
        return np.array([[fy, 0., h-1-cy],
                         [0., fx, cx],
                         [0., 0., 1.]], dtype=K.dtype)


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array([[np.cos(r), -np.sin(r), 0., 0.],
                  [np.sin(r), np.cos(r), 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]], dtype=np.float32)
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    scales = np.diag([1./scales[0], 1./scales[1], 1.])
    return np.dot(scales, K)


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2)
                    + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))
    return d


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs


def non_max_suppression(image, size_filter, proba):
    non_max = peak_local_max(image, min_distance=size_filter, threshold_abs=proba, \
                      exclude_border=True, indices=False)
    '''
    kp = np.where(non_max>0)
    
    if len(kp[0]) != 0:
        for i in range(len(kp[0]) ):

            window=non_max[kp[0][i]-size_filter:kp[0][i]+(size_filter+1), \
                           kp[1][i]-size_filter:kp[1][i]+(size_filter+1)]
            if np.sum(window)>1:
                window[:,:]=0
    '''
    return non_max


# --- MATCHES FROM FLOW UTILS ---
def matches_from_flow(flow, binary_mask, scaling=1.0):
    """
    Retrieves the pixel coordinates of 'good' matches in source and target images, based on provided flow field
    (relating the target to the source image) and a binary mask indicating where the flow is 'good'.
    Args:
        flow: tensor of shape B, 2, H, W (will be reshaped if it is not the case). Flow field relating the target
              to the source image, defined in the target image coordinate system.
        binary_mask: bool mask corresponding to valid flow vectors, shape B, H, W
        scaling: scalar or list of scalar (horizontal and then vertical direction):
                 scaling factor to apply to the retrieved pixel coordinates in both images.

    Returns:
        pixel coordinates of 'good' matches in the source image, Nx2 (numpy array)
        pixel coordinates of 'good' matches in the target image, Nx2 (numpy array)
    """

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

    pts2 = torch.cat((grid_x[binary_mask].unsqueeze(1),
                      grid_y[binary_mask].unsqueeze(1)), dim=1)
    pts1 = torch.cat((mapping_x[binary_mask].unsqueeze(1),
                      mapping_y[binary_mask].unsqueeze(1)),
                     dim=1)  # convert to mapping and then take the correspondences

    return pts1.cpu().numpy()*scaling, pts2.cpu().numpy()*scaling


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
                        input=normalize(mapping_from_homography_torch).to(device), size=mapping_output_shape,
                        mode='bilinear', align_corners=False)
                    mapping_from_homography_torch = unnormalize(mapping_from_homography_torch)
            else:
                print('rejected a homography')
                H = None
        except:
            mapping_from_homography_torch = None
            H = None
    return H, mapping_from_homography_torch


def assign_flow_to_keypoints(kp_source, kp_target, flow, mask_valid, confidence_map=None, min_to_kp=3.0):
    """

    Args:
        kp_source: numpy array Nx2, keypoints extracted in the source image
        kp_target: numpy array Nx2, keypoints extracted in the target image
        flow: flow relating the target image to the source, HxWx2.
              The resolution should be the same than the keypoint resolution
        mask_valid: mask of valid matches. HxW
        min_to_kp: minimum in pixels to assign a flow vector to a keypoint.
        confidence_map:

    Returns:
        matches: if confidence_map is None:
                     Nx2 for N matches, contains the index of the corresponding keypoints in source and target
                     image respectively
                 else:
                     Nx3 for N matches, contains the index of the corresponding keypoints in source and target
                     image respectively, and the confidence score

    """
    if isinstance(flow, torch.Tensor):
        flow = flow.cpu().numpy()
    if isinstance(mask_valid, torch.Tensor):
        mask_valid = mask_valid.squeeze().cpu().numpy()
    if confidence_map is not None and isinstance(confidence_map, torch.Tensor):
        confidence_map = confidence_map.squeeze().cpu().numpy()

    n_s = len(kp_source)
    n_t = len(kp_target)
    index_kp_target_all_ = np.arange(0, n_t, 1)

    # consider only keypoints that are in the mask
    list_kp_target_in_mask = mask_valid[kp_target[index_kp_target_all_, 1], kp_target[index_kp_target_all_, 0]]
    index_kp_target_all = index_kp_target_all_[list_kp_target_in_mask]
    n_valid = len(index_kp_target_all)

    mapping = convert_flow_to_mapping(flow, output_channel_first=False)  # supposed to be HxWx2
    p_source_est = mapping[kp_target[index_kp_target_all, 1], kp_target[index_kp_target_all, 0]].reshape(n_valid, 1, 2)
    # estimated point corresponding to kp_target

    kp_source_reshape = np.copy(kp_source).reshape(1, n_s, 2)

    # compute the difference between estimated keypoints and provided keypoints in source
    norm = np.linalg.norm(p_source_est - kp_source_reshape, ord=2, axis=2)  # n_valid x N_s
    # compute for each point estimated kp_source the minimum with all provided keypoint source
    min1 = np.amin(norm, axis=1)
    index_kp_source_all = np.argmin(norm, axis=1)  # the index of the kp_source closest to the p_source_est

    list_of_good_kp = min1 < min_to_kp
    index_match_kp_target = index_kp_target_all[list_of_good_kp].tolist()
    index_match_kp_source = index_kp_source_all[list_of_good_kp].tolist()

    matches = np.concatenate((np.array(index_match_kp_source).reshape(-1, 1),
                              np.array(index_match_kp_target).reshape(-1, 1)), axis=1)
    if confidence_map is not None:
        scores = confidence_map[kp_target[index_match_kp_target, 1], kp_target[index_match_kp_target, 0]].reshape(-1, 1)
        matches = np.concatenate((matches, scores), axis=1)
    return matches


def get_mutual_matches(matches_0_1, matches_1_0,  kp_0, kp_1, acceptable_error):
    """Keep only matches which are mutually consistent.

    Args:
        matches_1_0: Nx2 or Nx3, contains index of kp_1 followed by kp_0. Can also contain confidence value.
        matches_0_1: Mx2 or Nx3, contains index of kp_0 followed by kp_1. Can also contain confidence value.
        kp_0: K0 x 2
        kp_1: K1 x 2
        acceptable_error: threshold in pixel for cyclic error to consider a match mutual.

    Returns:
        matches: Lx2 or Lx3, contains index of kp_0 followed by kp_1. Can also contain confidence value.
    """
    if len(matches_1_0) > 0 and len(matches_0_1) > 0:
        # probably need to accept that error is below 1 pixel, otherwise equal will be hard
        kp_0_matches_0_1 = kp_0[matches_0_1[:, 0].astype(np.int32)]
        kp_0_matches_1_0 = kp_0[matches_1_0[:, 1].astype(np.int32)]
        kp_1_matches_0_1 = kp_1[matches_0_1[:, 1].astype(np.int32)]
        kp_1_matches_1_0 = kp_1[matches_1_0[:, 0].astype(np.int32)]

        error_kp_0 = np.linalg.norm(np.expand_dims(kp_0_matches_0_1, 1) -
                                    np.expand_dims(kp_0_matches_1_0, 0), ord=None, axis=2)  # N1xN2
        # N1x1x2 - 1xN2x2, shapes are broadcasted to N1 x N2 x 2
        mutual_kp_0 = np.where(error_kp_0 <= acceptable_error)  # same kp0 is used in both images
        index_kp_matches_0_1 = mutual_kp_0[0]
        index_kp_matches_1_0 = mutual_kp_0[1]
        mutual_matches = np.linalg.norm(kp_1_matches_0_1[index_kp_matches_0_1] -
                                        kp_1_matches_1_0[index_kp_matches_1_0],
                                        axis=1) <= acceptable_error
        '''
        assert (np.linalg.norm(kp_0_matches_0_1[index_kp_matches_0_1[mutual_matches][0]] -
                               kp_0_matches_1_0[index_kp_matches_1_0[mutual_matches][0]], axis=0)
                <= acceptable_error) and \
               (np.linalg.norm(kp_1_matches_0_1[index_kp_matches_0_1[mutual_matches][0]] -
                               kp_1_matches_1_0[index_kp_matches_1_0[mutual_matches][0]], axis=0)
                <= acceptable_error)
        '''
        if len(matches_0_1) > len(matches_1_0):
            # we keep the kp of matches 0 1
            mutual_index_kp_matches_0_1 = index_kp_matches_0_1[mutual_matches]
            matches = matches_0_1[mutual_index_kp_matches_0_1]
        else:
            mutual_index_kp_matches_1_0 = index_kp_matches_1_0[mutual_matches]
            if matches_1_0.shape[1] == 3:
                # also contains the score of the match
                matches = matches_1_0[:, [1, 0, 2]][mutual_index_kp_matches_1_0]
            else:
                matches = matches_1_0[:, [1, 0]][mutual_index_kp_matches_1_0]
        print('Mutual matches {}/{}'.format(mutual_matches.sum(),
                                            max(len(matches_0_1), len(matches_1_0))))
    elif len(matches_0_1) > 0:
        matches = matches_0_1
    else:
        if matches_1_0.shape[1] == 3:
            # also contains the score of the match
            matches = matches_1_0[:, [1, 0, 2]]
        else:
            matches = matches_1_0[:, [1, 0]]
    return matches
