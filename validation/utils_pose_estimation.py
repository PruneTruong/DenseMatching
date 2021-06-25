import numpy as np
import cv2
import torch
from imageio import imread


def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
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


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def get_new_resolution_with_minimum(minSize, I, strideNet):
    h, w = I.shape[:2]
    ratio = min(w / float(minSize), h / float(minSize))
    new_w, new_h = round(w / ratio), round(h / ratio)
    new_w, new_h = new_w // strideNet * strideNet, new_h // strideNet * strideNet
    return new_w, new_h


def read_image(path, device, rotation, resize_float, resize=None, min_size=None, strideNet=8):
    image = imread(str(path))
    if image is None:
        return None, None, None

    w, h = image.shape[1], image.shape[0]
    if min_size is not None:
        # it means we need to resize the image keeping aspect ratio so that smallest side is equal to min_size
        w_new, h_new = get_new_resolution_with_minimum(min_size, image, strideNet)
    else:
        w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new))#.astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation).copy()
        # needs the copy for later to be read in torch !
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales


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


def matches_from_flow(flow, matchBinary, scaling=1.0):
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
        matchBinary = matchBinary.cuda()

    mapping = flow + grid
    mapping_x = mapping.permute(0, 2, 3, 1)[:, :, :, 0]
    mapping_y = mapping.permute(0, 2, 3, 1)[:, :, :, 1]
    grid_x = grid.permute(0, 2, 3, 1)[:, :, :, 0]
    grid_y = grid.permute(0, 2, 3, 1)[:, :, :, 1]

    pts2 = torch.cat((grid_x[matchBinary].unsqueeze(1),
                      grid_y[matchBinary].unsqueeze(1)), dim=1)
    pts1 = torch.cat((mapping_x[matchBinary].unsqueeze(1),
                      mapping_y[matchBinary].unsqueeze(1)),
                     dim=1)  # convert to mapping and then take the correspondences

    return pts1.cpu().numpy()*scaling, pts2.cpu().numpy()*scaling
