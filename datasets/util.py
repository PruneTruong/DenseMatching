import numpy as np
import cv2
from utils_flow.util import pad_to_same_shape
import torch


def define_mask_zero_borders(image, epsilon=1e-8):
    """Computes the binary mask, equal to 0 when image is 0 and 1 otherwise."""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 4:
            if image.shape[1] == 3:
                # image b, 3, H, W
                image = image.transpose(0, 2, 3, 1)
            # image is b, H, W, 3
            occ_mask = np.logical_and(np.logical_and(image[:, :, :, 0] < epsilon,
                                                     image[:, :, :, 1] < epsilon),
                                      image[:, :, :, 2] < epsilon)
        else:
            if image.shape[0] == 3:
                # image 3, H, W
                image = image.transpose(1, 2, 0)
            # image is H, W, 3
            occ_mask = np.logical_and(np.logical_and(image[:, :, 0] < epsilon,
                                                     image[:, :, 1] < epsilon),
                                      image[:, :, 2] < epsilon)
        mask = ~occ_mask
        mask = mask.astype(np.bool) if float(torch.__version__[:3]) >= 1.1 else mask.astype(np.uint8)
    else:
        # torch tensor
        if len(image.shape) == 4:
            if image.shape[1] == 3:
                # image b, 3, H, W
                image = image.permute(0, 2, 3, 1)
            occ_mask = image[:, :, :, 0].le(epsilon) & image[:, :, :, 1].le(epsilon) & image[:, :, :, 2].le(epsilon)
        else:
            if image.shape[0] == 3:
                # image 3, H, W
                image = image.permute(1, 2, 0)
            occ_mask = image[:, :, 0].le(epsilon) & image[:, :, 1].le(epsilon) & image[:, :, 2].le(epsilon)
        mask = ~occ_mask
        mask = mask.bool() if float(torch.__version__[:3]) >= 1.1 else mask.byte()
    return mask


def horizontal_combine_images(img1, img2):
    ratio = img1.shape[0] / img2.shape[0]
    imgs_comb = np.hstack((img1, cv2.resize(img2, None, fx=ratio, fy=ratio)))
    return imgs_comb


def draw_matches(img1, img2, kp1, kp2):
    """

    Args:
        img1:
        img2:
        kp1: kp1 is shape Nx2, N number of feature points, first point in horizontal direction
        kp2: kp2 is shape Nx2, N number of feature points, first point in horizontal direction

    Returns:

    """
    img1, img2 = pad_to_same_shape(img1, img2)
    h, w = img1.shape[:2]
    img = horizontal_combine_images(img1, img2)

    if kp1.shape[0] == 0:
        return img
    # shape Mx1x2 M number of matches
    kp2[:, 0] = kp2[:, 0] + w

    for i in range(kp1.shape[0]):
        img = cv2.line(img, (kp1[i, 0], kp1[i, 1]), (kp2[i, 0], kp2[i, 1]), (255, 0, 0), 2)
    return img


def draw_keypoints(img, kp):
    """

    Args:
        img:
        kp: kp1 is shape Nx2, N number of feature points, first point in horizontal direction

    Returns:

    """
    image_copy = np.copy(img)
    nbr_points = kp.shape[0]
    for i in range(nbr_points):
        image = cv2.circle(image_copy, (np.uint(kp[i,0]),np.uint(kp[i,1])), 1, (0,255,0),thickness=5)
    return image


def split2list(images, split, default_split=0.9):
    if isinstance(split, str):
        with open(split) as f:
            split_values = [x.strip() == '1' for x in f.readlines()]
        assert(len(images) == len(split_values))
    elif split is None:
        split_values = np.random.uniform(0, 1, len(images)) < default_split
    else:
        try:
            split = float(split)
        except TypeError:
            print("Invalid Split value, it must be either a filepath or a float")
            raise
        split_values = np.random.uniform(0,1,len(images)) < split
    train_samples = [sample for sample, split in zip(images, split_values) if split]
    test_samples = [sample for sample, split in zip(images, split_values) if not split]
    return train_samples, test_samples