from __future__ import division
import os.path
import glob
import numpy as np
import cv2

from datasets.listdataset import ListDataset
from utils_flow.img_processing_utils import split2list


'''
Dataset routines for KITTI_flow, 2012 and 2015.
http://www.cvlibs.net/datasets/kitti/eval_flow.php
The datasets is not very big, you might want to only finetune on it for flownet
EPE are not representative in this datasets because of the sparsity of the GT.
OpenCV is needed to load 16bit png images
'''


def load_flow_from_png(png_path):
    # The -1 is here to specify not to change the image depth (16bit), and is compatible
    # with both OpenCV2 and OpenCV3
    if type(png_path) in [tuple, list]:
        flo_file_all = png_path[0]
        flo_file_noc = png_path[1]
        flo_file_all = cv2.imread(flo_file_all, -1)
        flo_img = flo_file_all[:, :, 2:0:-1].astype(np.float32)
        invalid = (flo_file_all[:, :, 0] == 0)  # in cv2 change the channel to the first, mask of invalid pixels
        valid_all = (flo_file_all[:, :, 0] == 1)

        flo_file_noc = cv2.imread(flo_file_noc, -1)
        valid_noc = (flo_file_noc[:, :, 0] == 1)

        valid = np.logical_and(valid_all == 1, valid_noc == 0)

        flo_img = flo_img - 32768
        flo_img = flo_img / 64
        flo_img[np.abs(flo_img) < 1e-10] = 1e-10
        flo_img[invalid, :] = 0  # float('nan') # invalid pixels, will not be represented when plotting images

    else:
        flo_file = cv2.imread(png_path, -1)
        flo_img = flo_file[:, :, 2:0:-1].astype(np.float32)
        invalid = (flo_file[:, :, 0] == 0)  # in cv2 change the channel to the first, mask of invalid pixels
        valid = (flo_file[:, :, 0] == 1)
        flo_img = flo_img - 32768
        flo_img = flo_img / 64
        flo_img[np.abs(flo_img) < 1e-10] = 1e-10
        flo_img[invalid, :] = 0
    # make a mask out of it !
    return flo_img, valid.astype(np.uint8)


def make_dataset(dir, split=0.9, occ=True, only_occ=False, dataset_name=None):
    '''
    ATTENTION HERE I MODIFIED WHICH IMAGE IS THE TARGET OR NOT
    Will search in training folder for folders 'flow_noc' or 'flow_occ'
       and 'colored_0' (KITTI 2012) or 'image_2' (KITTI 2015) '''

    if only_occ:
        flow_dir = 'flow_occ'
        flow_dir_noc = 'flow_noc'
    else:
        flow_dir = 'flow_occ' if occ else 'flow_noc'
        assert(os.path.isdir(os.path.join(dir, flow_dir)))
    img_dir = 'colored_0'
    if not os.path.isdir(os.path.join(dir, img_dir)):
        img_dir = 'image_2'
    assert(os.path.isdir(os.path.join(dir, img_dir)))

    images = []
    for flow_map in glob.iglob(os.path.join(dir, flow_dir, '*.png')):
        flow_map = os.path.basename(flow_map) # list of names of flows
        if only_occ:
            flow_map_noc = os.path.join(flow_dir_noc, flow_map)

        root_filename = flow_map[:-7] # name of image
        flow_map = os.path.join(flow_dir, flow_map) # path to flow_map
        img1 = os.path.join(img_dir, root_filename+'_11.png') # path to image_1
        img2 = os.path.join(img_dir, root_filename+'_10.png') # path to image_3, they say image 10 is the reference
        # ( it is target, the flow is defined for each point of target to source, mask is each point of target)
        if not (os.path.isfile(os.path.join(dir, img1)) or os.path.isfile(os.path.join(dir, img2))):
            continue

        if only_occ:
            if dataset_name is not None:
                images.append([[os.path.join(dataset_name, img1),
                                os.path.join(dataset_name, img2)],
                               os.path.join(dataset_name, flow_map)])
            else:
                images.append([[img1, img2], [flow_map, flow_map_noc]])
        else:
            if dataset_name is not None:
                images.append([[os.path.join(dataset_name, img1),
                                os.path.join(dataset_name, img2)],
                               os.path.join(dataset_name, flow_map)])
            else:
                images.append([[img1, img2], flow_map])

    return split2list(images, split, default_split=0.9)


def kitti_flow_loader(root, path_imgs, path_flo):
    imgs = [os.path.join(root,path) for path in path_imgs]
    if type(path_flo) in [tuple, list]:
        flo = [os.path.join(root, path_flo[0]), os.path.join(root, path_flo[1])]
    else:
        flo = os.path.join(root,path_flo)
    flow, mask = load_flow_from_png(flo)
    return [cv2.imread(img)[:, :, ::-1].astype(np.uint8) for img in imgs], flow, mask


def KITTI_occ(root, source_image_transform=None, target_image_transform=None, flow_transform=None,
              co_transform=None, split=0.0):
    """
    Builds the dataset of KITTI image pairs and corresponding ground-truth flow fields. The ground-truth mask
    includes all valid pixels (occluded and non-occluded).
    Args:
        root: path to root folder
        source_image_transform: image transformations to apply to source images
        target_image_transform: image transformations to apply to target images
        flow_transform: flow transformations to apply to ground-truth flow fields
        co_transform: transformations to apply to both images and ground-truth flow fields
        split: split (float) between training and testing, 0 means all pairs are in test_dataset

    Returns:
        train_dataset
        test_dataset

    """
    # root will be /Desktop/datasets/data_scene_flow/training/
    train_list, test_list = make_dataset(root, split, True)
    train_dataset = ListDataset(root, train_list, source_image_transform=source_image_transform,
                                target_image_transform=target_image_transform,
                                flow_transform=flow_transform, co_transform=co_transform,
                                loader=kitti_flow_loader, load_valid_mask=True)
    # All test sample are cropped to lowest possible size of KITTI images
    test_dataset = ListDataset(root, test_list, source_image_transform=source_image_transform,
                               target_image_transform=target_image_transform,
                               flow_transform=flow_transform,
                               co_transform=co_transform,
                               loader=kitti_flow_loader, load_valid_mask=True)
    # co_flow_and_images_transforms.CenterCrop((370, 1224))

    return train_dataset, test_dataset


def KITTI_noc(root, source_image_transform=None, target_image_transform=None, flow_transform=None,
              co_transform=None, split=0.0):
    """
    Builds the dataset of KITTI image pairs and corresponding ground-truth flow fields. The ground-truth mask
    only includes non occluded (visible) pixels.
    Args:
        root: path to root folder
        source_image_transform: image transformations to apply to source images
        target_image_transform: image transformations to apply to target images
        flow_transform: flow transformations to apply to ground-truth flow fields
        co_transform: transformations to apply to both images and ground-truth flow fields
        split: split (float) between training and testing, 0 means all pairs are in test_dataset

    Returns:
        train_dataset
        test_dataset

    """
    train_list, test_list = make_dataset(root, split, occ=False)
    train_dataset = ListDataset(root, train_list, source_image_transform=source_image_transform,
                                target_image_transform=target_image_transform,
                                flow_transform=flow_transform, co_transform=co_transform, loader=kitti_flow_loader,
                                load_valid_mask=True)
    # All test sample are cropped to lowest possible size of KITTI images
    test_dataset = ListDataset(root, test_list, source_image_transform=source_image_transform,
                               target_image_transform=target_image_transform,
                               flow_transform=flow_transform,
                               co_transform=co_transform,
                               loader=kitti_flow_loader, load_valid_mask=True)

    return train_dataset, test_dataset


def KITTI_only_occ(root, source_image_transform=None, target_image_transform=None, flow_transform=None,
                   co_transform=None, split=None):
    """
    Builds the dataset of KITTI image pairs and corresponding ground-truth flow fields. The ground-truth mask
    only includes occluded pixels.
    Args:
        root: path to root folder
        source_image_transform: image transformations to apply to source images
        target_image_transform: image transformations to apply to target images
        flow_transform: flow transformations to apply to ground-truth flow fields
        co_transform: transformations to apply to both images and ground-truth flow fields
        split: split (float) between training and testing, 0 means all pairs are in test_dataset

    Returns:
        train_dataset
        test_dataset

    """
    train_list, test_list = make_dataset(root, split, occ=False, only_occ=True)
    train_dataset = ListDataset(root, train_list, source_image_transform=source_image_transform,
                                target_image_transform=target_image_transform,
                                flow_transform=flow_transform, co_transform=co_transform, loader=kitti_flow_loader,
                                load_valid_mask=True)
    # All test sample are cropped to lowest possible size of KITTI images
    test_dataset = ListDataset(root, test_list, source_image_transform=source_image_transform,
                               target_image_transform=target_image_transform,
                               flow_transform=flow_transform,
                               co_transform=co_transform,
                               loader=kitti_flow_loader, load_valid_mask=True)

    return train_dataset, test_dataset
