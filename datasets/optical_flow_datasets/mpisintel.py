import os.path
import glob
import numpy as np
import cv2
import torch.utils.data as data

from datasets.listdataset import ListDataset
from utils_flow.img_processing_utils import split2list
from datasets.listdataset import load_flo
'''
Dataset routines for MPI Sintel.
http://sintel.is.tue.mpg.de/
clean version imgs are without shaders, final version imgs are fully rendered
The datasets is not very big, you might want to only pretrain on it for flownet
'''


def make_dataset(dataset_dir, split, dataset_type='clean'):
    flow_dir = 'flow'
    assert(os.path.isdir(os.path.join(dataset_dir,flow_dir)))
    img_dir = dataset_type
    assert(os.path.isdir(os.path.join(dataset_dir,img_dir)))

    images = []
    for flow_map in sorted(glob.glob(os.path.join(dataset_dir,flow_dir,'*','*.flo'))):
        flow_map = os.path.relpath(flow_map,os.path.join(dataset_dir,flow_dir))

        scene_dir, filename = os.path.split(flow_map)
        no_ext_filename = os.path.splitext(filename)[0]
        prefix, frame_nb = no_ext_filename.split('_')
        frame_nb = int(frame_nb)
        img1 = os.path.join(img_dir, scene_dir, '{}_{:04d}.png'.format(prefix, frame_nb + 1))
        img2 = os.path.join(img_dir, scene_dir, '{}_{:04d}.png'.format(prefix, frame_nb))
        # img2 is target, which corresponds to the first image for sintel
        flow_map = os.path.join(flow_dir, flow_map)
        if not (os.path.isfile(os.path.join(dataset_dir,img1)) and os.path.isfile(os.path.join(dataset_dir,img2))):
            continue
        images.append([[img1,img2],flow_map])
    return split2list(images, split, default_split=0.87)


def mpisintel_loader(root, path_imgs, path_flo, return_occlusion_mask=False):
    imgs = [os.path.join(root,path) for path in path_imgs]
    flo = os.path.join(root,path_flo)

    invalid_mask_dir = 'invalid'
    occlusion_mask_dir = 'occlusions'

    scene_dir, filename = os.path.split(path_flo)
    flow, scene_dir = os.path.split(scene_dir)
    filename = filename[:-4]

    path_invalid_mask = os.path.join(invalid_mask_dir, scene_dir, '{}.png'.format(filename))
    invalid_mask = cv2.imread(os.path.join(root, path_invalid_mask), 0).astype(np.uint8)
    valid_mask = (invalid_mask == 0)

    # if want to remove occluded regions
    path_occlusion_mask = os.path.join(occlusion_mask_dir, scene_dir, '{}.png'.format(filename))
    occluded_mask = cv2.imread(os.path.join(root, path_occlusion_mask), 0).astype(np.uint8)
    noc_mask = (occluded_mask == 0).astype(np.uint8)

    if return_occlusion_mask:
        return [cv2.imread(img)[:, :, ::-1].astype(np.uint8) for img in imgs], load_flo(flo), valid_mask.astype(np.uint8), \
               occluded_mask.astype(np.uint8)
    else:
        return [cv2.imread(img)[:, :, ::-1].astype(np.uint8) for img in imgs], load_flo(flo), valid_mask.astype(np.uint8)


def mpi_sintel_clean(root, source_image_transform=None, target_image_transform=None, flow_transform=None,
                     co_transform=None, split=0.0, load_occlusion_mask=False):
    """
    Builds the dataset of MPI Sintel image pairs and corresponding ground-truth flow fields, for the 'clean' split
    Args:
        root: path to root folder
        source_image_transform: image transformations to apply to source images
        target_image_transform: image transformations to apply to target images
        flow_transform: flow transformations to apply to ground-truth flow fields
        co_transform: transformations to apply to both images and ground-truth flow fields
        split: split (float) between training and testing, 0 means all pairs are in test_dataset
        load_occlusion_mask: is the loader also outputting a ground-truth occlusion mask ? The ground-truth occlusion
                             mask will then be include in the returned dictionary fields.

    Returns:
        train_dataset
        test_dataset

    """
    train_list, test_list = make_dataset(root, split, 'clean')
    train_dataset = ListDataset(root, train_list, source_image_transform=source_image_transform,
                                target_image_transform=target_image_transform,
                                flow_transform=flow_transform,
                                co_transform=co_transform, loader=mpisintel_loader, load_valid_mask=True,
                                load_occlusion_mask=load_occlusion_mask)
    test_dataset = ListDataset(root, test_list, source_image_transform=source_image_transform,
                               target_image_transform=target_image_transform,
                               flow_transform=flow_transform,
                               co_transform=co_transform, loader=mpisintel_loader, load_valid_mask=True,
                               load_occlusion_mask=load_occlusion_mask)
    # co_flow_and_images_transforms.CenterCrop((384, 1024))

    return train_dataset, test_dataset


def mpi_sintel_final(root, source_image_transform=None, target_image_transform=None, flow_transform=None,
                     co_transform=None, split=0.0, load_occlusion_mask=False):
    """
    Builds the dataset of MPI Sintel image pairs and corresponding ground-truth flow fields, for the 'clean' split
    Args:
        root: path to root folder
        source_image_transform: image transformations to apply to source images
        target_image_transform: image transformations to apply to target images
        flow_transform: flow transformations to apply to ground-truth flow fields
        co_transform: transformations to apply to both images and ground-truth flow fields
        split: split (float) between training and testing, 0 means all pairs are in test_dataset
        load_occlusion_mask: is the loader also outputting a ground-truth occlusion mask ? The ground-truth occlusion
                             mask will then be include in the returned dictionary fields.

    Returns:
        train_dataset
        test_dataset

    """
    train_list, test_list = make_dataset(root, split, 'final')
    train_dataset = ListDataset(root, train_list, source_image_transform=source_image_transform,
                                target_image_transform=target_image_transform,
                                flow_transform=flow_transform,
                                co_transform=co_transform, loader=mpisintel_loader, load_valid_mask=True,
                                load_occlusion_mask=load_occlusion_mask)
    test_dataset = ListDataset(root, test_list, source_image_transform=source_image_transform,
                               target_image_transform=target_image_transform,
                               flow_transform=flow_transform,
                               co_transform=co_transform, loader=mpisintel_loader, load_valid_mask=True,
                               load_occlusion_mask=load_occlusion_mask)

    return train_dataset, test_dataset


def mpi_sintel(root, source_image_transform=None, target_image_transform=None, flow_transform=None,
               co_transform=None, split=0.0, load_occlusion_mask=False, dstype='clean'):
    """
    Builds the dataset of MPI Sintel image pairs and corresponding ground-truth flow fields, for a given
    dstype ('clean', 'final')
    Args:
        root: path to root folder
        source_image_transform: image transformations to apply to source images
        target_image_transform: image transformations to apply to target images
        flow_transform: flow transformations to apply to ground-truth flow fields
        co_transform: transformations to apply to both images and ground-truth flow fields
        split: split (float) between training and testing, 0 means all pairs are in test_dataset
        load_occlusion_mask: is the loader also outputting a ground-truth occlusion mask ? The ground-truth occlusion
                             mask will then be include in the returned dictionary fields.
        dstype: 'clean' or 'final'

    Returns:
        train_dataset
        test_dataset

    """
    if dstype == 'clean':
        return mpi_sintel_clean(root, source_image_transform, target_image_transform, flow_transform,
                                co_transform, split, load_occlusion_mask)

    elif dstype == 'final':
        return mpi_sintel_final(root, source_image_transform, target_image_transform, flow_transform,
                                co_transform, split, load_occlusion_mask)

    else:
        raise ValueError('The split of sintel that you chose does not exist {}'.format(dstype))


def make_dataset_test_data(root):
    images = []
    list_files = sorted(glob.glob(os.path.join(root, '*.png')))
    total_nbr= len(list_files)
    for nbr in range(0, total_nbr-1):
        frame_nb = nbr + 1
        prefix = 'frame'
        img1 = os.path.join(root, '{}_{:04d}.png'.format(prefix, frame_nb + 1))
        img2 = os.path.join(root, '{}_{:04d}.png'.format(prefix, frame_nb))
        # img2 is target, which corresponds to the first image for sintel
        name_of_flow =  '{}{:04d}.flo'.format(prefix, frame_nb)
        images.append([[img1, img2], name_of_flow])
    return images


class MPISintelTestData(data.Dataset):
    def __init__(self, root, first_image_transform=None, second_image_transform=None):

        self.root = root

        self.path_list = make_dataset_test_data(root)
        self.first_image_transform = first_image_transform
        self.second_image_transform = second_image_transform

    def __getitem__(self, index):

        inputs = self.path_list[index]
        im1 = cv2.imread(inputs[0][0])[:, :, ::-1].astype(np.uint8)
        im2 = cv2.imread(inputs[0][1])[:, :, ::-1].astype(np.uint8)
        shape = im1.shape
        if self.first_image_transform is not None:
            im1 = self.first_image_transform(im1)
        if self.second_image_transform is not None:
            im2 = self.second_image_transform(im2)

        return {'source_image': im1,
                'target_image': im2,
                'image_shape': shape,
                'name_flow_file': inputs[1]}
        # attention here this is flow and not correspondence !! and it is nor normalised

    def __len__(self):
        return len(self.path_list)