from __future__ import division
import os.path
import torch.utils.data as data
import numpy as np
import cv2
import pickle
import torch
from packaging import version


class ETHInterval(data.Dataset):
    """ETH interval Dataset"""
    def __init__(self, root, path_list, source_image_transform=None, target_image_transform=None, flow_transform=None,
                 co_transform=None, *args, **kwargs):
        """
        Args:
            root: directory containing the dataset images
            path_list: list containing the name of images and corresponcence list
            source_image_transform: transforms to apply to source images
            target_image_transform: transforms to apply to target images
            flow_transform: transforms to apply to flow field
            co_transform: transforms to apply to both images and the flow field
        Output in __getitem__:
            source_image: source image
            target_image: target image
            flow_map: pixel correspondence map relating the target view to the source
            correspondence_mask: valid/invalid correspondences (sparse data here)
            source_image_size
        """
        self.root = root
        self.path_list = path_list
        with open(self.path_list, 'rb') as f:
            self.df = pickle.load(f)
        self.first_image_transform = source_image_transform
        self.second_image_transform = target_image_transform
        self.target_transform = flow_transform
        self.co_transform = co_transform

    def __getitem__(self, index):
        """
        Args:
            index

        Returns: Dictionary with fieldnames:
            source_image
            target_image
            flow_map
            correspondence_mask
            source_image_size
        """
        # for all inputs[0] must be the source and inputs[1] must be the target
        dict_index = self.df[index]
        source_image_path = dict_index['source_image']
        target_image_path = dict_index['target_image']
        Xs = np.float32(dict_index['Xs'])
        Ys = np.float32(dict_index['Ys'])
        Xt = np.float32(dict_index['Xt'])
        Yt = np.float32(dict_index['Yt'])

        source_image_path = os.path.join(self.root, source_image_path)
        if not os.path.isfile(source_image_path):
            print('Problem reading image at {}. Check your paths !'.format(source_image_path))
            exit(1)
        source_image = cv2.imread(source_image_path)[:, :, ::-1].astype(np.uint8)
        source_size = source_image.shape

        target_image_path = os.path.join(self.root, target_image_path)
        if not os.path.isfile(target_image_path):
            print('Problem reading image at {}. Check your paths !'.format(target_image_path))
            exit(1)
        target_image = cv2.imread(target_image_path)[:, :, ::-1].astype(np.uint8)
        inputs = [source_image, target_image]
        h, w = target_image.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        flowx = np.zeros((h, w), dtype=np.float32)
        flowy = np.zeros((h, w), dtype=np.float32)
        mask[np.int32(np.round(Yt)), np.int32(np.round(Xt))] = 1
        flowx[np.int32(np.round(Yt)), np.int32(np.round(Xt))] = Xs - Xt
        flowy[np.int32(np.round(Yt)), np.int32(np.round(Xt))] = Ys - Yt

        target = np.dstack((flowx, flowy))

        # mask is shape hxw
        if self.co_transform is not None:
            inputs, target, mask = self.co_transform(inputs, target, mask)

        if self.first_image_transform is not None:
            inputs[0] = self.first_image_transform(inputs[0])
        if self.second_image_transform is not None:
            inputs[1] = self.second_image_transform(inputs[1])
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'source_image': inputs[0],
                'target_image': inputs[1],
                'flow_map': target,
                'correspondence_mask': mask.astype(np.bool) if version.parse(torch.__version__) >= version.parse("1.1")
                  else mask.astype(np.uint8),
                'source_image_size': source_size,
                }

    def __len__(self):
        return len(self.df)

