from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from datasets.util import pad_to_same_shape
from .semantic_keypoints_datasets import SemanticKeypointsDataset, random_crop
from datasets.util import define_mask_zero_borders
import scipy.io as sio
import random


def read_mat(path, obj_name):
    r"""Reads specified objects from Matlab data file, (.mat)"""
    mat_contents = sio.loadmat(path)
    mat_obj = mat_contents[obj_name]

    return mat_obj


class PFPascalDataset(SemanticKeypointsDataset):
    """
    Proposal Flow image pair dataset (PF-Pascal).
    There is a certain number of pairs per category and the number of keypoints per pair also varies
    """
    def __init__(self, root, split, thres='img', source_image_transform=None,
                 target_image_transform=None, flow_transform=None, output_image_size=None, training_cfg=None):
        """
        Args:
            root:
            split: 'test', 'val', 'train'
            source_image_transform: image transformations to apply to source images
            target_image_transform: image transformations to apply to target images
            flow_transform: flow transformations to apply to ground-truth flow fields
        Output in __getittem__:
            source_image
            target_image
            source_image_size
            target_image_size
            flow_map
            correspondence_mask: valid correspondences (which are originally sparse).

        """
        super(PFPascalDataset, self).__init__('pfpascal', root, thres, split, source_image_transform,
                                              target_image_transform, flow_transform, training_cfg=training_cfg)

        self.train_data = pd.read_csv(self.spt_path)
        self.src_imnames = np.array(self.train_data.iloc[:, 0])
        self.trg_imnames = np.array(self.train_data.iloc[:, 1])
        self.cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.cls_ids = self.train_data.iloc[:, 2].values.astype('int') - 1

        if split == 'train':
            self.flip = self.train_data.iloc[:, 3].values.astype('int')
        self.src_kps = []
        self.trg_kps = []
        self.src_bbox = []
        self.trg_bbox = []
        # here reads bounding box and keypoints information from annotation files. Also in most of the csv files.
        for src_imname, trg_imname, cls in zip(self.src_imnames, self.trg_imnames, self.cls_ids):
            src_anns = os.path.join(self.ann_path, self.cls[cls],
                                    os.path.basename(src_imname))[:-4] + '.mat'
            trg_anns = os.path.join(self.ann_path, self.cls[cls],
                                    os.path.basename(trg_imname))[:-4] + '.mat'

            src_kp = torch.tensor(read_mat(src_anns, 'kps')).float()
            trg_kp = torch.tensor(read_mat(trg_anns, 'kps')).float()
            src_box = torch.tensor(read_mat(src_anns, 'bbox')[0].astype(float))
            trg_box = torch.tensor(read_mat(trg_anns, 'bbox')[0].astype(float))

            src_kps = []
            trg_kps = []
            for src_kk, trg_kk in zip(src_kp, trg_kp):
                if len(torch.isnan(src_kk).nonzero()) != 0 or \
                        len(torch.isnan(trg_kk).nonzero()) != 0:
                    continue
                else:
                    src_kps.append(src_kk)
                    trg_kps.append(trg_kk)
            self.src_kps.append(torch.stack(src_kps).t())
            self.trg_kps.append(torch.stack(trg_kps).t())
            self.src_bbox.append(src_box)
            self.trg_bbox.append(trg_box)

        self.src_imnames = list(map(lambda x: os.path.basename(x), self.src_imnames))
        self.trg_imnames = list(map(lambda x: os.path.basename(x), self.trg_imnames))

        # if need to resize the images, even for testing
        if output_image_size is not None:
            if not isinstance(output_image_size, tuple):
                output_image_size = (output_image_size, output_image_size)
        self.output_image_size = output_image_size

    def __getitem__(self, idx):
        """
        Args:
            idx:

        Returns: Dictionary with fieldnames:
            source_image
            target_image
            source_image_size
            target_image_size
            flow_map
            correspondence_mask: valid correspondences (which are originally sparse).
        """
        batch = super(PFPascalDataset, self).__getitem__(idx)

        batch['src_bbox'] = self.get_bbox(self.src_bbox, idx)
        batch['trg_bbox'] = self.get_bbox(self.trg_bbox, idx)

        if self.split != 'test':
            # for training, might want to have different output flow sizes
            if self.training_cfg['augment_with_crop']:
                batch['src_img'], batch['src_kps'] = random_crop(batch['src_img'], batch['src_kps'].clone(),
                                                                 batch['src_bbox'].int(),
                                                                 size=self.training_cfg['crop_size'])
                batch['trg_img'], batch['trg_kps'] = random_crop(batch['trg_img'], batch['trg_kps'].clone(),
                                                                 batch['trg_bbox'].int(),
                                                                 size=self.training_cfg['crop_size'])

            if self.training_cfg['augment_with_flip']:
                if random.random() < self.training_cfg['proba_of_batch_flip']:
                    batch['src_img'], batch['src_bbox'], batch['src_kps'] = self.horizontal_flip_img(
                        batch['src_img'], batch['src_bbox'], batch['src_kps'])
                    batch['trg_img'], batch['trg_bbox'], batch['trg_kps'] = self.horizontal_flip_img(
                        batch['trg_img'], batch['trg_bbox'], batch['trg_kps'])
                else:
                    if random.random() < self.training_cfg['proba_of_image_flip']:
                        batch['src_img'], batch['src_bbox'], batch['src_kps'] = self.horizontal_flip_img(
                            batch['src_img'], batch['src_bbox'], batch['src_kps'])
                    if random.random() < self.training_cfg['proba_of_image_flip']:
                        batch['trg_img'], batch['trg_bbox'], batch['trg_kps'] = self.horizontal_flip_img(
                            batch['trg_img'], batch['trg_bbox'], batch['trg_kps'])
            '''
            # Horizontal flipping of both images and key-points during training
            if self.flip[idx]:
                self.horizontal_flip(batch)
                batch['flip'] = 1
            else:
                batch['flip'] = 0
            '''

            source, target, flow, mask = self.recover_image_pair_for_training(batch['src_img'], batch['trg_img'],
                                                                              kp_source=torch.t(batch['src_kps']).clone(),
                                                                              kp_target=torch.t(batch['trg_kps']).clone())

            if self.source_image_transform is not None:
                source = self.source_image_transform(source)
            if self.target_image_transform is not None:
                target = self.target_image_transform(target)
            if self.flow_transform is not None:
                if type(flow) in [tuple, list]:
                    # flow field at different resolution
                    for i in range(len(flow)):
                        flow[i] = self.flow_transform(flow[i])
                else:
                    flow = self.flow_transform(flow)

            output = {'source_image': source, 'target_image': target, 'flow_map': flow, 'correspondence_mask': mask,
                      'sparse': True, 'source_image_size': batch['src_imsize']}
            if self.training_cfg['compute_mask_zero_borders']:
                mask_valid = define_mask_zero_borders(target)
                output['mask_zero_borders'] = mask_valid
            return output
        else:
            batch['pckthres'] = self.get_pckthres(batch, batch['src_imsize'])
            batch['src_bbox'] = self.get_bbox(self.src_bbox, idx)
            batch['trg_bbox'] = self.get_bbox(self.trg_bbox, idx)

            batch['src_img'], batch['trg_img'] = pad_to_same_shape(batch['src_img'], batch['trg_img'])
            h_size, w_size, _ = batch['trg_img'].shape

            flow, mask = self.keypoints_to_flow(torch.t(batch['src_kps']),
                                                torch.t(batch['trg_kps']), h_size=h_size, w_size=w_size)
            if self.source_image_transform is not None:
                batch['src_img'] = self.source_image_transform(batch['src_img'])
            if self.target_image_transform is not None:
                batch['trg_img'] = self.target_image_transform(batch['trg_img'])
            if self.flow_transform is not None:
                flow = self.flow_transform(flow)

            return {'source_image': batch['src_img'],
                    'target_image': batch['trg_img'],
                    'source_image_size': batch['src_imsize'],
                    'target_image_size': batch['trg_imsize'],
                    'flow_map': flow,
                    'correspondence_mask': mask.bool() if float(torch.__version__[:3]) >= 1.1 \
                        else mask.byte(),
                    'source_coor': torch.t(batch['src_kps']).clone(),
                    'target_coor': torch.t(batch['trg_kps']).clone(),
                    'L_bounding_box': batch['pckthres'], 'sparse': True}

    def get_bbox(self, bbox_list, idx):
        r"""Returns object bounding-box"""
        bbox = bbox_list[idx].clone()
        return bbox
