from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from packaging import version
import scipy.io as sio
import random


from utils_flow.img_processing_utils import pad_to_same_shape
from .semantic_keypoints_datasets import SemanticKeypointsDataset, random_crop
from utils_flow.img_processing_utils import define_mask_zero_borders


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
            output_image_size: size if images and annotations need to be resized, used when split=='test'
            training_cfg: training config
        Output in __getittem__  (for split=='test'):
            source_image
            target_image
            source_image_size
            target_image_size
            flow_map
            correspondence_mask: valid correspondences (which are originally sparse)
            source_kps
            target_kps
        """
        super(PFPascalDataset, self).__init__('pfpascal', root, thres, split, source_image_transform,
                                              target_image_transform, flow_transform, training_cfg=training_cfg,
                                              output_image_size=output_image_size)

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
                if torch.isnan(src_kk).sum() > 0 or torch.isnan(trg_kk).sum() > 0:
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

        Returns: If split is test, dictionary with fieldnames:
            source_image
            target_image
            source_image_size
            target_image_size
            flow_map
            correspondence_mask: valid correspondences (which are originally sparse)
            source_kps
            target_kps
        """
        batch = super(PFPascalDataset, self).__getitem__(idx)

        batch['sparse'] = True
        batch['src_bbox'] = self.get_bbox(self.src_bbox, idx, batch['src_imsize_ori'])
        batch['trg_bbox'] = self.get_bbox(self.trg_bbox, idx, batch['trg_imsize_ori'])

        if self.split != 'test':
            # for training, might want to have different output flow sizes
            if self.training_cfg['augment_with_crop']:
                batch['source_image'], batch['source_kps'], batch['src_bbox'] = random_crop(
                    batch['source_image'], batch['source_kps'].clone(), batch['src_bbox'].int(),
                    size=self.training_cfg['crop_size'], p=self.training_cfg['proba_of_crop'])

                batch['target_image'], batch['target_kps'], batch['trg_bbox'] = random_crop(
                    batch['target_image'], batch['target_kps'].clone(), batch['trg_bbox'].int(),
                    size=self.training_cfg['crop_size'], p=self.training_cfg['proba_of_crop'])

            if self.training_cfg['augment_with_flip']:
                if random.random() < self.training_cfg['proba_of_batch_flip']:
                    self.horizontal_flip(batch)
                else:
                    if random.random() < self.training_cfg['proba_of_image_flip']:
                        batch['source_image'], batch['src_bbox'], batch['source_kps'] = self.horizontal_flip_img(
                            batch['source_image'], batch['src_bbox'], batch['source_kps'])
                    if random.random() < self.training_cfg['proba_of_image_flip']:
                        batch['target_image'], batch['trg_bbox'], batch['target_kps'] = self.horizontal_flip_img(
                            batch['target_image'], batch['trg_bbox'], batch['target_kps'])

            '''
            # Horizontal flipping of both images and key-points during training
            if self.split == 'train' and self.flip[idx]:
                self.horizontal_flip(batch)
                batch['flip'] = 1
            else:
                batch['flip'] = 0
            '''

            batch = self.recover_image_pair_for_training(batch)
            batch['src_bbox'] = self.get_bbox(self.src_bbox, idx, batch['src_imsize_ori'],
                                              output_image_size=self.training_cfg['output_image_size'])
            batch['trg_bbox'] = self.get_bbox(self.trg_bbox, idx, batch['trg_imsize_ori'],
                                              output_image_size=self.training_cfg['output_image_size'])
            batch['pckthres'] = self.get_pckthres(batch, batch['source_image_size'])

            if self.source_image_transform is not None:
                batch['source_image'] = self.source_image_transform(batch['source_image'])
            if self.target_image_transform is not None:
                batch['target_image'] = self.target_image_transform(batch['target_image'])

            flow = batch['flow_map']
            if self.flow_transform is not None:
                if type(flow) in [tuple, list]:
                    # flow field at different resolution
                    for i in range(len(flow)):
                        flow[i] = self.flow_transform(flow[i])
                else:
                    flow = self.flow_transform(flow)
            batch['flow_map'] = flow

            if self.training_cfg['compute_mask_zero_borders']:
                mask_valid = define_mask_zero_borders(batch['target_image'])
                batch['mask_zero_borders'] = mask_valid
        else:
            batch['src_bbox'] = self.get_bbox(self.src_bbox, idx, batch['src_imsize_ori'],
                                              output_image_size=self.output_image_size)
            batch['trg_bbox'] = self.get_bbox(self.trg_bbox, idx, batch['trg_imsize_ori'],
                                              output_image_size=self.output_image_size)
            batch['pckthres'] = self.get_pckthres(batch, batch['source_image_size'])

            batch['source_image'], batch['target_image'] = pad_to_same_shape(batch['source_image'],
                                                                             batch['target_image'])
            h_size, w_size, _ = batch['target_image'].shape

            flow, mask = self.keypoints_to_flow(batch['source_kps'][:batch['n_pts']],
                                                batch['target_kps'][:batch['n_pts']],
                                                h_size=h_size, w_size=w_size)

            if self.source_image_transform is not None:
                batch['source_image'] = self.source_image_transform(batch['source_image'])
            if self.target_image_transform is not None:
                batch['target_image'] = self.target_image_transform(batch['target_image'])
            if self.flow_transform is not None:
                flow = self.flow_transform(flow)

            batch['flow_map'] = flow
            batch['correspondence_mask'] = mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") \
                else mask.byte()

        return batch

    def get_bbox(self, bbox_list, idx, original_image_size=None, output_image_size=None):
        r"""Returns object bounding-box"""
        bbox = bbox_list[idx].clone()
        if self.output_image_size is not None or output_image_size is not None:
            if output_image_size is None:
                bbox[0::2] *= (self.output_image_size[1] / original_image_size[1])  # w
                bbox[1::2] *= (self.output_image_size[0] / original_image_size[0])
            else:
                bbox[0::2] *= (float(output_image_size[1]) / float(original_image_size[1]))
                bbox[1::2] *= (float(output_image_size[0]) / float(original_image_size[0]))
        return bbox
