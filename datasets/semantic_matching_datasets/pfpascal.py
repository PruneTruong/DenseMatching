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
                batch['src_img'], batch['src_kps'] = random_crop(batch['src_img'], batch['src_kps'],
                                                                 batch['src_bbox'].int(),
                                                                 size=self.training_cfg['crop_size'])
                batch['trg_img'], batch['trg_kps'] = random_crop(batch['trg_img'], batch['trg_kps'],
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
                                                                              kp_source=torch.t(batch['src_kps']),
                                                                              kp_target=torch.t(batch['trg_kps']))

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
                    'source_coor': torch.t(batch['src_kps']),
                    'target_coor': torch.t(batch['trg_kps']),
                    'L_bounding_box': batch['pckthres'], 'sparse': True}

    def get_bbox(self, bbox_list, idx):
        r"""Returns object bounding-box"""
        bbox = bbox_list[idx].clone()
        return bbox


'''
# Alternative    
class PFPascalDataset(Dataset):
    """
    Proposal Flow image pair dataset (PF-Pascal).
    There is a certain number of pairs per category and the number of keypoints per pair also varies
    """
    def __init__(self, root, path_list=None, labels=None, category=None, source_image_transform=None,
                 target_image_transform=None, flow_transform=None, pck_procedure='scnet', evaluate_at_original_image_reso=True,
                 normalize=False, apply_padding_to_same_shape=True):
        """

        Args:
            root:
            path_list: path to csv file containing ground-truth info
            labels:
            category:
            source_image_transform: image transformations to apply to source images
            target_image_transform: image transformations to apply to target images
            flow_transform: flow transformations to apply to ground-truth flow fields
            pck_procedure:
            evaluate_at_original_image_reso:
            normalize:
            apply_padding_to_same_shape:
        """

        self.category_names=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
                             'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

        if path_list is None:
            pairs = pd.read_csv(os.path.join(root, 'PF-dataset-PASCAL', 'original_csv_files_weakalign',
                                             'test_pairs_pf_pascal.csv'))
        else:
            pairs = pd.read_csv(path_list)

        if labels is not None or category is not None:
            # select pairs we want based on the class
            # label is list of class index that we want or category is list of classes name that we want
            if category is not None:
                self.labels = float(np.where(np.array(self.category_names) == category)[0][0] + 1)
                # list of indices corresponding to the category name
            else:
                # labels is not None
                self.labels = labels
        else:
            # takes all pairs
            self.pairs = pairs

        self.img_A_names = self.pairs.iloc[:,0]
        self.img_B_names = self.pairs.iloc[:,1]
        self.point_A_coords = self.pairs.iloc[:, 3:5]
        self.point_B_coords = self.pairs.iloc[:, 5:]
        self.root = root
        self.first_image_transform = source_image_transform
        self.second_image_transform = target_image_transform
        self.flow_transform = flow_transform
        self.pck_procedure = pck_procedure
        self.apply_padding_to_same_shape = apply_padding_to_same_shape
        self.evaluate_at_original_image_reso = evaluate_at_original_image_reso
              
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # get pre-processed images
        img_target = imread(os.path.join(self.root, self.img_B_names.iloc[idx])).astype(np.uint8)
        img_source = imread(os.path.join(self.root, self.img_A_names.iloc[idx])).astype(np.uint8)
        image_source_size = np.float32(img_source.shape)
        image_target_size = np.float32(img_target.shape)  # load_size of the original images

        # get pre-processed point coords
        point_source_coords = self.get_points(self.point_A_coords, idx)
        point_target_coords = self.get_points(self.point_B_coords, idx)

        N_pts = point_source_coords.shape[0]

        if self.pck_procedure == 'pf':
            # max dimension of bouding box
            img_source, img_target = pad_to_same_shape(img_source, img_target)

            point_A_coords = torch.t(torch.Tensor(point_source_coords))
            L_pck = torch.FloatTensor([torch.max(point_A_coords[:, :N_pts].max(1)[0]-
                                                 point_A_coords[:, :N_pts].min(1)[0])])
            h_size, w_size, _ = img_target.shape
        elif self.pck_procedure == 'scnet':
            # modification to follow the evaluation procedure of SCNet
            # need to resize images to 224x224

            if self.evaluate_at_original_image_reso:
                img_source, img_target = pad_to_same_shape(img_source, img_target)
            else:
                img_target = cv2.resize(img_target, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
                img_source = cv2.resize(img_source, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
                image_source_size[0:2] = 224
                image_target_size[0:2] = 224

            point_source_coords[:, 0] *= 224.0/image_source_size[1]
            point_source_coords[:, 1] *= 224.0/image_source_size[0]
            point_target_coords[:, 0] *= 224.0/image_target_size[1]
            point_target_coords[:, 1] *= 224.0/image_target_size[0]

            L_pck = torch.FloatTensor([224.0])  # image load_size
            h_size = w_size = 224
        else:
            img_source, img_target = pad_to_same_shape(img_source, img_target)
            L_pck = max(image_source_size[0], image_source_size[1])  # original images load_size
            h_size, w_size, _ = img_target.shape

        flow = np.zeros((h_size, w_size, 2), dtype=np.float32)
        mask = np.zeros((h_size, w_size), dtype=np.uint8)

        # computes the flow
        valid_target = np.logical_and(np.int32(np.round(point_target_coords[:, 0])) < w_size,
                                      np.int32(np.round(point_target_coords[:, 1])) < h_size)
        # valid = valid_source * valid_target
        valid = valid_target
        point_target_coords = point_target_coords[valid]
        point_source_coords = point_source_coords[valid]

        flow[np.int32(np.round(point_target_coords[:, 1])), np.int32(np.round(point_target_coords[:, 0]))] = \
            point_source_coords - point_target_coords
        mask[np.int32(np.round(point_target_coords[:, 1])), np.int32(np.round(point_target_coords[:, 0]))] = 1

        if self.first_image_transform is not None:
            img_source = self.first_image_transform(img_source)
        if self.second_image_transform is not None:
            img_target = self.second_image_transform(img_target)
        if self.flow_transform is not None:
            flow = self.flow_transform(flow)

        output = {'source_image': img_source,
                  'target_image': img_target,
                  'source_image_size': image_source_size,
                  'target_image_size': image_target_size,
                  'flow_map': flow,
                  'correspondence_mask': mask.astype(np.bool) if float(torch.__version__[:3]) >= 1.1 \
                    else mask.astype(np.uint8),
                  'source_coor': point_source_coords,  # shape N,2
                  'target_coor': point_target_coords,  # shape N,2 not rounded yet
                  'L_bounding_box': L_pck}
        if self.evaluate_at_original_image_reso and self.pck_procedure == 'scnet':
            output['resizing_to'] = 224
        return output

    def get_points(self, point_coords_list, idx):
        X = np.fromstring(point_coords_list.iloc[idx, 0],sep=';')
        Y = np.fromstring(point_coords_list.iloc[idx, 1],sep=';')
        N_points = X.shape[0]

        point_coords = np.concatenate((X.reshape(N_points, 1), Y.reshape(N_points, 1)), axis=1)
        
        # make arrays float tensor for subsequent processing
        # point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords.astype(np.float32)

'''