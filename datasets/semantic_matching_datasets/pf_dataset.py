"""
extracted and modified from https://github.com/ignacio-rocco/weakalign
"""
from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from packaging import version
from torch.utils.data import Dataset
from imageio import imread

from utils_flow.img_processing_utils import pad_to_same_shape


class PFWillowDataset(Dataset):
    """
    Proposal Flow image pair dataset, in particular PF-Willow
    for proposal flow, there are 90 pairs per category, 10 keypoints for each image pair.
    """

    def __init__(self, root, path_list=None, source_image_transform=None, target_image_transform=None,
                 flow_transform=None):
        """
        Args:
            root:
            path_list: path to csv file containing ground-truth info
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

        if path_list is None:
            self.pairs = pd.read_csv(os.path.join(root, 'PF-dataset', 'test_pairs_pf.csv'))
        else:
            self.pairs = pd.read_csv(path_list)
        self.img_A_names = self.pairs.iloc[:, 0]
        self.img_B_names = self.pairs.iloc[:, 1]
        self.point_A_coords = self.pairs.iloc[:, 2:22].values.astype('float')
        self.point_B_coords = self.pairs.iloc[:, 22:].values.astype('float')
        self.root = root
        self.first_image_transform = source_image_transform
        self.second_image_transform = target_image_transform
        self.flow_transform = flow_transform
              
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Pad images to same shape.
        Args:
            idx:

        Returns:
            source_image
            target_image
            source_image_size
            target_image_size
            flow_map
            correspondence_mask: valid correspondences (which are originally sparse).
        """
        # get pre-processed images
        img_target = imread(os.path.join(self.root, self.img_B_names[idx])).astype(np.uint8)
        img_source = imread(os.path.join(self.root, self.img_A_names[idx])).astype(np.uint8)
        image_source_size = img_source.shape
        img_target_size = img_target.shape

        img_source, img_target = pad_to_same_shape(img_source, img_target)
        h_size, w_size, _ = img_target.shape

        # get pre-processed point coords
        point_source_coords = self.get_points(self.point_A_coords, idx)  # shape 10,2
        point_target_coords = self.get_points(self.point_B_coords, idx)

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

        # compute PCK reference length L_pck (equal to max bounding box side in image_A)
        point_A_coords = torch.t(torch.Tensor(point_source_coords))
        L_pck = torch.FloatTensor([torch.max(point_A_coords.max(1)[0]-point_A_coords.min(1)[0])])

        if self.first_image_transform is not None:
            img_source = self.first_image_transform(img_source)
        if self.second_image_transform is not None:
            img_target = self.second_image_transform(img_target)
        if self.flow_transform is not None:
            flow = self.flow_transform(flow)

        return {'source_image': img_source,
                'target_image': img_target,
                'source_image_size': image_source_size,
                'target_image_size': img_target_size,
                'flow_map': flow,
                'correspondence_mask': mask.astype(np.bool) if version.parse(torch.__version__) >= version.parse("1.1") \
                    else mask.astype(np.uint8),
                'source_coor': point_source_coords,
                'target_coor': point_target_coords,
                'L_bounding_box': L_pck}

    def get_points(self, point_coords_list, idx):
        point_coords = point_coords_list[idx, :].reshape(2, 10)
        return point_coords.astype(np.float32).T
    
    
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
            # selects only images with class in labels !
            self.pairs = pairs.loc[pairs['class'].isin(self.labels)]
            # self.categories = self.pairs.ilox[:,2] # list giving the class for each image pair
            '''
            alternative, here category not a list just a name
            
            self.labels = float(np.where(np.array(self.category_names) == category)[0][0] + 1) # indice corresponding to the class, but cannot be several ones 
            self.categories = self.pairs.iloc[:,2].as_matrix().astype('float') # list of the class for each image pair
            if category is not None:
                cat_idx = np.nonzero(self.categories==self.labels)[0] # gets a list of indices of the pairs corresponding to a particular category 
                self.categories=self.categories[cat_idx] # ==> list giving the class for each image pair
                self.pairs=self.pairs.iloc[cat_idx,:]
            '''
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
        image_target_size = np.float32(img_target.shape)  # size of the original images

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
        else:
            img_source, img_target = pad_to_same_shape(img_source, img_target)
            L_pck = max(image_source_size[0], image_source_size[1])  # original images size
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
                  'correspondence_mask': mask.astype(np.bool) if version.parse(torch.__version__) >= version.parse("1.1") \
                    else mask.astype(np.uint8),
                  'source_coor': point_source_coords,
                  'target_coor': point_target_coords,
                  'L_bounding_box': L_pck}
        return output

    def get_points(self, point_coords_list, idx):
        X = np.fromstring(point_coords_list.iloc[idx, 0],sep=';')
        Y = np.fromstring(point_coords_list.iloc[idx, 1],sep=';')
        N_points = X.shape[0]

        point_coords = np.concatenate((X.reshape(N_points, 1), Y.reshape(N_points, 1)), axis=1)
        
        # make arrays float tensor for subsequent processing
        # point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords.astype(np.float32)
