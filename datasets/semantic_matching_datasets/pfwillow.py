import os
import torch
import pandas as pd
import numpy as np
from datasets.util import pad_to_same_shape
from .semantic_keypoints_datasets import SemanticKeypointsDataset


class PFWillowDataset(SemanticKeypointsDataset):
    """
    Proposal Flow image pair dataset, in particular PF-Willow
    for proposal flow, there are 90 pairs per category, 10 keypoints for each image pair.
    """

    def __init__(self, root, split='test', thres='bbox', source_image_transform=None,
                 target_image_transform=None, flow_transform=None):
        super(PFWillowDataset, self).__init__('pfwillow', root, thres, split, source_image_transform,
                                                       target_image_transform, flow_transform)
        """
        Args:
            root:
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

        self.train_data = pd.read_csv(self.spt_path)
        self.src_imnames = np.array(self.train_data.iloc[:, 0])
        self.trg_imnames = np.array(self.train_data.iloc[:, 1])
        self.src_kps = self.train_data.iloc[:, 2:22].values
        self.trg_kps = self.train_data.iloc[:, 22:].values
        self.cls = ['car(G)', 'car(M)', 'car(S)', 'duck(S)',
                    'motorbike(G)', 'motorbike(M)', 'motorbike(S)',
                    'winebottle(M)', 'winebottle(wC)', 'winebottle(woC)']
        self.cls_ids = list(map(lambda names: self.cls.index(names.split('/')[1]), self.src_imnames))
        self.src_imnames = list(map(lambda x: os.path.join(*x.split('/')[1:]), self.src_imnames))
        self.trg_imnames = list(map(lambda x: os.path.join(*x.split('/')[1:]), self.trg_imnames))

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
        batch = super(PFWillowDataset, self).__getitem__(idx)
        batch['pckthres'] = self.get_pckthres(batch, batch['src_imsize'])

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

    def get_pckthres(self, batch, img_size):
        """Computes PCK threshold"""
        if self.thres == 'bbox':
            return max(batch['src_kps'].max(1)[0] - batch['src_kps'].min(1)[0]).clone()
        elif self.thres == 'img':
            return torch.tensor(max(batch['src_img'].shape[0], batch['src_img'].shape[1]))
        else:
            raise Exception('Invalid pck evaluation level: %s' % self.thres)

    def get_points(self, pts_list, idx):
        """Returns key-points of an image"""
        point_coords = pts_list[idx, :].reshape(2, 10)
        point_coords = torch.tensor(point_coords.astype(np.float32))
        xy, n_pts = point_coords.size()
        return point_coords, n_pts


'''
# Alternative
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
                'correspondence_mask': mask.astype(np.bool) if float(torch.__version__[:3]) >= 1.1 \
                    else mask.astype(np.uint8),
                'source_coor': point_source_coords,  # shape 10,2
                'target_coor': point_target_coords,  # shape 10,2 not rounded yet
                'L_bounding_box': L_pck}

    def get_points(self, point_coords_list, idx):
        point_coords = point_coords_list[idx, :].reshape(2, 10)
        return point_coords.astype(np.float32).T
'''