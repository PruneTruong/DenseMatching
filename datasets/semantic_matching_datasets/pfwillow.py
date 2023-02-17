import os
import torch
import pandas as pd
import numpy as np
from packaging import version


from utils_flow.img_processing_utils import pad_to_same_shape
from .semantic_keypoints_datasets import SemanticKeypointsDataset


class PFWillowDataset(SemanticKeypointsDataset):
    """
    Proposal Flow image pair dataset, in particular PF-Willow
    for proposal flow, there are 90 pairs per category, 10 keypoints for each image pair.
    """

    def __init__(self, root, split='test', thres='bbox', source_image_transform=None,
                 target_image_transform=None, flow_transform=None, output_image_size=None):
        super(PFWillowDataset, self).__init__('pfwillow', root, thres, split, source_image_transform,
                                              target_image_transform, flow_transform)
        """
        Args:
            root:
            source_image_transform: image transformations to apply to source images
            target_image_transform: image transformations to apply to target images
            flow_transform: flow transformations to apply to ground-truth flow fields
            output_image_size: size if images and annotations need to be resized, used when split=='test'
        Output in __getittem__:
            source_image
            target_image
            source_image_size
            target_image_size
            flow_map
            correspondence_mask: valid correspondences (which are originally sparse)
            source_kps
            target_kps
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
            correspondence_mask: valid correspondences (which are originally sparse)
            source_kps
            target_kps
        """
        batch = super(PFWillowDataset, self).__getitem__(idx)
        batch['pckthres'] = self.get_pckthres(batch, batch['source_image_size'])

        batch['source_image'], batch['target_image'] = pad_to_same_shape(batch['source_image'], batch['target_image'])
        h_size, w_size, _ = batch['target_image'].shape

        flow, mask = self.keypoints_to_flow(batch['source_kps'][:batch['n_pts']],
                                            batch['target_kps'][:batch['n_pts']], h_size=h_size, w_size=w_size)

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

    def get_pckthres(self, batch, img_size):
        """Computes PCK threshold"""
        if self.thres == 'bbox':
            return max(torch.t(batch['source_kps']).max(1)[0] - torch.t(batch['source_kps']).min(1)[0]).clone()
        elif self.thres == 'img':
            return torch.tensor(max(batch['source_image'].shape[0], batch['source_image'].shape[1]))
        else:
            raise Exception('Invalid pck evaluation level: %s' % self.thres)

    def get_points(self, pts_list, idx, org_imsize):
        """Returns key-points of an image"""
        point_coords = pts_list[idx, :].reshape(2, 10).copy()
        point_coords = torch.tensor(point_coords.astype(np.float32))

        if self.output_image_size is not None:
            # resize
            point_coords[0] *= self.output_image_size[1] / org_imsize[1]  # w
            point_coords[1] *= self.output_image_size[0] / org_imsize[0]  # h

        xy, n_pts = point_coords.size()
        return torch.t(point_coords), n_pts
