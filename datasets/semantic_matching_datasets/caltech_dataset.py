r"""Caltech-101 dataset"""
import os

import pandas as pd
import numpy as np
import torch


from utils_flow.img_processing_utils import pad_to_same_shape
from .semantic_keypoints_datasets import SemanticKeypointsDataset


class CaltechDataset(SemanticKeypointsDataset):
    r"""Inherits CorrespondenceDataset"""
    def __init__(self, root, split='test', thres='bbox', source_image_transform=None,
                 target_image_transform=None, flow_transform=None, output_image_size=None):
        """
        Args:
            root:
            split: 'test', 'val', 'train'
            source_image_transform: image transformations to apply to source images
            target_image_transform: image transformations to apply to target images
            flow_transform: flow transformations to apply to ground-truth flow fields
            output_image_size: size if images and annotations need to be resized, used when split=='test'
        Output in __getittem__  (for split=='test'):
            source_image
            target_image
            source_image_size
            target_image_size
            source_kps
            target_kps
        """
        super(CaltechDataset, self).__init__('caltech', root, thres, split, source_image_transform,
                                             target_image_transform, flow_transform,
                                             output_image_size=output_image_size)

        self.train_data = pd.read_csv(self.spt_path)
        self.src_imnames = np.array(self.train_data.iloc[:, 0])
        self.trg_imnames = np.array(self.train_data.iloc[:, 1])
        self.src_kps = self.train_data.iloc[:, 3:5]
        self.trg_kps = self.train_data.iloc[:, 5:]
        self.cls = ['Faces', 'Faces_easy', 'Leopards', 'Motorbikes', 'accordion', 'airplanes',
                    'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain',
                    'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'car_side',
                    'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_body',
                    'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup',
                    'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar',
                    'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo', 'flamingo_head',
                    'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone',
                    'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo',
                    'ketch', 'lamp', 'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly',
                    'menorah', 'metronome', 'minaret', 'nautilus', 'octopus', 'okapi', 'pagoda',
                    'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino',
                    'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion', 'sea_horse',
                    'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign',
                    'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch',
                    'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']
        self.cls_ids = self.train_data.iloc[:, 2].values.astype('int') - 1
        self.src_imnames = list(map(lambda x: os.path.join(*x.split('/')[1:]), self.src_imnames))
        self.trg_imnames = list(map(lambda x: os.path.join(*x.split('/')[1:]), self.trg_imnames))

        # if need to resize the images, even for testing
        if output_image_size is not None:
            if not isinstance(output_image_size, tuple):
                output_image_size = (output_image_size, output_image_size)
        self.output_image_size = output_image_size

    def __getitem__(self, idx):
        r"""Constructs and returns a batch for Caltech-101 dataset"""
        batch = super(CaltechDataset, self).__getitem__(idx)
        batch['source_image'], batch['target_image'] = pad_to_same_shape(batch['source_image'], batch['target_image'])
        h_size, w_size, _ = batch['target_image'].shape

        if self.source_image_transform is not None:
            batch['source_image'] = self.source_image_transform(batch['source_image'])
        if self.target_image_transform is not None:
            batch['target_image'] = self.target_image_transform(batch['target_image'])
        return batch

    def get_pckthres(self, batch, imsize):
        r"""No PCK measure for Caltech-101 dataset"""
        return None

    def get_points(self, pts, idx, org_imsize):
        r"""Return mask-points of an image"""
        x_pts = torch.tensor(list(map(lambda pt: float(pt), pts[pts.columns[0]][idx].split(','))))
        y_pts = torch.tensor(list(map(lambda pt: float(pt), pts[pts.columns[1]][idx].split(','))))

        if self.output_image_size is not None:
            # resize
            x_pts *= self.output_image_size[1] / org_imsize[1]  # w
            y_pts *= self.output_image_size[0] / org_imsize[0]  # h

        n_pts = x_pts.size(0)
        if n_pts > self.max_pts:
            raise Exception('The number of keypoints is above threshold: %d' % n_pts)
        pad_pts = torch.zeros((2, self.max_pts - n_pts)) - 1

        kps = torch.cat([torch.stack([x_pts, y_pts]), pad_pts], dim=1)

        return torch.t(kps), n_pts
