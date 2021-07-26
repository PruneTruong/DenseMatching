import json
import glob
import os
import imageio
import torch
from .semantic_keypoints_datasets import SemanticKeypointsDataset, random_crop
from datasets.util import pad_to_same_shape
from datasets.util import define_mask_zero_borders
import random


class SPairDataset(SemanticKeypointsDataset):
    """Spair dataset"""
    def __init__(self,  root, split, thres='bbox', source_image_transform=None,
                 target_image_transform=None, flow_transform=None, training_cfg=None):
        """
        Args:
            root:
            split:
            thres:
            source_image_transform: image transformations to apply to source images
            target_image_transform: image transformations to apply to target images
            flow_transform: flow transformations to apply to ground-truth flow fields
            training_cfg:
        Output in __getittem__ (for split=='test'):
            source_image
            target_image
            source_image_size
            target_image_size
            flow_map
            correspondence_mask: valid correspondences (which are originally sparse).
        """
        super(SPairDataset, self).__init__('spair', root, thres, split, source_image_transform,
                                           target_image_transform, flow_transform, training_cfg=training_cfg)

        self.train_data = open(self.spt_path).read().split('\n')
        self.train_data = self.train_data[:len(self.train_data) - 1]
        self.src_imnames = list(map(lambda x: x.split('-')[1] + '.jpg', self.train_data))
        self.trg_imnames = list(map(lambda x: x.split('-')[2].split(':')[0] + '.jpg', self.train_data))
        self.cls = os.listdir(self.img_path)
        self.cls.sort()

        anntn_files = []
        for data_name in self.train_data:
            anntn_files.append(glob.glob('%s/%s.json' % (self.ann_path, data_name))[0])
        anntn_files = list(map(lambda x: json.load(open(x)), anntn_files))
        self.src_kps = list(map(lambda x: torch.tensor(x['src_kps']).t().float(), anntn_files))
        self.trg_kps = list(map(lambda x: torch.tensor(x['trg_kps']).t().float(), anntn_files))
        self.src_bbox = list(map(lambda x: torch.tensor(x['src_bndbox']).float(), anntn_files))
        self.trg_bbox = list(map(lambda x: torch.tensor(x['trg_bndbox']).float(), anntn_files))
        self.cls_ids = list(map(lambda x: self.cls.index(x['category']), anntn_files))

        self.vpvar = list(map(lambda x: torch.tensor(x['viewpoint_variation']), anntn_files))
        self.scvar = list(map(lambda x: torch.tensor(x['scale_variation']), anntn_files))
        self.trncn = list(map(lambda x: torch.tensor(x['truncation']), anntn_files))
        self.occln = list(map(lambda x: torch.tensor(x['occlusion']), anntn_files))

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
        batch = super(SPairDataset, self).__getitem__(idx)

        batch['src_bbox'] = self.get_bbox(self.src_bbox, idx)
        batch['trg_bbox'] = self.get_bbox(self.trg_bbox, idx)

        if self.split != 'test':
            # for training, might want to have different output flow sizes
            if self.training_cfg['augment_with_crop']:
                batch['src_img'], batch['src_kps'] = random_crop(batch['src_img'], batch['src_kps'],
                                                                 batch['src_bbox'].clone(),
                                                                 size=self.training_cfg['crop_size'])
                batch['trg_img'], batch['trg_kps'] = random_crop(batch['trg_img'], batch['trg_kps'],
                                                                 batch['trg_bbox'].clone(),
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

            '''
            if self.output_image_size is not None:
                batch['trg_img'] = cv2.resize(batch['trg_img'], dsize=(self.output_image_size[1],
                                                                       self.output_image_size[0]),
                                              interpolation=cv2.INTER_LINEAR)
                batch['src_kps'][0] *= self.output_image_size[1] / batch['src_imsize'][1]
                batch['src_kps'][1] *= self.output_image_size[0] / batch['src_imsize'][0]
                batch['src_bbox'][0::2] *= self.output_image_size[1] / batch['src_imsize'][1]
                batch['src_bbox'][1::2] *= self.output_image_size[0] / batch['src_imsize'][0]
                batch['pckthres'] = self.get_pckthres(batch, self.output_image_size)

                batch['trg_img'] = cv2.resize(batch['trg_img'], dsize=(self.output_image_size[1],
                                                                       self.output_image_size[0]),
                                              interpolation=cv2.INTER_LINEAR)
                batch['trg_kps'][0] *= self.output_image_size[1] / batch['trg_imsize'][1]
                batch['trg_kps'][1] *= self.output_image_size[0] / batch['trg_imsize'][0]
                batch['trg_bbox'][0::2] *= self.output_image_size[1] / batch['trg_imsize'][1]
                batch['trg_bbox'][1::2] *= self.output_image_size[0] / batch['trg_imsize'][0]
            '''

            batch['src_img'], batch['trg_img'] = pad_to_same_shape(batch['src_img'], batch['trg_img'])
            h_size, w_size, _ = batch['trg_img'].shape
            # batch['src_kpidx'] = self.match_idx(batch['src_kps'], batch['n_pts'])
            # batch['trg_kpidx'] = self.match_idx(batch['trg_kps'], batch['n_pts'])
            batch['vpvar'] = self.vpvar[idx]
            batch['scvar'] = self.scvar[idx]
            batch['trncn'] = self.trncn[idx]
            batch['occln'] = self.occln[idx]

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

    def get_image(self, img_names, idx):
        r"""Returns image tensor"""
        path = os.path.join(self.img_path, self.cls[self.cls_ids[idx]], img_names[idx])

        return imageio.imread(path)

    def get_bbox(self, bbox_list, idx):
        r"""Returns object bounding-box"""
        bbox = bbox_list[idx].clone()
        return bbox