import os
import random
import imageio
import torch
import cv2
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from ..util import pad_to_same_shape, pad_to_size, resize_keeping_aspect_ratio


def resize(img, kps, size=(256, 256)):
    h, w = img.shape[:2]
    resized_img = cv2.resize(img, dsize=(size[1], size[0]), interpolation=cv2.INTER_LINEAR)

    kps = kps.t()
    resized_kps = torch.zeros_like(kps, dtype=torch.float)
    resized_kps[:, 0] = kps[:, 0] * (size[1] / w)
    resized_kps[:, 1] = kps[:, 1] * (size[0] / h)

    return resized_img, resized_kps.t()


def random_crop(img, kps, bbox, size=(256, 256), p=0.5):
    if random.uniform(0, 1) > p:
        return resize(img, kps, size)
    h, w = img.shape[:2]
    kps = kps.t()
    left = random.randint(0, bbox[0])
    top = random.randint(0, bbox[1])
    height = random.randint(bbox[3], h) - top
    width = random.randint(bbox[2], w) - left

    resized_img = img[top: top + height, left: left + width]
    resized_img = cv2.resize(resized_img, dsize=(size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    resized_kps = torch.zeros_like(kps, dtype=torch.float)
    resized_kps[:, 0] = (kps[:, 0] - left) * (size[1] / width)
    resized_kps[:, 1] = (kps[:, 1] - top) * (size[0] / height)
    # resized_kps = torch.clamp(resized_kps, 0, size[0] - 1)

    return resized_img, resized_kps.t()


class SemanticKeypointsDataset(Dataset):
    """Parent class of PFPascal, PFWillow, Caltech, and SPair"""

    def __init__(self, benchmark, root, thres, split, source_image_transform=None,
                 target_image_transform=None, flow_transform=None, training_cfg=None,
                 ):
        """CorrespondenceDataset constructor"""
        super(SemanticKeypointsDataset, self).__init__()

        # {Directory name, Layout path, Image path, Annotation path, PCK threshold}
        self.metadata = {
            'pfwillow': (os.path.basename(root),  # usually 'PF-dataset',
                         'test_pairs_pf.csv',
                         '',
                         '',
                         'bbox'),
            'pfpascal': (os.path.basename(root),  # usually 'PF-dataset-PASCAL',
                         '_pairs_pf_pascal.csv',
                         'JPEGImages',
                         'Annotations',
                         'img'),
            'caltech': (os.path.basename(root),  # usually 'Caltech-101',
                        'test_pairs_caltech_with_category.csv',
                        '101_ObjectCategories',
                        '',
                        ''),
            'spair': (os.path.basename(root),  # usually 'SPair-71k',
                      'Layout/large',
                      'JPEGImages',
                      'PairAnnotation',
                      'bbox')
        }
        root = os.path.dirname(root)  # remove the last name

        default_conf = {
            'seed': 400,
            'two_views': True,
            'exchange_images_with_proba': 0.5,
            'augment_with_crop': True,
            'augment_with_flip': False,
            'proba_of_image_flip': 0.5,
            'proba_of_batch_flip': 0.5,
            'crop_size': [400, 400],
            'output_image_size': [400, 400],
            'pad_to_same_shape': True,
            'output_flow_size': [[400, 400], [256, 256]],
            'compute_mask_zero_borders': False
        }
        self.training_cfg = default_conf

        if training_cfg is not None:
            self.training_cfg.update(training_cfg)

        # Directory path for train, val, or test splits
        base_path = os.path.join(os.path.abspath(root), self.metadata[benchmark][0])
        if benchmark == 'pfpascal':
            self.spt_path = os.path.join(base_path, split + self.metadata[benchmark][1])
        elif benchmark == 'spair':
            self.spt_path = os.path.join(base_path, self.metadata[benchmark][1], split + '.txt')
        else:
            self.spt_path = os.path.join(base_path, self.metadata[benchmark][1])

        # Directory path for images
        self.img_path = os.path.join(base_path, self.metadata[benchmark][2])

        # Directory path for annotations
        if benchmark == 'spair':
            self.ann_path = os.path.join(base_path, self.metadata[benchmark][3], split)
        else:
            self.ann_path = os.path.join(base_path, self.metadata[benchmark][3])

        # Miscellaneous
        if benchmark == 'caltech':
            self.max_pts = 400
        else:
            self.max_pts = 40
        self.split = split
        self.benchmark = benchmark
        self.range_ts = torch.arange(self.max_pts)
        self.thres = self.metadata[benchmark][4] if thres == 'auto' else thres

        # To get initialized in subclass constructors
        self.train_data = []
        self.src_imnames = []
        self.trg_imnames = []
        self.cls = []
        self.cls_ids = []
        self.src_kps = []
        self.trg_kps = []
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.flow_transform = flow_transform

    def __len__(self):
        """Returns the number of pairs"""
        return len(self.train_data)

    def __getitem__(self, idx):
        """Constructs and return a batch"""

        # Image names
        batch = dict()
        batch['src_imname'] = self.src_imnames[idx]
        batch['trg_imname'] = self.trg_imnames[idx]

        # Class of instances in the images
        batch['category_id'] = self.cls_ids[idx]
        batch['category'] = self.cls[batch['category_id']]

        # Image as numpy (original height, original width)
        src_pil = self.get_image(self.src_imnames, idx)
        trg_pil = self.get_image(self.trg_imnames, idx)
        batch['src_imsize'] = src_pil.shape  # h,w
        batch['trg_imsize'] = trg_pil.shape

        batch['src_img'] = src_pil
        batch['trg_img'] = trg_pil

        # Key-points (re-scaled)
        batch['src_kps'], num_pts = self.get_points(self.src_kps, idx)
        batch['trg_kps'], _ = self.get_points(self.trg_kps, idx)
        batch['n_pts'] = torch.tensor(num_pts)

        # The number of pairs in training split
        batch['datalen'] = len(self.train_data)

        return batch

    def get_image(self, imnames, idx):
        """Reads numpy image from path"""
        path = os.path.join(self.img_path, imnames[idx])
        return imageio.imread(path)

    def get_pckthres(self, batch, imsize):
        """Computes PCK threshold"""
        if self.thres == 'bbox':
            bbox = batch['src_bbox'].clone()
            bbox_w = (bbox[2] - bbox[0])
            bbox_h = (bbox[3] - bbox[1])
            pckthres = torch.max(bbox_w, bbox_h)
        elif self.thres == 'img':
            imsize_t = imsize
            pckthres = torch.tensor(max(imsize_t[0], imsize_t[1]))
        else:
            raise Exception('Invalid pck threshold type: %s' % self.thres)
        return pckthres.float()

    def get_points(self, pts_list, idx):
        """Returns key-points of an image"""
        # already tensors here, return is torch tensor 2xN
        point_coords = pts_list[idx]
        xy, n_pts = pts_list[idx].size()
        return point_coords, n_pts

    @staticmethod
    def keypoints_to_flow(point_source_coords, point_target_coords, h_size, w_size):
        flow = torch.zeros((h_size, w_size, 2)).float()
        mask = torch.zeros((h_size, w_size)).byte()

        # computes the flow
        valid_target = torch.round(point_target_coords[:, 0]).le(w_size - 1) & \
                       torch.round(point_target_coords[:, 1]).le(h_size - 1) & \
                       torch.round(point_target_coords[:, 0]).ge(0) & torch.round(point_target_coords[:, 1]).ge(0)
        # valid = valid_source * valid_target
        valid = valid_target
        point_target_coords = point_target_coords[valid]
        point_source_coords = point_source_coords[valid]

        flow[torch.round(point_target_coords[:, 1]).long(), torch.round(point_target_coords[:, 0]).long()] = \
            point_source_coords - point_target_coords
        mask[torch.round(point_target_coords[:, 1]).long(), torch.round(point_target_coords[:, 0]).long()] = 1
        return flow, mask

    def recover_image_pair_for_training(self, source, target, kp_source, kp_target):

        if self.training_cfg['pad_to_same_shape']:
            # either pad to same shape
            source, target = pad_to_same_shape(source, target)

        if self.training_cfg['output_image_size'] is not None:
            if isinstance(self.training_cfg['output_image_size'], list):
                # resize to a fixed load_size and rescale the keypoints accordingly
                h1, w1 = source.shape[:2]
                source = cv2.resize(source, (self.training_cfg['output_image_size'][1],
                                             self.training_cfg['output_image_size'][0]))
                kp_source[:, 0] *= float(self.training_cfg['output_image_size'][1]) / float(w1)
                kp_source[:, 1] *= float(self.training_cfg['output_image_size'][0]) / float(h1)

                h2, w2 = target.shape[:2]
                target = cv2.resize(target, (self.training_cfg['output_image_size'][1],
                                             self.training_cfg['output_image_size'][0]))
                kp_target[:, 0] *= float(self.training_cfg['output_image_size'][1]) / float(w2)
                kp_target[:, 1] *= float(self.training_cfg['output_image_size'][0]) / float(h2)
            else:
                # rescale both images so that the largest dimension is equal to the desired load_size of image and
                # then pad to obtain load_size 256x256 or whatever desired load_size. and change keypoints accordingly
                source, ratio_1 = resize_keeping_aspect_ratio(source, self.training_cfg['output_image_size'])
                source = pad_to_size(source, self.training_cfg['output_image_size'])
                kp_source *= ratio_1

                target, ratio_2 = resize_keeping_aspect_ratio(target, self.training_cfg['output_image_size'])
                target = pad_to_size(target, self.training_cfg['output_image_size'])
                kp_target *= ratio_2

        h, w, _ = target.shape
        # create the flow field from the matches and the mask for training
        if self.training_cfg['output_flow_size'] is None:
            size_of_flow = [[h, w]]
            # creates a flow of the same load_size as the images
        else:
            size_of_flow = self.training_cfg['output_flow_size']

        if not isinstance(size_of_flow[0], list):
            # must be list of sizes
            size_of_flow = [size_of_flow]

        list_of_flow = []
        list_of_mask = []
        for i_size in size_of_flow:
            [h_size, w_size] = i_size
            # resize the keypoint accordingly
            points2D1 = kp_source.clone()
            points2D2 = kp_target.clone()
            points2D1[:, 0] *= float(w_size) / float(w)
            points2D1[:, 1] *= float(h_size) / float(h)
            points2D2[:, 0] *= float(w_size) / float(w)
            points2D2[:, 1] *= float(h_size) / float(h)

            # computes the flow
            flow, mask = self.keypoints_to_flow(points2D1, points2D2, h_size, w_size)
            mask = mask.bool() if float(torch.__version__[:3]) >= 1.1 else mask.byte()

            list_of_flow.append(flow)
            list_of_mask.append(mask)

        if len(size_of_flow) == 1:
            return source, target, list_of_flow[-1], list_of_mask[-1]

        return source, target, list_of_flow, list_of_mask

    def horizontal_flip_img(self, img, bbox, kp):
        tmp = bbox[0].clone()
        bbox[0] = img.shape[1] - bbox[2]
        bbox[2] = img.shape[1] - tmp
        kp[0] = img.shape[1] - kp[0]
        img = np.flip(img, 1)
        return img, bbox, kp

    def horizontal_flip(self, batch):
        tmp = batch['src_bbox'][0].clone()
        batch['src_bbox'][0] = batch['src_img'].size(2) - batch['src_bbox'][2]
        batch['src_bbox'][2] = batch['src_img'].size(2) - tmp

        tmp = batch['trg_bbox'][0].clone()
        batch['trg_bbox'][0] = batch['trg_img'].size(2) - batch['trg_bbox'][2]
        batch['trg_bbox'][2] = batch['trg_img'].size(2) - tmp

        batch['src_kps'][0][:batch['n_pts']] = batch['src_img'].size(2) - batch['src_kps'][0][:batch['n_pts']]
        batch['trg_kps'][0][:batch['n_pts']] = batch['trg_img'].size(2) - batch['trg_kps'][0][:batch['n_pts']]

        batch['src_img'] = np.flip(batch['src_img'], 1)
        batch['trg_img'] = np.flip(batch['trg_img'], 1)


class ImagePairDataset(Dataset):
    """
    From image pairs, retrieve image pairs or single images, without any ground-truth flow fields.
    """

    def __init__(self, dataset_csv_path, dataset_image_path, dataset_size=0, output_image_size=(240, 240),
                 two_views=True, source_image_transform=None, target_image_transform=None, random_crop=False):
        """
        Args:
            dataset_csv_path:
            dataset_image_path:
            dataset_size:
            output_image_size:
            two_views:
            source_image_transform:
            target_image_transform:
            random_crop:
        """

        if not isinstance(output_image_size, tuple):
            output_image_size = (output_image_size, output_image_size)
        self.random_crop = random_crop
        self.out_h, self.out_w = output_image_size
        self.train_data = pd.read_csv(dataset_csv_path)
        if dataset_size is not None and dataset_size != 0:
            dataset_size = min((dataset_size, len(self.train_data)))
            self.train_data = self.train_data.iloc[0:dataset_size, :]

        self.two_views = two_views
        if self.two_views:
            self.img_A_names = self.train_data.iloc[:, 0]
            self.img_B_names = self.train_data.iloc[:, 1]
            self.set = self.train_data.iloc[:, 2].values
            self.flip = self.train_data.iloc[:, 3].values.astype('int')
        else:
            self.images = self.train_data.iloc[:, 0]
            self.images.append(self.train_data.iloc[:, 1])

        self.dataset_image_path = dataset_image_path
        self.transform_source = source_image_transform
        self.transform_target = target_image_transform

    def __len__(self):
        if self.two_views:
            return len(self.img_A_names)
        else:
            return len(self.images)

    def __getitem__(self, idx):
        if self.two_views:
            # get pre-processed images
            image_A, im_size_A = self.get_image(self.img_A_names, idx, self.flip[idx])
            image_B, im_size_B = self.get_image(self.img_B_names, idx, self.flip[idx])

            if self.transform_source:
                image_A = self.transform_source(image_A)

            if self.transform_target:
                image_B = self.transform_target(image_B)

            sample = {'source_image': image_A, 'target_image': image_B, 'source_im_size': im_size_A,
                      'target_im_size': im_size_B, 'sparse': False}
        else:
            image, im_size = self.get_image(self.images, idx, flip=random.random() < 0.5)
            sample = {'image': image, 'image_size': im_size}
        return sample

    def get_image(self, img_name_list, idx, flip):
        img_name = os.path.join(self.dataset_image_path, img_name_list.iloc[idx])
        image = imageio.imread(img_name)

        # if grayscale convert to 3-channel image
        if image.ndim == 2:
            image = np.repeat(np.expand_dims(image, 2), axis=2, repeats=3)

        # do random crop
        if self.random_crop:
            h, w, c = image.shape
            top = np.random.randint(h / 4)
            bottom = int(3 * h / 4 + np.random.randint(h / 4))
            left = np.random.randint(w / 4)
            right = int(3 * w / 4 + np.random.randint(w / 4))
            image = image[top:bottom, left:right, :]

        # flip horizontally if needed
        if flip:
            image = np.flip(image, 1)

        # get image load_size
        im_size = np.asarray(image.shape)

        image = cv2.resize(image, dsize=(self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)

        return image, im_size
