import os
import random
import torch
import cv2
from torch.utils.data import Dataset
from packaging import version
import pandas as pd
import numpy as np


from utils_flow.img_processing_utils import pad_to_same_shape, pad_to_size, resize_keeping_aspect_ratio


def resize(img, kps, bbox, size=(256, 256)):
    """
    Args:
        img: HxWx3, numpy array
        kps: Nx2, torch Tensor
        size:

    Returns:

    """
    h, w = img.shape[:2]
    resized_img = cv2.resize(img, dsize=(size[1], size[0]), interpolation=cv2.INTER_LINEAR)

    kps = kps.clone()
    resized_kps = torch.zeros_like(kps, dtype=torch.float)
    resized_kps[:, 0] = kps[:, 0] * size[1] / w
    resized_kps[:, 1] = kps[:, 1] * size[0] / h

    resized_bbx = bbox.clone()
    resized_bbx[0::2] = resized_bbx[0::2].float() * size[1] / w
    resized_bbx[1::2] = resized_bbx[1::2].float() * size[0] / h
    return resized_img, resized_kps, resized_bbx


def random_crop(img, kps, bbox, size=(256, 256), p=0.5):
    """
    Args:
        img: HxWx3, numpy array
        kps: Nx2, torch Tensor
        size:

    Returns:

    """
    if random.uniform(0, 1) > p:
        return resize(img, kps, bbox, size)
    h, w = img.shape[:2]

    left = random.randint(0, max(0, bbox[0]))
    top = random.randint(0, max(0, bbox[1]))
    height = random.randint(min(bbox[3], h), h) - top
    width = random.randint(min(bbox[2], w), w) - left

    try:
        resized_img = img[top: top + height, left: left + width]
        resized_img = cv2.resize(resized_img, dsize=(size[1], size[0]), interpolation=cv2.INTER_LINEAR)

        kps = kps.clone()
        resized_kps = torch.zeros_like(kps, dtype=torch.float)
        resized_kps[:, 0] = (kps[:, 0] - left) * (size[1] / width)
        resized_kps[:, 1] = (kps[:, 1] - top) * (size[0] / height)
        resized_kps[:, 0] = torch.clamp(resized_kps[:, 0], 0, max=size[1] - 1)
        resized_kps[:, 1] = torch.clamp(resized_kps[:, 1], 0, max=size[0] - 1)

        resized_bbx = bbox.clone()
        resized_bbx[0::2] = (bbox[0::2] - left) * (size[1] / width)
        resized_bbx[1::2] = (bbox[1::2] - top) * (size[0] / height)
    except:
        resized_img = img
        resized_kps = kps
        resized_bbx = bbox
    return resized_img, resized_kps, resized_bbx


class SemanticKeypointsDataset(Dataset):
    """Parent class of PFPascal, PFWillow, Caltech, and SPair"""

    def __init__(self, benchmark, root, thres, split, source_image_transform=None,
                 target_image_transform=None, flow_transform=None, training_cfg=None, output_image_size=None,
                 ):
        """CorrespondenceDataset constructor """
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
            'proba_of_crop': 0.5,
            'augment_with_flip': False,
            'proba_of_image_flip': 0.,
            'proba_of_batch_flip': 0.5,
            'crop_size': [400, 400],
            'output_image_size': [400, 400],
            'pad_to_same_shape': False,
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

        # if need to resize the images, even for testing
        if output_image_size is not None:
            if not isinstance(output_image_size, tuple):
                output_image_size = (output_image_size, output_image_size)
        self.output_image_size = output_image_size

    def __len__(self):
        """Returns the number of pairs"""
        return len(self.train_data)

    def __getitem__(self, idx):
        """Constructs and return a batch"""

        # Image names
        batch = dict()
        batch['sparse'] = True
        batch['src_imname'] = self.src_imnames[idx]
        batch['trg_imname'] = self.trg_imnames[idx]

        # Class of instances in the images
        batch['category_id'] = self.cls_ids[idx]
        batch['category'] = self.cls[batch['category_id']]

        # Image as numpy (original height, original width)
        src_numpy = self.get_image(self.src_imnames, idx)
        trg_numpy = self.get_image(self.trg_imnames, idx)
        batch['src_imsize_ori'] = np.array(src_numpy.shape[:2])  # h,w

        batch['trg_imsize_ori'] = np.array(trg_numpy.shape[:2])

        src_numpy = self.resize_image(src_numpy.copy())
        trg_numpy = self.resize_image(trg_numpy.copy())
        batch['source_image_size'] = np.array(src_numpy.shape[:2])  # h,w
        batch['target_image_size'] = np.array(trg_numpy.shape[:2])

        batch['source_image'] = src_numpy
        batch['target_image'] = trg_numpy

        # Key-points (re-scaled)
        batch['source_kps'], num_pts = self.get_points(self.src_kps, idx, batch['src_imsize_ori'])  # Nx2
        batch['target_kps'], _ = self.get_points(self.trg_kps, idx, batch['trg_imsize_ori'])  # Nx2
        batch['n_pts'] = torch.tensor(num_pts)

        # The number of pairs in training split
        batch['datalen'] = len(self.train_data)

        return batch

    def get_image(self, imnames, idx):
        """Reads numpy image from path"""
        path = os.path.join(self.img_path, imnames[idx])
        return cv2.imread(path, 1)[:, :, ::-1]

    def resize_image(self, image):
        if self.output_image_size is not None:
            image = cv2.resize(image, self.output_image_size[::-1])
        return image

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

    def get_points(self, pts_list, idx, org_imsize):
        """Returns key-points of an image"""
        # already tensors here, return is torch tensor 2xN
        point_coords = pts_list[idx]
        xy, n_pts = pts_list[idx].size()
        pad_pts = torch.zeros((xy, self.max_pts - n_pts)) - 1

        if self.output_image_size is not None:
            # resize
            point_coords[0] *= self.output_image_size[1] / org_imsize[1]  # w
            point_coords[1] *= self.output_image_size[0] / org_imsize[0]  # h

        point_coords = torch.cat([point_coords, pad_pts], dim=1)  # so we can have larger batch size
        return torch.t(point_coords), n_pts  # Nx2, N

    @staticmethod
    def keypoints_to_flow(point_source_coords, point_target_coords, h_size, w_size):
        flow = torch.zeros((h_size, w_size, 2)).float()
        mask = torch.zeros((h_size, w_size)).byte()

        # computes the flow
        # if outside of the image, clamp
        point_target_coords[:, 0] = torch.clamp(point_target_coords[:, 0], 0, w_size - 1)
        point_target_coords[:, 1] = torch.clamp(point_target_coords[:, 1], 0, h_size - 1)

        flow[torch.round(point_target_coords[:, 1]).long(), torch.round(point_target_coords[:, 0]).long()] = \
            point_source_coords - point_target_coords
        mask[torch.round(point_target_coords[:, 1]).long(), torch.round(point_target_coords[:, 0]).long()] = 1
        return flow, mask

    def recover_image_pair_for_training(self, batch):

        source = batch['source_image']
        target = batch['target_image']
        kp_source = batch['source_kps'].clone()
        kp_target = batch['target_kps'].clone()
        n_pts = batch['n_pts']
        if self.training_cfg['pad_to_same_shape']:
            # either pad to same shape
            source, target = pad_to_same_shape(source, target)

        if self.training_cfg['output_image_size'] is not None:
            if isinstance(self.training_cfg['output_image_size'], list):
                # resize to a fixed size and rescale the keypoints accordingly
                h1, w1 = source.shape[:2]
                source = cv2.resize(source, (self.training_cfg['output_image_size'][1],
                                             self.training_cfg['output_image_size'][0]))
                kp_source[:n_pts, 0] *= float(self.training_cfg['output_image_size'][1]) / float(w1)
                kp_source[:n_pts, 1] *= float(self.training_cfg['output_image_size'][0]) / float(h1)

                h2, w2 = target.shape[:2]
                target = cv2.resize(target, (self.training_cfg['output_image_size'][1],
                                             self.training_cfg['output_image_size'][0]))
                kp_target[:n_pts, 0] *= float(self.training_cfg['output_image_size'][1]) / float(w2)
                kp_target[:n_pts, 1] *= float(self.training_cfg['output_image_size'][0]) / float(h2)
            else:
                # rescale both images so that the largest dimension is equal to the desired size of image and
                # then pad to obtain size 256x256 or whatever desired size. and change keypoints accordingly
                source, ratio_1 = resize_keeping_aspect_ratio(source, self.training_cfg['output_image_size'])
                source = pad_to_size(source, self.training_cfg['output_image_size'])
                kp_source[:n_pts] *= ratio_1

                target, ratio_2 = resize_keeping_aspect_ratio(target, self.training_cfg['output_image_size'])
                target = pad_to_size(target, self.training_cfg['output_image_size'])
                kp_target[:n_pts] *= ratio_2

        h, w, _ = target.shape
        # create the flow field from the matches and the mask for training
        if self.training_cfg['output_flow_size'] is None:
            size_of_flow = [[h, w]]
            # creates a flow of the same size as the images
        else:
            size_of_flow = self.training_cfg['output_flow_size']

        if not isinstance(size_of_flow[0], list):
            # must be a list of sizes
            size_of_flow = [size_of_flow]

        list_of_flow = []
        list_of_mask = []
        for i_size in size_of_flow:
            [h_size, w_size] = i_size
            # resize the keypoint accordingly, only use the good keypoint (not the padded ones)
            points2D1 = kp_source.clone()[:batch['n_pts']]
            points2D2 = kp_target.clone()[:batch['n_pts']]
            points2D1[:, 0] *= float(w_size) / float(w)
            points2D1[:, 1] *= float(h_size) / float(h)
            points2D2[:, 0] *= float(w_size) / float(w)
            points2D2[:, 1] *= float(h_size) / float(h)

            # computes the flow
            flow, mask = self.keypoints_to_flow(points2D1, points2D2, h_size, w_size)
            mask = mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask.byte()

            list_of_flow.append(flow)
            list_of_mask.append(mask)

        batch['source_image'] = source
        batch['target_image'] = target
        batch['source_image_size'] = np.array(source.shape[:2])
        batch['target_image_size'] = np.array(target.shape[:2])
        batch['source_kps'] = kp_source
        batch['target_kps'] = kp_target
        if len(size_of_flow) == 1:
            batch['correspondence_mask'] = list_of_mask[-1]
            batch['flow_map'] = list_of_flow[-1]
        else:
            batch['correspondence_mask'] = list_of_mask
            batch['flow_map'] = list_of_flow
        return batch

    @staticmethod
    def horizontal_flip_img(img, bbox, kp):
        tmp = bbox[0].clone()
        bbox[0] = img.shape[1] - bbox[2]
        bbox[2] = img.shape[1] - tmp
        kp = kp.clone()
        kp[:, 0] = img.shape[1] - kp[:, 0]
        img = np.flip(img, 1)
        return img, bbox, kp

    def horizontal_flip(self, batch):
        tmp = batch['src_bbox'][0].clone()
        batch['src_bbox'][0] = batch['source_image'].shape[1] - batch['src_bbox'][2]
        batch['src_bbox'][2] = batch['source_image'].shape[1] - tmp

        tmp = batch['trg_bbox'][0].clone()
        batch['trg_bbox'][0] = batch['target_image'].shape[1] - batch['trg_bbox'][2]
        batch['trg_bbox'][2] = batch['target_image'].shape[1] - tmp

        batch['source_kps'][:batch['n_pts'], 0] = batch['source_image'].shape[1] - batch['source_kps'][:batch['n_pts'], 0]
        batch['target_kps'][:batch['n_pts'], 0] = batch['target_image'].shape[1] - batch['target_kps'][:batch['n_pts'], 0]

        batch['source_image'] = np.flip(batch['source_image'], 1)
        batch['target_image'] = np.flip(batch['target_image'], 1)


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

        self.dataset_image_path = os.path.dirname(dataset_image_path)  # because the csv file starts from the folder
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
        image = cv2.imread(img_name)[:, :, ::-1]

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

        # get image size
        im_size = np.asarray(image.shape)

        image = cv2.resize(image, dsize=(self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)

        return image, im_size
