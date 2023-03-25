import os
from pathlib import Path
import numpy as np
import cv2
import random
from packaging import version
from torch.utils.data import Dataset
import torch
import time
import copy
import jpeg4py

from utils_flow.img_processing_utils import pad_to_same_shape, pad_to_size, resize_keeping_aspect_ratio
from utils_flow.img_processing_utils import define_mask_zero_borders


def valid_size(size):
    if not isinstance(size, (tuple, list)):
        size = (size, size)
    return size


class MegaDepthDataset(Dataset):
    """MegaDepth dataset. Retrieves either pairs of matching images and their corresponding ground-truth flow
    (that is actually sparse) or single images. """
    def __init__(self, root, cfg, split='train', source_image_transform=None,
                 target_image_transform=None, flow_transform=None, co_transform=None, compute_mask_zero_borders=False,
                 store_scene_info_in_memory=False):
        """
        Args:
            root: root directory
            cfg: config (dictionary)
            split: 'train' or 'val'
            source_image_transform: image transformations to apply to source images
            target_image_transform: image transformations to apply to target images
            flow_transform: flow transformations to apply to ground-truth flow fields
            co_transform: transforms to apply to both image pairs and corresponding flow field
            compute_mask_zero_borders: output mask of zero borders ?
            store_scene_info_in_memory: store all scene info in cpu memory? requires at least 50GB for training but
                                        sampling at each epoch is faster.

        Output in __getitem__:
            if self.two_views:
                source_image
                target_image
                flow_map: flow fields in flow coordinate system, relating flow to source image
                correspondence_mask: visible and valid correspondences
                source_image_size
                sparse: True

                if mask_zero_borders:
                    mask_zero_borders: bool tensor equal to 1 where the target image is not equal to 0, 0 otherwise

            else:
                image
                scene: id of the scene

        """

        default_conf = {
            'seed': 400,
            'train_split': 'train_scenes_MegaDepth.txt',
            'train_debug_split': 'train_debug_scenes_MegaDepth.txt',
            'val_split': 'validation_scenes_MegaDepth.txt',
            'scene_info_path': '',
            'train_debug_num_per_scene': 10,
            'train_num_per_scene': 100,
            'val_num_per_scene': 25,

            'min_overlap_ratio': 0.3,
            'max_overlap_ratio': 1.,
            'max_scale_ratio': np.inf,

            'two_views': True,
            'exchange_images_with_proba': 0.5,
            'sort_by_overlap': False,
            
            'output_image_size': [520, 520],
            'pad_to_same_shape': True, 
            'output_flow_size': [[520, 520], [256, 256]],
            }
        
        self.root = root
        self.cfg = default_conf
        self.cfg.update(cfg)

        self.scenes = []
        with open(Path(__file__).parent / self.cfg[split+'_split'], 'r') as f:
            self.scenes = f.read().split()

        self.scene_info_path = self.cfg['scene_info_path']
        
        self.two_views = self.cfg['two_views']
        self.split = split

        self.output_image_size = valid_size(self.cfg['output_image_size'])
        self.pad_to_same_shape = self.cfg['pad_to_same_shape']
        self.output_flow_size = self.cfg['output_flow_size']

        # processing of final images
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.flow_transform = flow_transform
        self.co_transform = co_transform

        self.compute_mask_zero_borders = compute_mask_zero_borders

        self.items = []
        self.store_scene_info_in_memory = store_scene_info_in_memory
        if not self.two_views:
            # if single view, always just store
            self.store_scene_info_in_memory = True

        if self.store_scene_info_in_memory:
            # it will take around 35GB, you need at least 50GB of cpu memory to train
            self.save_scene_info()
            self.sample_new_items(self.cfg['seed'])
        else:
            self.sample_new_items(self.cfg['seed'])
        print('MegaDepth: {} dataset comprises {} image pairs'.format(self.split, self.__len__()))

    def save_scene_info(self):
        print('MegaDepth {}: Storing info about scenes on memory...\nThis will take some time'.format(self.split))
        start = time.time()
        self.images = {}
        if self.two_views:
            # self.depth, self.poses, self.intrinsics, self.points3D_id_to_2D = {}, {}, {}, {}
            self.points3D_id_to_2D = {}
            self.pairs = {}

        # for scene in tqdm(self.scenes):
        for i, scene in enumerate(self.scenes):
            path = os.path.join(self.scene_info_path, '%s.0.npz' % scene)
            if not os.path.exists(path):
                print(f'Scene {scene} does not have an info file')
                continue
            info = np.load(path, allow_pickle=True)

            valid = ((info['image_paths'] != None) & (info['depth_paths'] != None))
            self.images[scene] = info['image_paths'][valid].copy()
            # self.depth[scene] = info['depth_paths'][valid]
            # self.poses[scene] = info['poses'][valid]
            # self.intrinsics[scene] = info['intrinsics'][valid]

            if self.two_views:
                self.points3D_id_to_2D[scene] = info['points3D_id_to_2D'][valid].copy()

                # write pairs that have a correct overlap ratio
                mat = info['overlap_matrix'][valid][:, valid]  # N_img x N_img where N_img is len(self.images[scene])
                pairs = (
                    (mat > self.cfg['min_overlap_ratio'])
                    & (mat <= self.cfg['max_overlap_ratio']))
                pairs = np.stack(np.where(pairs), -1)
                self.pairs[scene] = [(i, j, mat[i, j]) for i, j in pairs]

            info.close()
            del info
        total = time.time() - start
        print('Storing took {} s'.format(total))

    def sample_new_items(self, seed):
        print('MegaDepth {}: Sampling new images or pairs with seed {}. \nThis will take some time...'
              .format(self.split, seed))
        start_time = time.time()
        self.items = []

        num = self.cfg[self.split + '_num_per_scene']

        # for scene in tqdm(self.scenes):
        for i, scene in enumerate(self.scenes):
            path = os.path.join(self.scene_info_path, '%s.0.npz' % scene)
            if not os.path.exists(path):
                print(f'Scene {scene} does not have an info file')
                continue
            if self.two_views and self.store_scene_info_in_memory:
                # sampling is just accessing the pairs
                pairs = np.array(self.pairs[scene])
                if len(pairs) > num:
                    selected = np.random.RandomState(seed).choice(
                        len(pairs), num, replace=False)
                    pairs = pairs[selected]

                pairs_one_direction = [(scene, int(i), int(j), k) for i, j, k in pairs]
                self.items.extend(pairs_one_direction)
            elif self.two_views:
                # sample all infos from the scenes
                info = np.load(path, allow_pickle=True, mmap_mode='r')
                valid = ((info['image_paths'] != None) & (info['depth_paths'] != None))
                paths = info['image_paths'][valid]

                points3D_id_to_2D = info['points3D_id_to_2D'][valid]

                mat = info['overlap_matrix'][valid][:, valid]
                info.close()
                pairs = (
                        (mat > self.cfg['min_overlap_ratio'])
                        & (mat <= self.cfg['max_overlap_ratio']))
                pairs = np.stack(np.where(pairs), -1)
                if len(pairs) > num:
                    selected = np.random.RandomState(seed).choice(
                        len(pairs), num, replace=False)
                    pairs = pairs[selected]

                for pair_idx in range(len(pairs)):
                    idx1 = pairs[pair_idx, 0]
                    idx2 = pairs[pair_idx, 1]
                    matches = np.array(
                        list(points3D_id_to_2D[idx1].keys() & points3D_id_to_2D[idx2].keys()))

                    point2D1 = [np.array(points3D_id_to_2D[idx1][match], dtype=np.float32).reshape(1, 2) for
                                match in matches]

                    point2D2 = [np.array(points3D_id_to_2D[idx2][match], dtype=np.float32).reshape(1, 2) for
                                match in matches]

                    image_pair_bundle = {
                        'image_path1': paths[idx1],
                        # 'depth_path1': depth_paths[idx1],
                        # 'intrinsics1': intrinsics[idx1],
                        # 'pose1': poses[idx1],
                        'image_path2': paths[idx2],
                        # 'depth_path2': depth_paths[idx2],
                        # 'intrinsics2': intrinsics[idx2],
                        # 'pose2': poses[idx2],
                        '2d_matches_1': point2D1.copy(),
                        '2d_matches_2': point2D2.copy()
                    }
                    self.items.append(image_pair_bundle)
            else:
                # single view, just sample new paths to imahes
                ids = np.arange(len(self.images[scene]))
                if len(ids) > num:
                    ids = np.random.RandomState(seed).choice(
                        ids, num, replace=False)
                ids = [(scene, i) for i in ids]
                self.items.extend(ids)

        if 'debug' in self.split:
            orig_copy = copy.deepcopy(self.items)
            for _ in range(10):
                self.items = self.items + copy.deepcopy(orig_copy)

        np.random.RandomState(seed).shuffle(self.items)
        end_time = time.time() - start_time
        print('Sampling took {} s. Sampled {} items'.format(end_time, len(self.items)))

    def __len__(self):
        return len(self.items)

    def _read_pair_info(self, scene, idx1, idx2):

        # when scene info are stored in memory
        matches = np.array(list(self.points3D_id_to_2D[scene][idx1].keys() & self.points3D_id_to_2D[scene][idx2].keys()))

        # obtain 2D coordinate for all matches between the pair
        point2D1 = [np.array(self.points3D_id_to_2D[scene][idx1][match], dtype=np.float32).reshape(1, 2) for
                    match in matches]
        point2D2 = [np.array(self.points3D_id_to_2D[scene][idx2][match], dtype=np.float32).reshape(1, 2) for
                    match in matches]

        image_pair_bundle = {
            'image_path1': self.images[scene][idx1],
            # 'depth_path1': depth_paths[idx1],
            # 'intrinsics1': intrinsics[idx1],
            # 'pose1': poses[idx1],
            'image_path2': self.images[scene][idx2],
            # 'depth_path2': depth_paths[idx2],
            # 'intrinsics2': intrinsics[idx2],
            # 'pose2': poses[idx2],
            '2d_matches_1': point2D1,
            '2d_matches_2': point2D2
        }

        return image_pair_bundle

    def _read_single_view(self, scene, idx):
        path = os.path.join(self.root, self.images[scene][idx])
        image = cv2.imread(path)
        if len(image.shape) != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = image[:, :, ::-1]  # go from BGR to RGB

        if self.output_image_size is not None:
            if isinstance(self.output_image_size, list):
                # resize to a fixed size and rescale the keypoints accordingly
                image = cv2.resize(image, (self.output_image_size[1], self.output_image_size[0]))

            else:
                # rescale both images so that the largest dimension is equal to the desired size of image and
                # then pad to obtain size 256x256 or whatever desired size. and change keypoints accordingly
                image, ratio_ = resize_keeping_aspect_ratio(image, self.output_image_size)
                image = pad_to_size(image, self.output_image_size)

        if self.source_image_transform is not None:
            image = self.source_image_transform(image)

        return {'image': image}

    @staticmethod
    def loader(path):
        if path.endswith('.jpg') or path.endswith('.jpeg'):
            try:
                image = jpeg4py.JPEG(path).decode()
            except:
                image = cv2.imread(path)
        else:
            image = cv2.imread(path)

        if len(image.shape) != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = image[:, :, ::-1]  # go from BGR to RGB
        return image

    def recover_pair(self, pair_metadata, exchange_images=True):
        if exchange_images:
            image_path1 = os.path.join(self.root, pair_metadata['image_path2'])
            image_path2 = os.path.join(self.root, pair_metadata['image_path1'])
            points2D1_from_file = np.concatenate(pair_metadata['2d_matches_2'], axis=0)  # Nx2
            points2D2_from_file = np.concatenate(pair_metadata['2d_matches_1'], axis=0)  # Nx2
        else:
            image_path1 = os.path.join(self.root, pair_metadata['image_path1'])
            image_path2 = os.path.join(self.root, pair_metadata['image_path2'])
            points2D1_from_file = np.concatenate(pair_metadata['2d_matches_1'], axis=0)  # Nx2
            points2D2_from_file = np.concatenate(pair_metadata['2d_matches_2'], axis=0)  # Nx2

        image1 = self.loader(image_path1)
        image2 = self.loader(image_path2)

        if self.pad_to_same_shape:
            # either pad to same shape
            image1, image2 = pad_to_same_shape(image1, image2)

        if self.output_image_size is not None:
            if isinstance(self.output_image_size, list):
                # resize to a fixed size and rescale the keypoints accordingly
                h1, w1 = image1.shape[:2]
                image1 = cv2.resize(image1, (self.output_image_size[1], self.output_image_size[0]))
                points2D1_from_file[:, 0] *= float(self.output_image_size[1]) / float(w1)
                points2D1_from_file[:, 1] *= float(self.output_image_size[0]) / float(h1)

                h2, w2 = image2.shape[:2]
                image2 = cv2.resize(image2, (self.output_image_size[1], self.output_image_size[0]))
                points2D2_from_file[:, 0] *= float(self.output_image_size[1]) / float(w2)
                points2D2_from_file[:, 1] *= float(self.output_image_size[0]) / float(h2)
            else:
                # rescale both images so that the largest dimension is equal to the desired size of image and
                # then pad to obtain size 256x256 or whatever desired size. and change keypoints accordingly
                image1, ratio_1 = resize_keeping_aspect_ratio(image1, self.output_image_size)
                image1 = pad_to_size(image1, self.output_image_size)
                points2D1_from_file *= ratio_1

                image2, ratio_2 = resize_keeping_aspect_ratio(image2, self.output_image_size)
                image2 = pad_to_size(image2, self.output_image_size)
                points2D2_from_file *= ratio_2

        h, w, _ = image2.shape
        # create the flow field from the matches and the mask for training
        if self.output_flow_size is None:
            size_of_flow = [[h, w]]
            # creates a flow of the same size as the images
        else:
            size_of_flow = self.output_flow_size

        if not isinstance(size_of_flow[0], list):
            size_of_flow = [size_of_flow]

        list_of_flow = []
        list_of_mask = []
        for i_size in size_of_flow:
            [h_size, w_size] = i_size
            flow = np.zeros((h_size, w_size, 2), dtype=np.float32)
            mask = np.zeros((h_size, w_size), dtype=np.uint8)

            # resize the keypoint accordingly
            points2D1 = points2D1_from_file.copy()
            points2D2 = points2D2_from_file.copy()
            points2D1[:, 0] *= float(w_size) / float(w)
            points2D1[:, 1] *= float(h_size) / float(h)

            points2D2[:, 0] *= float(w_size) / float(w)
            points2D2[:, 1] *= float(h_size) / float(h)

            # computes the flow
            points2D1_rounded = np.rint(points2D1).astype(np.int32)
            points2D2_rounded = np.rint(points2D2).astype(np.int32)

            valid_h = (points2D1_rounded[:, 1] >= 0) & (points2D1_rounded[:, 1] < h_size) & \
                      (points2D2_rounded[:, 1] >= 0) & (points2D2_rounded[:, 1] < h_size)
            valid_w = (points2D1_rounded[:, 0] >= 0) & (points2D1_rounded[:, 0] < w_size) & \
                      (points2D2_rounded[:, 0] >= 0) & (points2D2_rounded[:, 0] < w_size)

            valid = valid_h * valid_w
            points2D2 = points2D2[valid]
            points2D1 = points2D1[valid]
            points2D2_rounded = points2D2_rounded[valid]

            flow[points2D2_rounded[:, 1], points2D2_rounded[:, 0]] = points2D1 - points2D2
            mask[points2D2_rounded[:, 1], points2D2_rounded[:, 0]] = 1

            list_of_flow.append(flow)
            list_of_mask.append(mask)

        if len(size_of_flow) == 1:
            return image1, image2, list_of_flow[-1], list_of_mask[-1]

        return image1, image2, list_of_flow, list_of_mask

    def __getitem__(self, idx):
        """
        Args:
            idx

        Returns: Dictionary with fieldnames:
            if self.two_views:
                source_image
                target_image
                flow_map: flow fields in target coordinate system, relating target to source image
                correspondence_mask: visible and valid correspondences
                source_image_size
                sparse: True

                if mask_zero_borders:
                    mask_zero_borders: bool tensor equal to 1 where the target image is not equal to 0, 0 otherwise

            else:
                image
                scene: id of the scene
        """
        if self.two_views:
            if self.store_scene_info_in_memory:
                scene, idx1, idx2, overlap = self.items[idx]
                pair_metadata = self._read_pair_info(scene, idx1, idx2)
            else:
                pair_metadata = self.items[idx]

            source, target, flow, mask = self.recover_pair(pair_metadata,
                                                           random.random() < self.cfg['exchange_images_with_proba'])

            if self.co_transform is not None:
                [source, target], flow, mask = self.co_transform([source, target], flow, mask)
            source_size = source.shape

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

            if self.compute_mask_zero_borders:
                mask_valid = define_mask_zero_borders(target)

            if type(mask) in [tuple, list]:
                # flow field at different resolution
                for i in range(len(flow)):
                    mask[i] = torch.from_numpy(mask[i].astype(np.bool if version.parse(torch.__version__) >= version.parse("1.1") else np.uint8))
            else:
                mask = torch.from_numpy(mask.astype(np.bool if version.parse(torch.__version__) >= version.parse("1.1") else np.uint8))

            output = {'source_image': source, 'target_image': target, 'flow_map': flow, 'correspondence_mask': mask,
                      'source_image_size': source_size, 'sparse': True}
            if self.compute_mask_zero_borders:
                output['mask_zero_borders'] = mask_valid.astype(np.bool) if version.parse(torch.__version__) >= version.parse("1.1") \
                                                   else mask_valid.astype(np.uint8)

        else:
            # only retrieved a single image
            scene, idx = self.items[idx]
            output = self._read_single_view(scene, idx)
            output['scene'] = scene
        return output
