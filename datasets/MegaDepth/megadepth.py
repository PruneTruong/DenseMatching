import os
import pickle
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from utils_flow.util import pad_to_same_shape, pad_to_size, resize_keeping_aspect_ratio
import cv2
import random
from datasets.util import define_mask_zero_borders


class MegaDepthDataset(Dataset):
    """MegaDepth dataset. Retrieves either pairs of matching images and their corresponding ground-truth flow
    (that is actually sparse) or single images. """
    def __init__(self, root, cfg, load_pre_saved_dataset, split='train', source_image_transform=None,
                 target_image_transform=None, flow_transform=None, co_transform=None, compute_mask_zero_borders=False,
                 pickle_information=None):
        """

        Args:
            root: root directory
            cfg: config (dictionary)
            load_pre_saved_dataset: load pre computed image pairs using pickle information ?
            split: 'train' or 'val'
            source_image_transform: image transformations to apply to source images
            target_image_transform: image transformations to apply to target images
            flow_transform: flow transformations to apply to ground-truth flow fields
            co_transform: transforms to apply to both image pairs and corresponding flow field
            compute_mask_zero_borders: output mask of zero borders ?
            pickle_information: path to pickle information

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
            'val_split': 'validation_scenes_MegaDepth.txt',
            'scene_info_path': '',
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

        self.output_image_size = self.cfg['output_image_size']
        self.pad_to_same_shape = self.cfg['pad_to_same_shape']
        self.output_flow_size = self.cfg['output_flow_size']

        # processing of final images
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.flow_transform = flow_transform
        self.co_transform = co_transform

        self.compute_mask_zero_borders = compute_mask_zero_borders

        self.load_pre_saved_dataset = load_pre_saved_dataset

        if not self.two_views and self.load_pre_saved_dataset:
            raise ValueError('there is no pre saved dataset for single view')

        if self.load_pre_saved_dataset:
            if not os.path.exists(pickle_information):
                raise ValueError('the path to info that you provided does not exist: {}'.format(pickle_information))
            with open(pickle_information, "rb") as fp:  # Unpickling
                self.items = pickle.load(fp)
        else:
            self.sample_new_items(self.cfg['seed'])
        print('MegaDepth: {} dataset comprises {} image pairs'.format(self.split, self.__len__()))

    def build_pair_dataset_and_save_info_to_disk(self):
        self.items = []
        np.random.seed(self.cfg.seed)
        print('Building the {} dataset...'.format(self.split))

        for scene in tqdm(self.scenes, total=len(self.scenes)):
            # for each scene
            scene_info_path = os.path.join(
                self.scene_info_path, '%s.0.npz' % scene
            )
            if not os.path.exists(scene_info_path):
                continue

            scene_info = np.load(scene_info_path, allow_pickle=True)
            overlap_matrix = scene_info['overlap_matrix']
            scale_ratio_matrix = scene_info['scale_ratio_matrix']

            # select valid pairs within the scene that have proper overlap and scale ratio
            valid = np.logical_and(
                np.logical_and(
                    overlap_matrix >= self.cfg['min_overlap_ratio'],
                    overlap_matrix <= self.cfg['max_overlap_ratio']
                ),
                scale_ratio_matrix <= self.cfg['max_scale_ratio']
            )

            pairs = np.vstack(np.where(valid)) # valid pairs that correspond to criteria
            try:
                num = self.cfg[self.split + '_num_per_scene']
                if num is None:
                    selected_ids = range(pairs.shape[1])
                else:
                    selected_ids = np.random.choice(
                        pairs.shape[1], num
                    )
                # select some pairs if you dont want to have all pairs in the scene

                # get info about the scene images
                image_paths = scene_info['image_paths']
                # depth_paths = scene_info['depth_paths']
                points3D_id_to_2D = scene_info['points3D_id_to_2D']
                # points3D_id_to_ndepth = scene_info['points3D_id_to_ndepth']
                # intrinsics = scene_info['intrinsics']
                # poses = scene_info['poses']
            except:
                # there are no pairs that correspond to the criteria
                continue

            for pair_idx in selected_ids:
                idx1 = pairs[0, pair_idx]
                idx2 = pairs[1, pair_idx]
                matches = np.array(list(
                    points3D_id_to_2D[idx1].keys() &
                    points3D_id_to_2D[idx2].keys()
                ))

                # Scale filtering
                # matches_nd1 = np.array([points3D_id_to_ndepth[idx1][match] for match in matches])
                # matches_nd2 = np.array([points3D_id_to_ndepth[idx2][match] for match in matches])
                # scale_ratio = np.maximum(matches_nd1 / matches_nd2, matches_nd2 / matches_nd1)
                # matches = matches[np.where(scale_ratio <= self.cfg['max_scale_ratio'])[0]]

                # obtain 2D coordinate for all matches between the pair
                point2D1 = [np.array(points3D_id_to_2D[idx1][match], dtype=np.float32).reshape(1, 2) for
                            match in matches] # check shape
                point2D2 = [np.array(points3D_id_to_2D[idx2][match], dtype=np.float32).reshape(1, 2) for
                            match in matches]

                # nd1 = np.array([points3D_id_to_ndepth[idx1][match] for match in matches])
                # nd2 = np.array([points3D_id_to_ndepth[idx2][match] for match in matches])
                # put all the info in the bundle to create the flow for this image pair
                image_pair_bundle = ({
                    'image_path1': image_paths[idx1],
                    # 'depth_path1': depth_paths[idx1],
                    # 'intrinsics1': intrinsics[idx1],
                    # 'pose1': poses[idx1],
                    'image_path2': image_paths[idx2],
                    # 'depth_path2': depth_paths[idx2],
                    # 'intrinsics2': intrinsics[idx2],
                    # 'pose2': poses[idx2],
                    '2d_matches_1': point2D1,
                    '2d_matches_2': point2D2
                })

                # save the image pair information to disk
                name_of_dataset = '{}_{}_to_{}'.format(self.split, self.cfg['min_overlap_ratio'],
                                                       self.cfg['max_overlap_ratio'])

                if not os.path.isdir(os.path.join(self.root, name_of_dataset)):
                    os.makedirs(os.path.join(self.root, name_of_dataset))
                name = os.path.join(name_of_dataset,
                                    'MegaDepth_{}_scene_{}_pair_idx_{}.txt'.format(self.split, scene, pair_idx))
                with open(name, "wb") as fp:  # Pickling
                    pickle.dump(image_pair_bundle, fp)

                # save information corresponding to this particular image pair pickle
                self.items.append({
                    'scene': scene,
                    'pair_idx': pair_idx,
                    'info_path': name })

        # save the list containing into to all image pair individual pickles
        name = os.path.join(self.root, 'MegaDepth_{}_{}_to_{}.txt'
                            .format(self.split, self.cfg['min_overlap_ratio'], self.cfg['max_overlap_ratio']))
        with open(name, "wb") as fp:  # Pickling
            pickle.dump(self.items, fp)
            print('Saved the built dataset at {}'.format(name))

    def sample_new_items(self, seed):
        print(f'Sampling new images or pairs with seed {seed}')
        self.images = {}
        if self.two_views:
            # self.depth, self.poses, self.intrinsics, self.points3D_id_to_2D = {}, {}, {}, {}
            self.points3D_id_to_2D = {}
        self.items = []

        # for scene in tqdm(self.scenes):
        for i, scene in enumerate(self.scenes):
            path = os.path.join(self.scene_info_path, '%s.0.npz' % scene)
            if not os.path.exists(path):
                print(f'Scene {scene} does not have an info file')
                continue
            info = np.load(path, allow_pickle=True)
            num = self.cfg[self.split + '_num_per_scene']

            valid = ((info['image_paths'] != None) & (info['depth_paths'] != None))
            self.images[scene] = info['image_paths'][valid]

            if self.two_views:
                self.points3D_id_to_2D[scene] = info['points3D_id_to_2D'][valid]
                # self.depth[scene] = info['depth_paths'][valid]
                # self.poses[scene] = info['poses'][valid]
                # self.intrinsics[scene] = info['intrinsics'][valid]

                mat = info['overlap_matrix'][valid][:, valid]
                pairs = (
                    (mat > self.cfg['min_overlap_ratio'])
                    & (mat <= self.cfg['max_overlap_ratio']))
                pairs = np.stack(np.where(pairs), -1)
                if len(pairs) > num:
                    selected = np.random.RandomState(seed).choice(
                        len(pairs), num, replace=False)
                    pairs = pairs[selected]

                pairs_one_direction = [(scene, i, j, mat[i, j]) for i, j in pairs]
                self.items.extend(pairs_one_direction)
                # pairs_other_direction = [(scene, j, i, mat[i, j]) for i, j in pairs]
                # self.items.extend(pairs_other_direction)
            else:
                ids = np.arange(len(self.images[scene]))
                if len(ids) > num:
                    ids = np.random.RandomState(seed).choice(
                        ids, num, replace=False)
                ids = [(scene, i) for i in ids]
                self.items.extend(ids)

        if self.two_views and self.cfg['sort_by_overlap']:
            self.items.sort(key=lambda i: i[-1], reverse=True)
        else:
            np.random.RandomState(seed).shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def _read_single_view(self, scene, idx):
        path = os.path.join(self.root, self.images[scene][idx])
        image = cv2.imread(path)
        if len(image.shape) != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = image[:, :, ::-1]  # go from BGR to RGB

        if self.output_image_size is not None:
            if isinstance(self.output_image_size, list):
                # resize to a fixed load_size and rescale the keypoints accordingly
                image = cv2.resize(image, (self.output_image_size[1], self.output_image_size[0]))

            else:
                # rescale both images so that the largest dimension is equal to the desired load_size of image and
                # then pad to obtain load_size 256x256 or whatever desired load_size. and change keypoints accordingly
                image, ratio_ = resize_keeping_aspect_ratio(image, self.output_image_size)
                image = pad_to_size(image, self.output_image_size)

        if self.source_image_transform is not None:
            image = self.source_image_transform(image)

        return {'image': image}

    def _read_pair_info(self, scene, idx1, idx2):
        image_paths = self.images[scene]
        points3D_id_to_2D = self.points3D_id_to_2D[scene]

        matches = np.array(list(points3D_id_to_2D[idx1].keys() & points3D_id_to_2D[idx2].keys()))

        # obtain 2D coordinate for all matches between the pair
        point2D1 = [np.array(points3D_id_to_2D[idx1][match], dtype=np.float32).reshape(1, 2) for
                    match in matches]
        point2D2 = [np.array(points3D_id_to_2D[idx2][match], dtype=np.float32).reshape(1, 2) for
                    match in matches]

        image_pair_bundle = ({
            'image_path1': image_paths[idx1],
            # 'depth_path1': depth_paths[idx1],
            # 'intrinsics1': intrinsics[idx1],
            # 'pose1': poses[idx1],
            'image_path2': image_paths[idx2],
            # 'depth_path2': depth_paths[idx2],
            # 'intrinsics2': intrinsics[idx2],
            # 'pose2': poses[idx2],
            '2d_matches_1': point2D1,
            '2d_matches_2': point2D2
        })

        return image_pair_bundle

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

        image1 = cv2.imread(image_path1)
        if len(image1.shape) != 3:
            image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
        else:
            image1 = image1[:, :, ::-1]  # go from BGR to RGB

        image2 = cv2.imread(image_path2)
        if len(image2.shape) != 3:
            image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)
        else:
            image2 = image2[:, :, ::-1]  # go from BGR to RGB

        if self.pad_to_same_shape:
            # either pad to same shape
            image1, image2 = pad_to_same_shape(image1, image2)

        if self.output_image_size is not None:
            if isinstance(self.output_image_size, list):
                # resize to a fixed load_size and rescale the keypoints accordingly
                h1, w1 = image1.shape[:2]
                image1 = cv2.resize(image1, (self.output_image_size[1], self.output_image_size[0]))
                points2D1_from_file[:, 0] *= float(self.output_image_size[1]) / float(w1)
                points2D1_from_file[:, 1] *= float(self.output_image_size[0]) / float(h1)

                h2, w2 = image2.shape[:2]
                image2 = cv2.resize(image2, (self.output_image_size[1], self.output_image_size[0]))
                points2D2_from_file[:, 0] *= float(self.output_image_size[1]) / float(w2)
                points2D2_from_file[:, 1] *= float(self.output_image_size[0]) / float(h2)
            else:
                # rescale both images so that the largest dimension is equal to the desired load_size of image and
                # then pad to obtain load_size 256x256 or whatever desired load_size. and change keypoints accordingly
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
            # creates a flow of the same load_size as the images
        else:
            size_of_flow = self.output_flow_size

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
            valid_h = np.logical_and(np.int32(np.round(points2D2[:, 1])) < h_size, points2D1[:, 1] < h_size)
            valid_w = np.logical_and(np.int32(np.round(points2D2[:, 0])) < w_size, points2D1[:, 0] < w_size)
            valid = valid_h * valid_w
            points2D2 = points2D2[valid]
            points2D1 = points2D1[valid]

            flow[np.int32(np.round(points2D2[:, 1])), np.int32(np.round(points2D2[:, 0]))] = points2D1 - points2D2
            mask[np.int32(np.round(points2D2[:, 1])), np.int32(np.round(points2D2[:, 0]))] = 1

            list_of_flow.append(flow)
            list_of_mask.append(mask)

        if len(size_of_flow) == 1:
            return image1, image2, list_of_flow[-1], list_of_mask[-1]

        return image1, image2, list_of_flow, list_of_mask

    def __getitem__(self, idx):
        """
        Args:
            idx

        Returns:
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
            if self.load_pre_saved_dataset:
                # obtain the image, the flow and the correspondence mask
                # image 1 is source image, image2 is target image
                old_path_with_info = self.items[idx]['info_path']
                # to remove
                relative_path = os.path.join(os.path.split(os.path.split(os.path.split(old_path_with_info)[0])[0])[1],
                                             os.path.split(os.path.split(old_path_with_info)[0])[1],
                                             os.path.split(old_path_with_info)[1])
                path_with_info = os.path.join(self.root + relative_path)

                with open(path_with_info, "rb") as fp:  # Unpickling
                    pair_metadata = pickle.load(fp)
            else:
                scene, idx0, idx1, overlap = self.items[idx]
                pair_metadata = self._read_pair_info(scene, idx0, idx1)

            image1, image2, target, mask = self.recover_pair(pair_metadata,
                                                             random.random() < self.cfg['exchange_images_with_proba'])

            if self.co_transform is not None:
                [image1, image2], target, mask = self.co_transform([image1, image2], target, mask)
            source_size = image1.shape

            if self.source_image_transform is not None:
                image1 = self.source_image_transform(image1)
            if self.target_image_transform is not None:
                image2 = self.target_image_transform(image2)
            if self.flow_transform is not None:
                if type(target) in [tuple, list]:
                    # flow field at different resolution
                    for i in range(len(target)):
                        target[i] = self.flow_transform(target[i])
                else:
                   target = self.flow_transform(target)

            if self.compute_mask_zero_borders:
                mask_valid = define_mask_zero_borders(image2)

            if type(mask) in [tuple, list]:
                # flow field at different resolution
                for i in range(len(target)):
                    mask[i] = torch.from_numpy(mask[i].astype(np.bool if float(torch.__version__[:3]) >= 1.1 else np.uint8))
            else:
                mask = torch.from_numpy(mask.astype(np.bool if float(torch.__version__[:3]) >= 1.1 else np.uint8))

            output = {'source_image': image1, 'target_image': image2, 'flow_map': target, 'correspondence_mask': mask,
                      'source_image_size': source_size, 'sparse': True}
            if self.compute_mask_zero_borders:
                output['mask_zero_borders'] = mask_valid.astype(np.bool) if float(torch.__version__[:3]) >= 1.1 \
                                                   else mask_valid.astype(np.uint8)

        else:
            # only retrieved a single image
            scene, idx = self.items[idx]
            output = self._read_single_view(scene, idx)
            output['scene'] = scene
        return output
