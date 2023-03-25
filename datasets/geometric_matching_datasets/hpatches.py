from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
from os import path as osp
from packaging import version
import torch


from utils_flow.flow_and_mapping_operations import convert_mapping_to_flow


def HPatchesdataset(root, csv_path, image_transform=None, flow_transform=None, co_transform=None,
                    use_original_size=False, get_mapping=False, image_size=None, **kwargs):
    """
    Builds the dataset of HPatches image pairs and corresponding ground-truth flow fields. T
    Args:
        root: path to root folder
        csv_path: path to csv file with ground-truth data informations
        image_transform: image transformations to apply to source and target images
        flow_transform: flow transformations to apply to ground-truth flow fields
        co_transform: transformations to apply to both images and ground-truth flow fields
        use_original_size: load images and flow at original load_size ?
        get_mapping: bool get mapping ?
        image_size: image load_size used for evaluation, will be ignored if original_size is True.
                    default=(240, 240) as in DGC-Net

    Returns:
        train_dataset: None here
        test_dataset

    """
    if image_size is not None:
        test_dataset = HPatchesDataset(root, csv_path, image_transform, flow_transform, co_transform,
                                       use_original_size=use_original_size, get_mapping=get_mapping, image_size=image_size)
    else:
        test_dataset = HPatchesDataset(root, csv_path, image_transform, flow_transform, co_transform,
                                       use_original_size=use_original_size, get_mapping=get_mapping)
    return None, test_dataset


class HPatchesDataset(Dataset):
    """HPatches datasets (for evaluation)"""
    def __init__(self, root, path_list, image_transform, flow_transform, co_transform, use_original_size=False,
                 get_mapping=False, image_size=(240, 240)):
        """
        Args:
            root: root containing image and flow folders
            path_list: path to csv file with ground-truth data information
            image_transform: image transformations to apply to source and target images
            flow_transform: flow transformations to apply to ground-truth flow fields
            co_transform: transformations to apply to both images and ground-truth flow fields
            use_original_size: load images and flow at original size ?
            get_mapping: bool get mapping ?
            image_size: image size used for evaluation, will be ignored if origical_size is True.
                        default=(240, 240) as in DGC-Net
        Output in __getitem__:
                source_image
                target_image
                correspondence_mask: visible and valid correspondences
                source_image_size
                homography: homography corresponding to the view-point changes between the image pair
            if get_mapping:
                mapping: pixel correspondence map in target coordinate system, relating target to source image
            else:
                flow_map: flow fields in target coordinate system, relating target to source image
        """

        self.root = root
        self.path_list = path_list
        self.df = pd.read_csv(path_list)
        self.transform = image_transform
        self.target_transform = flow_transform
        self.co_transform = co_transform
        self.image_size = image_size
        self.original_size=use_original_size
        self.get_mapping = get_mapping

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Args:
            idx: index

        Returns: Dictionary with fieldnames:
                source_image
                target_image
                correspondence_mask: visible and valid correspondences
                source_image_size
                homography: homography corresponding to the view-point changes between the image pair
            if get_mapping:
                mapping: pixel correspondence map in target coordinate system, relating target to source image
            else:
                flow_map: flow fields in target coordinate system, relating target to source image
        """
        data = self.df.iloc[idx]
        obj = str(data.obj)

        obj_dir = osp.join('{}'.format(self.root), obj)
        im1_id, im2_id = str(data.im1), str(data.im2)

        h_ref_orig, w_ref_orig = data.Him.astype('int'), data.Wim.astype('int')
        h_trg_orig, w_trg_orig, _ = \
            cv2.imread(osp.join(obj_dir, im2_id + '.ppm'), -1).shape
        if self.original_size:
            h_scale, w_scale = h_trg_orig, w_trg_orig
        else:
            h_scale, w_scale = self.image_size[0], self.image_size[1]

        H = data[5:].astype('double').values.reshape((3, 3))

        # As gt homography is calculated for (h_orig, w_orig) images,
        # we need to
        # map it to (h_scale, w_scale), that is 240x240
        # H_scale = S * H * inv(S)

        S1 = np.array([[w_scale / w_ref_orig, 0, 0],
                       [0, h_scale / h_ref_orig, 0],
                       [0, 0, 1]])
        S2 = np.array([[w_scale / w_trg_orig, 0, 0],
                       [0, h_scale / h_trg_orig, 0],
                       [0, 0, 1]])

        H_scale = np.dot(np.dot(S2, H), np.linalg.inv(S1))

        # inverse homography matrix
        Hinv = np.linalg.inv(H_scale)

        # estimate the grid
        X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                           np.linspace(0, h_scale - 1, h_scale))
        X, Y = X.flatten(), Y.flatten()

        # create matrix representation
        XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T

        # multiply Hinv to XYhom to find the warped grid
        XYwarpHom = np.dot(Hinv, XYhom)

        # vector representation
        XwarpHom = torch.from_numpy(XYwarpHom[0, :]).float()
        YwarpHom = torch.from_numpy(XYwarpHom[1, :]).float()
        ZwarpHom = torch.from_numpy(XYwarpHom[2, :]).float()

        Xwarp=XwarpHom / (ZwarpHom + 1e-8)
        Ywarp=YwarpHom / (ZwarpHom + 1e-8)
        # and now the grid
        grid_gt = torch.stack([Xwarp.view(h_scale, w_scale),
                               Ywarp.view(h_scale, w_scale)], dim=-1)

        # mask
        mask = grid_gt[:, :, 0].ge(0) & grid_gt[:, :, 0].le(w_scale-1) & \
               grid_gt[:, :, 1].ge(0) & grid_gt[:, :, 1].le(h_scale-1)

        img1 = \
            cv2.resize(cv2.imread(osp.join(self.root,
                                           obj,
                                           im1_id + '.ppm'), -1),
                       (h_scale, w_scale))
        img2 = \
            cv2.resize(cv2.imread(osp.join(self.root,
                                           obj,
                                           im2_id + '.ppm'), -1),
                       (h_scale, w_scale))
        _, _, ch = img1.shape
        if ch == 3:
            img1_tmp = cv2.imread(osp.join(self.root,
                                           obj,
                                           im1_id + '.ppm'), -1)
            img2_tmp = cv2.imread(osp.join(self.root,
                                           obj,
                                           im2_id + '.ppm'), -1)
            img1 = cv2.cvtColor(cv2.resize(img1_tmp,
                                           (w_scale, h_scale)),
                                cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(cv2.resize(img2_tmp,
                                           (w_scale,h_scale)),
                                cv2.COLOR_BGR2RGB)

        if self.get_mapping:
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            if self.target_transform is not None:
                mapping = self.target_transform(grid_gt.detach())
            return {'source_image': img1,
                    'target_image': img2,
                    'mapping': mapping,  # attention it is not normalised !!
                    'correspondence_mask': mask.bool(),  # mask_x and mask_y
                    'source_image_size': img1.shape,
                    'homography': H_scale
                    }
        else:
            target = convert_mapping_to_flow(grid_gt.detach().numpy(), output_channel_first=False)
            inputs = [img1, img2]

            # global transforms
            if self.co_transform is not None:
                inputs, target = self.co_transform(inputs, target)
            # transforms here will always contain conversion to tensor (then channel is before)
            if self.transform is not None:
                inputs[0] = self.transform(inputs[0])
                inputs[1] = self.transform(inputs[1])
            if self.target_transform is not None:
                target = self.target_transform(target)

            return {'source_image': inputs[0],
                    'target_image': inputs[1],
                    'flow_map': target,
                    'correspondence_mask': mask.bool() if version.parse(torch.__version__) >= version.parse("1.1")
                        else mask.byte(),
                    'source_image_size': img1.shape,
                    'homography': H_scale
                    }

