import torch.utils.data as data
import os
import os.path
import cv2
import numpy as np
import torch
import jpeg4py
from packaging import version


from utils_flow.flow_and_mapping_operations import get_gt_correspondence_mask, convert_flow_to_mapping
from utils_data.io import load_flo
from utils_flow.img_processing_utils import define_mask_zero_borders


def default_loader(root, path_imgs, path_flo):
    imgs = [os.path.join(root, path) for path in path_imgs]
    flo = os.path.join(root, path_flo)

    if imgs[0].endswith('.jpg') or imgs[0].endswith('.jpeg'):
        try:
            return [jpeg4py.JPEG(img).decode().astype(np.uint8) for img in imgs], load_flo(flo)
        except:
            return [cv2.imread(img, -1)[:, :, ::-1].astype(np.uint8) for img in imgs], load_flo(flo)
    else:
        return [cv2.imread(img, -1)[:, :, ::-1].astype(np.uint8) for img in imgs], load_flo(flo)


class ListDataset(data.Dataset):
    """General Dataset creation class"""
    def __init__(self, root, path_list, source_image_transform=None, target_image_transform=None, flow_transform=None,
                 co_transform=None, loader=default_loader, load_valid_mask=False, load_size=False,
                 load_occlusion_mask=False, get_mapping=False, compute_mask_zero_borders=False):
        """

        Args:
            root: root directory containing image pairs and flow folders
            path_list: path to csv files with ground-truth information
            source_image_transform: image transformations to apply to source images
            target_image_transform: image transformations to apply to target images
            flow_transform: flow transformations to apply to ground-truth flow fields
            co_transform: transforms to apply to both image pairs and corresponding flow field
            loader: image and flow loader type
            load_valid_mask: is the loader outputting a valid mask ?
            load_size: is the loader outputting size of original source image ?
            load_occlusion_mask: is the loader outputting a ground-truth occlusion mask ?
            get_mapping: get mapping ?
            compute_mask_zero_borders: output mask of zero borders ?
        Output in __getitem__:
            source_image
            target_image
            correspondence_mask: visible and valid correspondences
            source_image_size
            sparse: False (only dense outputs here)

            if load_occlusion_mask:
                load_occlusion_mask: ground-truth occlusion mask, bool tensor equal to 1 where the pixel in the target
                                image is occluded in the source image, 0 otherwise

            if mask_zero_borders:
                mask_zero_borders: bool tensor equal to 1 where the target image is not equal to 0, 0 otherwise

            if get_mapping:
                mapping: pixel correspondence map in target coordinate system, relating target to source image
            else:
                flow_map: flow fields in target coordinate system, relating target to source image
        """
        self.root = root
        self.path_list = path_list
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.flow_transform = flow_transform
        self.co_transform = co_transform
        self.loader = loader
        self.load_valid_mask = load_valid_mask
        self.load_size = load_size
        self.load_occlusion_mask = load_occlusion_mask
        self.get_mapping = get_mapping
        self.mask_zero_borders = compute_mask_zero_borders

    def __getitem__(self, index):
        """
        Args:
            index:

        Returns: dictionary with fieldnames
            source_image
            target_image
            correspondence_mask: visible and valid correspondences
            source_image_size
            sparse: False (only dense outputs here)

            if load_occlusion_mask:
                occlusion_mask: ground-truth occlusion mask, bool tensor equal to 1 where the pixel in the flow
                                image is occluded in the source image, 0 otherwise

            if mask_zero_borders:
                mask_zero_borders: bool tensor equal to 1 where the flow image is not equal to 0, 0 otherwise

            if get_mapping:
                mapping: pixel correspondence map in flow coordinate system, relating flow to source image
            else:
                flow_map: flow fields in flow coordinate system, relating flow to source image
        """
        # for all inputs[0] must be the source and inputs[1] must be the flow
        inputs_paths, flow_path = self.path_list[index]

        if not self.load_valid_mask:
            if self.load_size:
                inputs, flow, source_size = self.loader(self.root, inputs_paths, flow_path)
            else:
                inputs, flow = self.loader(self.root, inputs_paths, flow_path)
                source_size = inputs[0].shape
            if self.co_transform is not None:
                inputs, flow = self.co_transform(inputs, flow)

            mask = get_gt_correspondence_mask(flow)
        else:
            if self.load_occlusion_mask:
                if self.load_size:
                    inputs, flow, mask, occ_mask, source_size = self.loader(self.root, inputs_paths, flow_path,
                                                                            return_occlusion_mask=True)
                else:
                    # loader comes with a mask of valid correspondences
                    inputs, flow, mask, occ_mask = self.loader(self.root, inputs_paths, flow_path,
                                                               return_occlusion_mask=True)
                    source_size = inputs[0].shape
            else:
                if self.load_size:
                    inputs, flow, mask, source_size = self.loader(self.root, inputs_paths, flow_path)
                else:
                    # loader comes with a mask of valid correspondences
                    inputs, flow, mask = self.loader(self.root, inputs_paths, flow_path)
                    source_size = inputs[0].shape
            # mask is shape hxw
            if self.co_transform is not None:
                inputs, flow, mask = self.co_transform(inputs, flow, mask)

        if self.mask_zero_borders:
            mask_valid = define_mask_zero_borders(np.array(inputs[1]))

        # after co transform that could be reshapping the flow
        # transforms here will always contain conversion to tensor (then channel is before)
        if self.source_image_transform is not None:
            inputs[0] = self.source_image_transform(inputs[0])
        if self.target_image_transform is not None:
            inputs[1] = self.target_image_transform(inputs[1])
        if self.flow_transform is not None:
            flow = self.flow_transform(flow)

        output = {'source_image': inputs[0],
                  'target_image': inputs[1],
                  'correspondence_mask': mask.astype(np.bool) if \
                        version.parse(torch.__version__) >= version.parse("1.1") else mask.astype(np.uint8),
                  'source_image_size': source_size,
                  'sparse': False}
        if self.load_occlusion_mask:
            output['occlusion_mask'] = occ_mask

        if self.mask_zero_borders:
            output['mask_zero_borders'] = mask_valid.astype(np.bool) if \
                version.parse(torch.__version__) >= version.parse("1.1") else mask_valid.astype(np.uint8)

        if self.get_mapping:
            output['correspondence_map'] = convert_flow_to_mapping(flow)
        else:
            output['flow_map'] = flow
        return output

    def __len__(self):
        return len(self.path_list)
