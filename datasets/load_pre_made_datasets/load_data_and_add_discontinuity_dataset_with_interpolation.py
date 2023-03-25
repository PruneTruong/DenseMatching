import torch.utils.data as data
import os
import os.path
import cv2
import numpy as np
import torch
import jpeg4py
import random
from packaging import version


from utils_flow.pixel_wise_mapping import remap_using_flow_fields, warp
from utils_data.augmentations.geometric_distortions import ElasticTransform
from utils_flow.flow_and_mapping_operations import (convert_flow_to_mapping, convert_mapping_to_flow,
                                                    get_gt_correspondence_mask)
from utils_data.io import load_flo
from utils_flow.img_processing_utils import define_mask_zero_borders


def default_loader(root, path_imgs, path_flo):
    imgs = [os.path.join(root, path) for path in path_imgs]
    flo = os.path.join(root, path_flo)

    if imgs[0].endswith('.jpg') or imgs[0].endswith('.jpeg'):
        try:
            return [jpeg4py.JPEG(img).decode().astype(np.uint8) for img in imgs], load_flo(flo)
        except:
            return [cv2.imread(img)[:, :, ::-1].astype(np.uint8) for img in imgs], load_flo(flo)
    else:
        img_list = [cv2.imread(img)[:, :, ::-1].astype(np.uint8) for img in imgs]
        flow = load_flo(flo)
        return img_list, flow


class DiscontinuityDatasetV2(data.Dataset):
    def __init__(self, root, path_list, source_image_transform=None, target_image_transform=None, flow_transform=None,
                 co_transform=None, loader=default_loader, load_valid_mask=False, load_size=False, get_mapping=False,
                 compute_mask_zero_borders=False, max_nbr_perturbations=10, min_nbr_perturbations=3,
                 elastic_parameters=None, max_sigma_mask=10, min_sigma_mask=3):
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
            get_mapping: get mapping ?
            compute_mask_zero_borders: output mask of zero borders ?
            min_nbr_perturbations
            max_nbr_perturbations
            elastic_parameters: dictionary containing parameters for the elastic flow
                                default is {"max_sigma": 0.04, "min_sigma": 0.1, "min_alpha": 1, "max_alpha": 0.4}
            max_sigma_mask: max sigma for the binary masks, in which the perturbations are applied
            min_sigma_mask: min sigma for the binary masks, in which the perturbations are applied
        Output in __getitem__:
            source_image
            target_image
            correspondence_mask: valid correspondences
            source_image_size
            sparse: False (only dense outputs here)

            if mask_zero_borders:
                mask_zero_borders: bool tensor equal to 1 where the target image is not equal to 0, 0 otherwise

            if get_mapping:
                mapping: pixel correspondence map in target coordinate system, relating target to source image
            else:
                flow_map: flow fields in target coordinate system, relating target to source image
        """

        self.root = root
        self.path_list = path_list
        self.first_image_transform = source_image_transform
        self.second_image_transform = target_image_transform
        self.target_transform = flow_transform
        self.co_transform = co_transform
        self.loader = loader
        self.mask = load_valid_mask
        self.size = load_size
        self.get_mapping = get_mapping
        self.mask_zero_borders=compute_mask_zero_borders
        self.max_nbr_perturbations = max_nbr_perturbations
        self.min_nbr_perturbations = min_nbr_perturbations
        if elastic_parameters is None:
            elastic_parameters = {"max_sigma": 0.04, "min_sigma": 0.1, "min_alpha": 1, "max_alpha": 0.4}
        self.max_sigma_mask = max_sigma_mask
        self.min_sigma_mask = min_sigma_mask
        self.ElasticTrans = ElasticTransform(elastic_parameters, get_flow=True, approximate=True)

    @staticmethod
    def get_gaussian(shape, mu, sigma):
        x = np.indices(shape)
        mu = np.float32(mu).reshape(2, 1, 1)
        n = sigma * np.sqrt(2 * np.pi) ** len(x)
        return np.exp(-0.5 * (((x - mu) / sigma) ** 2).sum(0)) / n

    def __getitem__(self, index):
        # for all inputs[0] must be the source and inputs[1] must be the final_flow
        inputs, final_flow = self.path_list[index]

        if not self.mask:
            if self.size:
                inputs, final_flow, source_size = self.loader(self.root, inputs, final_flow)
            else:
                inputs, final_flow = self.loader(self.root, inputs, final_flow)
                source_size = inputs[0].shape
            if self.co_transform is not None:
                inputs, final_flow = self.co_transform(inputs, final_flow)

            mask_valid_correspondences = get_gt_correspondence_mask(final_flow)
        else:
            if self.size:
                inputs, final_flow, mask_valid_correspondences, source_size = self.loader(self.root, inputs, final_flow)
            else:
                # loader comes with a mask of valid correspondences
                inputs, final_flow, mask_valid_correspondences = self.loader(self.root, inputs, final_flow)
                source_size = inputs[0].shape
            # mask is shape hxw
            if self.co_transform is not None:
                inputs, final_flow, mask_valid_correspondences = self.co_transform(inputs, final_flow,
                                                                                   mask_valid_correspondences)

        mapping = convert_flow_to_mapping(final_flow, output_channel_first=False)
        shape = final_flow.shape[:2]

        target_image = inputs[1]

        nbr_perturbations = random.randint(self.min_nbr_perturbations, self.max_nbr_perturbations)

        sigma_, alpha = self.ElasticTrans.get_random_paremeters(shape, seed=None)
        flow_x_pertu, flow_y_pertu = self.ElasticTrans.get_mapping_from_distorted_image_to_undistorted(
            shape, sigma_, alpha, seed=None)

        flow_pertu = np.dstack((flow_x_pertu, flow_y_pertu))
        mask_final = np.zeros(shape, np.float32)

        # make the mask
        for i in range(nbr_perturbations):
            sigma = random.randint(self.min_sigma_mask, self.max_sigma_mask)
            coordinate_in_mask=False
            while coordinate_in_mask is False:
                x = random.randint(0 + sigma * 3, shape[1] - sigma * 3)
                y = random.randint(0 + sigma * 3, shape[0] - sigma * 3)
                if mask_valid_correspondences[y, x]:
                    coordinate_in_mask = True
            mask = self.get_gaussian(shape, mu=[x, y], sigma=sigma)

            max = mask.max()
            mask = np.clip(2.0 / max * mask, 0.0, 1.0)
            mask_final = mask_final + mask

        mask = np.clip(mask_final, 0.0, 1.0)
        flow_pertu = flow_pertu * np.tile(np.expand_dims(mask, axis=2), (1, 1, 2))

        final_mapping = warp(torch.Tensor(mapping).unsqueeze(0).permute(0, 3, 1, 2),
                             torch.Tensor(flow_pertu).unsqueeze(0).permute(0, 3, 1, 2))
        final_mapping = final_mapping.squeeze(0).permute(1, 2, 0).cpu().numpy()
        new_target_image = remap_using_flow_fields(target_image, flow_pertu[:, :, 0], flow_pertu[:, :, 1])
        inputs[1] = new_target_image

        final_flow = convert_mapping_to_flow(final_mapping, output_channel_first=False)

        if self.mask_zero_borders:
            mask_valid = define_mask_zero_borders(np.array(inputs[1]))
        # after co transform that could be reshapping the final_flow
        # transforms here will always contain conversion to tensor (then channel is before)
        if self.first_image_transform is not None:
            inputs[0] = self.first_image_transform(inputs[0])
        if self.second_image_transform is not None:
            inputs[1] = self.second_image_transform(inputs[1])
        if self.target_transform is not None:
            final_flow = self.target_transform(final_flow)

        output = {'source_image': inputs[0],
                  'target_image': inputs[1],
                  'correspondence_mask': mask_valid_correspondences.astype(np.bool) if
                  version.parse(torch.__version__) >= version.parse("1.1") else mask_valid_correspondences.astype(np.uint8),
                  'source_image_size': source_size,
                  'sparse': False
                  }

        if self.mask_zero_borders:
            output['mask_zero_borders'] = mask_valid.astype(np.bool) if version.parse(torch.__version__) >= version.parse("1.1") \
                                              else mask_valid.astype(np.uint8),
        if self.get_mapping:
            output['correspondence_map']: convert_flow_to_mapping(final_flow)
        else:
            output['flow_map'] = final_flow
        return output

    def __len__(self):
        return len(self.path_list)
