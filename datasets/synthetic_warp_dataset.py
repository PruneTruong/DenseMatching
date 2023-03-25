from torch.utils.data import Dataset
import torch
from packaging import version
import torch.nn.functional as F
import numpy as np


from utils_flow.flow_and_mapping_operations import get_gt_correspondence_mask
from utils_flow.pixel_wise_mapping import warp


class WarpingDataset(torch.utils.data.Dataset):
    """Dataset applying random warps to single images to obtain pairs of matching images and their corresponding
    ground truth flow fields. The final image pair is composed of the central crops of the desired dimensions in both
    images.
    """

    def __init__(self, original_image_dataset, synthetic_flow_generator, compute_mask_zero_borders=False,
                 min_percent_valid_corr=0.1, crop_size=256, output_size=256, source_image_transform=None,
                 target_image_transform=None, flow_transform=None, co_transform=None, padding_mode='zeros'):
        """
        Args:
            original_image_dataset: dataset for the single images
            synthetic_flow_generator: module generating the dense flow fields
            compute_mask_zero_borders: compute mask zero borders?
            min_percent_valid_corr: compute_mask_zero_borders is True and if the percentage of valid flow regions is
                                    below this value, we use the ground-truth valid mask instead of the zero border mask
            crop_size: crop size, after applying the geometric transformations
            output_size: after the cropping, can optionally further resize the images, flow and masks
            source_image_transform: image transformations to apply to source images
            target_image_transform: image transformations to apply to target images
            flow_transform: flow transformations to apply to ground-truth flow fields
            co_transform: transforms to apply to both image pairs and corresponding flow field
            padding_mode: padding mode for warping. 'border' could be better, instead of 'zeros'

        Output in __getitem__:
            source_image
            target_image
            correspondence_mask: visible and valid correspondences
            source_image_size
            flow_map: flow fields in target coordinate system, relating target to source image
            sparse: False (only dense outputs here)

            if mask_zero_borders:
                mask_zero_borders: bool tensor equal to 1 where the target image is not equal to 0, 0 otherwise
        """
        self.original_image_dataset = original_image_dataset
        self.apply_mask_zero_borders = compute_mask_zero_borders
        self.padding_mode = padding_mode

        self.synthetic_flow_generator = synthetic_flow_generator
        if not isinstance(crop_size, (tuple, list)):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size

        if not isinstance(output_size, (tuple, list)):
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.min_percent_valid_corr = min_percent_valid_corr

        # processing of final images
        self.source_image_transform = source_image_transform
        if target_image_transform is None and source_image_transform is not None:
            self.target_image_transform = source_image_transform
        else:
            self.target_image_transform = target_image_transform
        self.flow_transform = flow_transform
        self.co_transform = co_transform

    def __len__(self):
        return self.original_image_dataset.__len__()

    def sample_new_items(self, seed):
        # sample new item in the background image dataset
        if hasattr(self.original_image_dataset, 'sample_new_items'):
            getattr(self.original_image_dataset, 'sample_new_items')(seed)

    def __getitem__(self, idx):
        """
        Args:
            idx:

        Returns: dictionary with fieldnames
            source_image
            target_image
            correspondence_mask: visible and valid correspondences
            source_image_size
            flow_map: flow fields in target coordinate system, relating target to source image
            sparse: False (only dense outputs here)

            if mask_zero_borders:
                mask_zero_borders: bool tensor equal to 1 where the target image is not equal to 0, 0 otherwise
        """

        dict = self.original_image_dataset.__getitem__(idx)
        image = dict['image']

        # process the image to have it in dimension (1, 3, H, W)
        if isinstance(image,  np.ndarray):
            # numpy array
            image = torch.from_numpy(image)
        if len(image.shape) == 4:
            if image.shape[1] != 3:
                image = image.permute(0, 3, 1, 2)
        else:
            if image.shape[0] != 3:
                image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)
        image = image.float()

        # take original images, have them at resolution 520x520
        b, _, h, w = image.shape
        if h <= self.output_size[0] or w <= self.output_size[1]:
            print('Image or flow is the same size than desired output size in warping dataset ! ')

        # get synthetic homography transformation from the synthetic flow generator
        flow_gt = self.synthetic_flow_generator().detach()
        flow_gt.require_grad = False
        bs, _, h_f, w_f = flow_gt.shape
        if h_f != h or w_f != w:
            # reshape and rescale the flow so it has the size of the original images
            flow_gt = F.interpolate(flow_gt, (h, w), mode='bilinear', align_corners=False)
            flow_gt[:, 0] *= float(w) / float(w_f)
            flow_gt[:, 1] *= float(h) / float(h_f)
        target_image, mask_zero_borders = warp(image, flow_gt, padding_mode=self.padding_mode, return_mask=True)
        target_image = target_image.byte()
        # because here, i still have my images in [0-255] s

        # crop a center patch from the images and the ground-truth flow field, so that black borders are removed
        x_start = w // 2 - self.crop_size[1] // 2
        y_start = h // 2 - self.crop_size[0] // 2
        source_image_resized = image[:, :, y_start: y_start + self.crop_size[0], x_start: x_start + self.crop_size[1]]
        target_image_resized = target_image[:, :, y_start: y_start + self.crop_size[0],
                                            x_start: x_start + self.crop_size[1]]
        flow_gt_resized = flow_gt[:, :, y_start: y_start + self.crop_size[0], x_start: x_start + self.crop_size[1]]
        mask_zero_borders = mask_zero_borders[:, y_start: y_start + self.crop_size[0], x_start: x_start + self.crop_size[1]]

        # resize to final outptu size, this is to prevent the crop from removing all common areas
        if self.output_size != self.crop_size:
            source_image_resized = F.interpolate(source_image_resized, self.output_size,
                                                 mode='area')
            target_image_resized = F.interpolate(target_image_resized, self.output_size,
                                                 mode='area')
            flow_gt_resized = F.interpolate(flow_gt_resized, self.output_size,
                                            mode='bilinear', align_corners=False)
            flow_gt_resized[:, 0] *= float(self.output_size[1]) / float(self.crop_size[1])
            flow_gt_resized[:, 1] *= float(self.output_size[0]) / float(self.crop_size[0])

            mask_zero_borders = F.interpolate(mask_zero_borders.float().unsqueeze(1), self.output_size,
                                              mode='bilinear', align_corners=False).byte().squeeze(1)
            mask_zero_borders = mask_zero_borders.bool() if version.parse(torch.__version__) >= version.parse("1.1") \
                else mask_zero_borders.byte()

        # put back the images and flow to numpy array, channel last
        # TO DO: change, this is not great
        source_image_resized = source_image_resized.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        target_image_resized = target_image_resized.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        flow_gt_resized = flow_gt_resized.squeeze(0).permute(1, 2, 0).numpy()
        mask_zero_borders = mask_zero_borders.squeeze(0).numpy()

        if self.co_transform is not None:
            [source_image_resized, target_image_resized], flow_gt_resized, mask_zero_borders = \
                self.co_transform([source_image_resized, target_image_resized], flow_gt_resized, mask=mask_zero_borders)

        # create ground truth mask (for eval at least)
        mask_gt = get_gt_correspondence_mask(flow_gt_resized)

        if self.apply_mask_zero_borders:
            # if mask_gt is all zero (no commun areas), overwrite to use the mask in anycase
            if mask_gt.sum() < mask_gt.shape[-1] * mask_gt.shape[-2] * self.min_percent_valid_corr:
                mask_zero_borders = mask_gt
            else:
                # if padding is 'zeros', could identify mask_zero_borders from intensity of the target image directly
                # mask_zero_borders = define_mask_zero_borders(target_image_resized)
                mask_gt *= mask_zero_borders  # also removes the black area from the valid mask

        if self.source_image_transform is not None:
            source_image_resized = self.source_image_transform(source_image_resized)
        if self.target_image_transform is not None:
            target_image_resized = self.target_image_transform(target_image_resized)
        if self.flow_transform is not None:
            flow_gt_resized = self.flow_transform(flow_gt_resized)

        # save the new batch information
        output = {'source_image': source_image_resized,
                  'target_image': target_image_resized,
                  'flow_map': flow_gt_resized,
                  'correspondence_mask': mask_gt
                  }
        if self.apply_mask_zero_borders:
            output['mask_zero_borders'] = mask_zero_borders
        output['sparse'] = False
        return output
