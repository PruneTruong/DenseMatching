import numpy as np
import cv2 as cv
from packaging import version
import cv2
import torch.nn.functional as F
import random
import torch


from .base_video_dataset import BaseVideoDataset
from datasets.object_augmented_dataset.bounding_box_utils import masks_to_bboxes
from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from utils_flow.flow_and_mapping_operations import get_gt_correspondence_mask


def from_homography_to_pixel_wise_mapping(shape, H):
    """
    From a homography relating target image to source image, computes pixel wise mapping and pixel wise displacement
    from pixels of target image to source image.
    Args:
        shape: shape of target image
        H: homography

    Returns:
        disp_x: displacement of each pixel of target image in the horizontal direction
        disp_y: displacement of each pixel of target image in the vertical direction
    """
    h_scale, w_scale = shape[:2]

    # estimate the grid
    X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                       np.linspace(0, h_scale - 1, h_scale))
    X, Y = X.flatten(), Y.flatten()
    # X is same shape as shape, with each time the horizontal index of the pixel

    # create matrix representation --> each contain horizontal coordinate, vertical and 1
    XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T

    # multiply Hinv to XYhom to find the warped grid
    XYwarpHom = np.dot(H, XYhom)
    Xwarp = XYwarpHom[0, :]/(XYwarpHom[2, :]+1e-8)
    Ywarp = XYwarpHom[1, :]/(XYwarpHom[2, :]+1e-8)

    # reshape to obtain the ground truth mapping
    map_x = Xwarp.reshape((h_scale, w_scale))
    map_y = Ywarp.reshape((h_scale, w_scale))
    disp_x = map_x.astype(np.float32)-X.reshape((h_scale, w_scale))
    dixp_y = map_y.astype(np.float32)-Y.reshape((h_scale, w_scale))

    return disp_x, dixp_y


class AugmentedImagePairsDatasetMultipleObjects(BaseVideoDataset):
    """
    Augment an image pair by applying random transformations to an object (foreground) and pasting it on the
    background images.  Currently, the foreground object is pasted at random locations in different frames.
    Update the ground-truth flow field relating the image pair accordingly.
    """
    def __init__(self, foreground_image_dataset, background_image_dataset, foreground_transform=None,
                 source_image_transform=None, target_image_transform=None, flow_transform=None,
                 co_transform=None, compute_occlusion_mask=False, compute_out_of_view_mask=False,
                 compute_object_reprojection_mask=False, compute_mask_zero_borders=False, random_nbr_objects=False,
                 number_of_objects=4, object_proba=0.8, output_flow_size=None):
        """
        Args:
            foreground_image_dataset - A segmentation dataset from which foreground objects are cropped using the
                                       segmentation mask
            background_image_dataset - Dataset used to sample the original image pairs and their
                                       corresponding ground-truth flow field
            foreground_transform - Random transformations to be applied to the foreground object in every frame
            source_image_transform: image transformations to apply to source images
            target_image_transform: image transformations to apply to target images
            flow_transform: flow transformations to apply to ground-truth flow fields
            co_transform: transforms to apply to both image pairs and corresponding flow field
            compute_occlusion_mask: compute the occlusion mask and have this in dict field 'correspondence_mask' ?
            compute_out_of_view_mask: compute the out_of_view mask and have this in dict field 'correspondence_mask' ?
            compute_object_reprojection_mask: compute the reprojection mask (part of occlusion) and have this in
                                              dict field 'correspondence_mask' ?
            compute_mask_zero_borders: output mask of zero borders ?
            number_of_objects: maximum number of objects to add to the background image pair
            object_proba: add objects with probability below object_proba
            output_flow_size: if None, ground-truth flow has image dimensions. This can be a list of sizes, e.g.
                              [[520, 520], [256, 256]]. Then the ground-truth is returned in both sizes, in dict
                              for fieldname 'flow_map'

        Output in __getitem__:
            source_image: new source image with pasted objects
            target_image: new target image with pasted objects
            flow_map: if self.output_flow_size is a list of sizes, contains a list of flow_fields. Otherwise, contains
                      a single flow field. The flow fields are in target coordinate system, relating target to source image.
            correspondence_mask: if self.output_flow_size is a list of sizes, contains a list of bool binary masks.
                                 Each indicates visible and valid correspondences
            source_image_size
            sparse: False

            if mask_zero_borders:
                mask_zero_borders: bool tensor equal to 1 where the target image is not equal to 0, 0 otherwise
        """
        assert foreground_image_dataset.has_segmentation_info()

        super().__init__(foreground_image_dataset.get_name() + '_syn_vid_blend', foreground_image_dataset.root,
                         foreground_image_dataset.image_loader)
        self.foreground_image_dataset = foreground_image_dataset
        self.background_image_dataset = background_image_dataset

        # image and flow transformations
        self.first_image_transform = source_image_transform
        self.second_image_transform = target_image_transform
        self.flow_transform = flow_transform
        self.co_transform = co_transform

        self.object_proba = object_proba
        self.foreground_transform = foreground_transform

        self.compute_zero_border_mask = compute_mask_zero_borders
        self.compute_occlusion_mask = compute_occlusion_mask
        self.compute_out_of_view_mask = compute_out_of_view_mask
        self.compute_object_reprojection_mask = compute_object_reprojection_mask
        self.random_nbr_objects = random_nbr_objects
        self.number_of_objects = number_of_objects
        self.size_flow = output_flow_size

    def get_name(self):
        return self.name

    def is_video_sequence(self):
        return False

    def has_class_info(self):
        return self.foreground_image_dataset.has_class_info()

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return self.foreground_image_dataset.get_num_images()

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.get_images_in_class[class_name]

    def get_sequence_info(self, seq_id):
        image_info = self.foreground_image_dataset.get_image_info(seq_id)

        image_info = {k: v.unsqueeze(0) for k, v in image_info.items()}
        return image_info

    def get_class_name(self, seq_id):
        return self.foreground_image_dataset.get_class_name(seq_id)

    @staticmethod
    def _paste_target(fg_image, fg_box, fg_mask, bg_image, paste_loc):
        fg_mask = fg_mask.view(fg_mask.shape[0], fg_mask.shape[1], 1)
        fg_box = fg_box.long().tolist()

        x1 = int(paste_loc[0] - 0.5 * fg_box[2])
        x2 = x1 + fg_box[2]

        y1 = int(paste_loc[1] - 0.5 * fg_box[3])
        y2 = y1 + fg_box[3]

        x1_pad = max(-x1, 0)
        y1_pad = max(-y1, 0)

        x2_pad = max(x2 - bg_image.shape[1], 0)
        y2_pad = max(y2 - bg_image.shape[0], 0)

        bg_mask = torch.zeros((bg_image.shape[0], bg_image.shape[1], 1), dtype=fg_mask.dtype,
                              device=fg_mask.device)

        if x1_pad >= fg_mask.shape[1] or x2_pad >= fg_mask.shape[1] or y1_pad >= fg_mask.shape[0] or y2_pad >= \
                fg_mask.shape[0]:
            return bg_image, bg_mask.squeeze(-1),

        fg_mask_patch = fg_mask[fg_box[1] + y1_pad:fg_box[1] + fg_box[3] - y2_pad,
                                fg_box[0] + x1_pad:fg_box[0] + fg_box[2] - x2_pad, :]

        fg_image_patch = fg_image[fg_box[1] + y1_pad:fg_box[1] + fg_box[3] - y2_pad,
                                  fg_box[0] + x1_pad:fg_box[0] + fg_box[2] - x2_pad, :]

        bg_image[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :] = \
            bg_image[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :] * (1 - fg_mask_patch.numpy()) \
            + fg_mask_patch.numpy() * fg_image_patch

        bg_mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :] = fg_mask_patch

        return bg_image, bg_mask.squeeze(-1)

    def __len__(self):
        """ Returns size of the dataset
        returns:
            int - number of samples in the dataset
        """
        return self.background_image_dataset.__len__()

    def sample_new_items(self, seed):
        # sample new item in the background image dataset
        if hasattr(self.background_image_dataset, 'sample_new_items'):
            getattr(self.background_image_dataset, 'sample_new_items')(seed)

    def __getitem__(self, index):
        """
        TODO: clean-up

        Args:
            index

            self.background_image_dataset.__getitem__(index) must return a dictionary, with fields 'source_image',
            'target_image', 'flow_map', 'correspondence_mask'. 'mask_zero_borders' if self.mask_zero_borders

        Returns: Dictionary with fieldnames:
            source_image
            target_image
            flow_map: if self.output_flow_size is a list of sizes, contains a list of flow_fields. Otherwise, contains
                      a single flow field. The flow fields are in target coordinate system, relating target to source image.
            correspondence_mask: if self.output_flow_size is a list of sizes, contains a list of bool binary masks.
                                 Each indicates visible and valid correspondences
            source_image_size
            sparse: False

            if mask_zero_borders:
                mask_zero_borders: bool tensor equal to 1 where the target image is not equal to 0, 0 otherwise
        """

        # get background image, which correspond to the original image pair
        background_sample = self.background_image_dataset.__getitem__(index)
        bg_frame_list = [background_sample['source_image'], background_sample['target_image']]
        size_bg = bg_frame_list[0].shape

        # retrieve ground-truth flow relating the original target to source image
        if isinstance(background_sample['flow_map'], np.ndarray):
            flow_bg = torch.from_numpy(background_sample['flow_map'])
        else:
            flow_bg = background_sample['flow_map']
        if flow_bg.shape[0] == 2:
            flow_bg = flow_bg.permute(1, 2, 0)

        # retrieve ground-truth valid mask relating the original target to source image
        if isinstance(background_sample['correspondence_mask'], np.ndarray):
            correspondence_mask = torch.from_numpy(background_sample['correspondence_mask'])
        else:
            correspondence_mask = background_sample['correspondence_mask']
        correspondence_mask = correspondence_mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") \
            else correspondence_mask.byte()

        if self.compute_zero_border_mask:
            mask_zero_borders = background_sample['mask_zero_borders']
            mask_zero_borders = torch.from_numpy(background_sample['mask_zero_borders']) if \
                isinstance(mask_zero_borders, np.ndarray) else mask_zero_borders

        # to have several objects, we start from flow file is background and occlusion is none
        occluded_mask = ~correspondence_mask  # invert bool
        flow_file = flow_bg

        mask_of_objects_in_source = torch.zeros_like(occluded_mask)
        mask_of_objects_in_target = torch.zeros_like(occluded_mask)
        mask_of_reprojected_object_from_source_to_target = torch.zeros_like(occluded_mask)

        [source_image, target_image] = bg_frame_list

        if np.random.rand() < self.object_proba:  # only add objects with a certain probability
            if self.random_nbr_objects:
                number_of_objects = random.randint(1, self.number_of_objects)
            else:
                number_of_objects = self.number_of_objects
            for ob_id in range(0, number_of_objects):
                # Handle foreground
                seq_id = random.randint(0, self.get_num_sequences() - 1)
                anno = self.get_sequence_info(seq_id)

                image_fg, fg_anno, fg_object_meta = self.foreground_image_dataset.get_image(seq_id, anno=anno)

                # get segmentation mask and bounding box of the foreground object to past
                mask_fg = fg_anno['mask'][0]  # float32
                bbox_fg = fg_anno['bbox'][0]

                # if the object is too big, reduce it:
                number_of_pixels = bg_frame_list[0].shape[0]*bg_frame_list[0].shape[1]
                if mask_fg.sum() > 0.5 * number_of_pixels:
                    scale = random.uniform(0.1, 0.4) * number_of_pixels / mask_fg.sum()
                    image_fg, bbox_fg, mask_fg, _ = self.foreground_transform.transform_with_specific_values(
                        image_fg, bbox_fg, mask_fg, do_flip=False, theta=0, shear_values=(0, 0),
                        scale_factors=(scale, scale), tx=0, ty=0)

                # for the target image, put the object at random location on the target background
                loc_y_target = random.randint(0, bg_frame_list[1].shape[0] - 1)
                loc_x_target = random.randint(0, bg_frame_list[1].shape[1] - 1)

                target_image, target_mask_fg = self._paste_target(image_fg, masks_to_bboxes(mask_fg, fmt='t'),
                                                               mask_fg, bg_frame_list[1],  # original target image
                                                               (loc_x_target, loc_y_target))

                # computes the geometric transformation applied to the object to paste it in the source image
                # the transformation is defined in self.foreground_transform and we additionally add translation
                # to decide more or less the location of object in the source image
                # proba half, transform is only a small translation, half its a random location
                if np.random.rand() < 0.5:
                    # make a small translation only
                    translation_x = random.randrange(-8, 8)
                    loc_x_source = loc_x_target + translation_x
                    if loc_x_source > (bg_frame_list[1].shape[1] - 1) or loc_x_source < 0:
                        loc_x_source = loc_x_target - translation_x

                    translation_y = random.randrange(-8, 8)
                    loc_y_source = loc_y_target + translation_y
                    if loc_y_source > (bg_frame_list[1].shape[0] - 1) or loc_y_source < 0:
                        loc_y_source = loc_y_target - translation_y
                else:
                    loc_y_source = random.randint(0, bg_frame_list[1].shape[0] - 1)
                    loc_x_source = random.randint(0, bg_frame_list[1].shape[1] - 1)

                # corresponding translation
                tx = loc_x_source - loc_x_target
                ty = loc_y_source - loc_y_target

                # transform the newly augmentated target image, and keep track of the object bounding box and
                # segmentation mask, also outputs the homography h corresponding to the geometric transformation of
                # the object.
                source_image_fg, source_bbx_fg, source_mask_fg, h = self.foreground_transform.transform(
                    image=target_image, bbox=masks_to_bboxes(target_mask_fg, fmt='t'), mask=target_mask_fg,
                    tx=tx, ty=ty)

                # get the flow corresponding to this transformation
                flow_x, flow_y = from_homography_to_pixel_wise_mapping(target_image.shape[:2], h)
                flow_fg_object = np.dstack([flow_x, flow_y])

                source_mask_fg = source_mask_fg.bool() if version.parse(torch.__version__) >= version.parse("1.1") \
                    else source_mask_fg.byte()
                target_mask_fg = target_mask_fg.bool() if version.parse(torch.__version__) >= version.parse("1.1") \
                    else target_mask_fg.byte()

                # compute newly augmented source image. it is the source background except at location of object !
                source_image = np.where(source_mask_fg.unsqueeze(2).numpy(), source_image_fg, bg_frame_list[0])


                # compute the final flow as composition of background and foreground
                if source_mask_fg.sum() != 0:
                    # the object that is in target is not fully occluded in the source (the object is also present
                    # in the source image)
                    flow_file = torch.where(target_mask_fg.unsqueeze(2), torch.from_numpy(flow_fg_object).float(),
                                            flow_file)

                    mask_of_objects_in_target = mask_of_objects_in_target | target_mask_fg

                    # current object in source covering old objects in source
                    area_of_source_objects_covered_by_new_object = mask_of_objects_in_source & source_mask_fg
                    mask_of_objects_in_source = mask_of_objects_in_source | source_mask_fg
                    if self.compute_occlusion_mask or self.compute_object_reprojection_mask:
                        # occlusion due to object in source, i.e. places it covers does not have correspondence
                        # in target image.
                        # warped according to original bg !
                        mask_fg_object_source_in_target_frame = np.clip(
                            remap_using_flow_fields(source_mask_fg.float().numpy(), flow_bg[:, :, 0].numpy(),
                                                    flow_bg[:, :, 1].numpy()), 0, 1)
                        mask_fg_object_source_in_target_frame = torch.Tensor(
                            cv2.erode(cv2.dilate(mask_fg_object_source_in_target_frame,
                                                 np.ones((3, 3), np.uint8), iterations=1),
                                      np.ones((3, 3), np.uint8), iterations=1))
                        mask_fg_object_source_in_target_frame = mask_fg_object_source_in_target_frame.bool() \
                            if version.parse(torch.__version__) >= version.parse("1.1")\
                            else mask_fg_object_source_in_target_frame.byte()

                        # not fully valid if object was further covered
                        if area_of_source_objects_covered_by_new_object.sum() > 0:
                            # warp area covering old objects  with current object in source by all flow
                            # should be included in occluded regions
                            mask_occluded_in_source = np.clip(
                                remap_using_flow_fields(source_mask_fg.float().numpy(), flow_file[:, :, 0].numpy(),
                                                        flow_file[:, :, 1].numpy()), 0, 1)
                            mask_occluded_in_source = torch.Tensor(mask_occluded_in_source).byte()
                            mask_occluded_in_source = mask_occluded_in_source.bool() \
                                if version.parse(torch.__version__) >= version.parse("1.1") else mask_occluded_in_source.byte()
                            mask_of_reprojected_object_from_source_to_target = \
                                mask_of_reprojected_object_from_source_to_target | mask_occluded_in_source

                        # add current mask of reprojection to the final mask of reprojection
                        mask_of_reprojected_object_from_source_to_target = \
                            mask_of_reprojected_object_from_source_to_target | mask_fg_object_source_in_target_frame

                        # remove current target mask from reproction mask (there is an object here,
                        # so correct match at the current level), but can cover older objects.
                        mask_of_reprojected_object_from_source_to_target = \
                            mask_of_reprojected_object_from_source_to_target & ~target_mask_fg
                else:
                    # the object that is in target is fully occluded in the source (the object is NOT present
                    # in the source image). Therefore, the flow needs to be background everywhere.
                    # the target object is therefore an occlusion for the background basically.
                    mask_of_reprojected_object_from_source_to_target = \
                        mask_of_reprojected_object_from_source_to_target | target_mask_fg
                    flow_file = flow_file

                bg_frame_list = [source_image, target_image]

        # from the reprojection, make sure that we do not occlude some part of target object
        valid_flow = get_gt_correspondence_mask(flow_file)  # remove out of regions flow
        occluded_mask = ~valid_flow | mask_of_reprojected_object_from_source_to_target
        # occluded_mask = mask_of_reprojected_object_from_source_to_target
        if self.compute_zero_border_mask:
            # final mask is mask_zero_border or the object if they cover some previously invalid regions
            mask_zero_borders = mask_zero_borders | mask_of_objects_in_target

        # choose what the correspondence_mask represents
        if self.compute_occlusion_mask:
            correspondence_mask = ~occluded_mask
        elif self.compute_out_of_view_mask:
            correspondence_mask = get_gt_correspondence_mask(flow_file)
        elif self.compute_zero_border_mask:
            correspondence_mask = mask_zero_borders
        elif self.compute_object_reprojection_mask:
            correspondence_mask = ~mask_of_reprojected_object_from_source_to_target
        else:
            correspondence_mask = correspondence_mask
        correspondence_mask = correspondence_mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") \
            else correspondence_mask.byte()

        if self.co_transform is not None:
            # this one is wrong if flow_file is a list
            [source_image, target_image], flow_file, correspondence_mask = \
                self.co_transform([source_image, target_image], flow_file, correspondence_mask)
        if self.first_image_transform is not None:
            source_image = self.first_image_transform(source_image)
        if self.second_image_transform is not None:
            target_image = self.second_image_transform(target_image)

        if self.size_flow is not None:
            list_of_flows = []
            list_of_masks = []
            h_o, w_o = flow_file.shape[:2]
            for i_size in self.size_flow:
                # [h_size, w_size] = i_size
                flow_resized = F.interpolate(input=flow_file.permute(2, 0, 1).unsqueeze(1), size=i_size,
                                             mode='bilinear', align_corners=False).squeeze()
                flow_resized[0] *= float(i_size[0]) / float(h_o)
                flow_resized[1] *= float(i_size[1]) / float(w_o)
                list_of_flows.append(flow_resized) # already in dimension c, h, w
                mask_resized = F.interpolate(input=correspondence_mask.unsqueeze(0).unsqueeze(0).float(), size=i_size,
                                             mode='bilinear', align_corners=False).squeeze()
                list_of_masks.append(mask_resized.bool() if version.parse(torch.__version__) >= version.parse("1.1")
                                     else mask_resized.byte())

            flow_file = list_of_flows
            correspondence_mask = list_of_masks

        else:
            if self.flow_transform is not None:
                flow_file = self.flow_transform(flow_file)

        output = {'source_image': source_image,  'target_image': target_image, 'flow_map': flow_file,
                  'correspondence_mask': correspondence_mask, 'source_image_size': size_bg, 'sparse': False}
        if self.compute_zero_border_mask:
            output['mask_zero_borders'] = mask_zero_borders
        return output


class RandomAffine:
    """Apply random affine transformation."""
    def __init__(self, p_flip=0.0, max_rotation=0.0, max_shear=0.0, max_scale=0.0, max_ar_factor=0.0,
                 border_mode='constant', pad_amount=0):
        """

        Args:
            p_flip:
            max_rotation:
            max_shear:
            max_scale:
            max_ar_factor:
            border_mode:
            pad_amount:
        """
        super().__init__()
        self.p_flip = p_flip
        self.max_rotation = max_rotation
        self.max_shear = max_shear
        self.max_scale = max_scale
        self.max_ar_factor = max_ar_factor

        if border_mode == 'constant':
            self.border_flag = cv.BORDER_CONSTANT
        elif border_mode == 'replicate':
            self.border_flag == cv.BORDER_REPLICATE
        else:
            raise Exception

        self.pad_amount = pad_amount

    def roll(self):
        do_flip = random.random() < self.p_flip
        theta = random.uniform(-self.max_rotation, self.max_rotation)

        shear_x = random.uniform(-self.max_shear, self.max_shear)
        shear_y = random.uniform(-self.max_shear, self.max_shear)

        ar_factor = np.exp(random.uniform(-self.max_ar_factor, self.max_ar_factor))
        scale_factor = np.exp(random.uniform(-self.max_scale, self.max_scale))

        return do_flip, theta, (shear_x, shear_y), (scale_factor, scale_factor * ar_factor)

    def _construct_t_mat(self, image_shape, do_flip, theta, shear_values, scale_factors, tx, ty):
        im_h, im_w = image_shape
        t_mat = np.identity(3)

        if do_flip:
            t_mat[0, 0] = -1.0
            t_mat[0, 2] = im_w

        t_rot = cv.getRotationMatrix2D((im_w * 0.5, im_h * 0.5), theta, 1.0)
        t_rot = np.concatenate((t_rot, np.array([0.0, 0.0, 1.0]).reshape(1, 3)))

        t_shear = np.array([[1.0, shear_values[0], -shear_values[0] * 0.5 * im_w],
                            [shear_values[1], 1.0, -shear_values[1] * 0.5 * im_h],
                            [0.0, 0.0, 1.0]])

        t_scale = np.array([[scale_factors[0], 0.0, (1.0 - scale_factors[0]) * 0.25 * im_w],
                            [0.0, scale_factors[1], (1.0 - scale_factors[1]) * 0.25 * im_h],
                            [0.0, 0.0, 1.0]])

        t_translation = np.identity(3)
        t_translation[0,2] = tx
        t_translation[1,2] = ty

        t_mat = t_scale @ t_rot @ t_shear @ t_mat @ t_translation

        t_mat[0, 2] += self.pad_amount
        t_mat[1, 2] += self.pad_amount

        t_mat_affine = t_mat[:2, :]

        return t_mat_affine, t_mat

    def transform_image(self, image, do_flip, theta, shear_values, scale_factors, tx, ty):
        if torch.is_tensor(image):
            raise Exception('Only supported for numpy input')

        t_mat, h = self._construct_t_mat(image.shape[:2], do_flip, theta, shear_values, scale_factors, tx, ty)
        image_t = cv.warpPerspective(image, h, (image.shape[1], image.shape[0]), cv.INTER_LINEAR,
                                   borderMode=self.border_flag)

        return image_t

    def transform_bbox(self, bbox, image_shape, *rand_params):
        """Assumes [x, y, w, h]"""
        # Check if not overloaded

        coord = bbox.clone().view(-1, 2).t().flip(0)

        x1 = coord[1, 0]
        x2 = coord[1, 0] + coord[1, 1]

        y1 = coord[0, 0]
        y2 = coord[0, 0] + coord[0, 1]

        coord_all = torch.tensor([[y1, y1, y2, y2], [x1, x2, x2, x1]])

        coord_transf = self.transform_coords(coord_all, image_shape, *rand_params).flip(0)
        tl = torch.min(coord_transf, dim=1)[0]
        sz = torch.max(coord_transf, dim=1)[0] - tl
        bbox_out = torch.cat((tl, sz), dim=-1).reshape(bbox.shape)
        return bbox_out

    def transform_coords(self, coords, image_shape, do_flip, theta, shear_values, scale_factors, tx, ty):
        t_mat, h = self._construct_t_mat(image_shape[:2], do_flip, theta, shear_values, scale_factors, tx, ty)

        t_mat_tensor = torch.from_numpy(t_mat).float()

        coords_xy1 = torch.stack((coords[1, :], coords[0, :], torch.ones_like(coords[1, :])))

        coords_xy_t = torch.mm(t_mat_tensor, coords_xy1)

        return coords_xy_t[[1, 0], :]

    def transform_mask(self, mask, do_flip, theta, shear_values, scale_factors, tx, ty):
        t_mat, h = self._construct_t_mat(mask.shape[:2], do_flip, theta, shear_values, scale_factors, tx, ty)
        mask_t = cv.warpPerspective(mask.numpy(), h, (mask.shape[1], mask.shape[0]), cv.INTER_LINEAR,
                                    borderMode=self.border_flag)

        return torch.from_numpy(mask_t)

    def transform(self, image, bbox, mask, tx, ty):
        do_flip, theta, shear_values, scale_factors = self.roll()
        t_mat, h = self._construct_t_mat(image.shape[:2], do_flip, theta, shear_values, scale_factors, tx, ty)
        image_t = self.transform_image(image, do_flip, theta, shear_values, scale_factors, tx, ty)
        bbx = self.transform_bbox(bbox, image.shape[:2], do_flip, theta, shear_values, scale_factors, tx, ty)
        mask = self.transform_mask(mask, do_flip, theta, shear_values, scale_factors, tx, ty)
        return image_t, bbx, mask, h

    def transform_with_specific_values(self, image, bbox, mask, do_flip, theta, shear_values, scale_factors, tx, ty):
        t_mat, h = self._construct_t_mat(image.shape[:2], do_flip, theta, shear_values, scale_factors, tx, ty)
        image_t = self.transform_image(image, do_flip, theta, shear_values, scale_factors, tx, ty)
        bbx = self.transform_bbox(bbox, image.shape[:2], do_flip, theta, shear_values, scale_factors, tx, ty)
        mask = self.transform_mask(mask, do_flip, theta, shear_values, scale_factors, tx, ty)
        return image_t, bbx, mask, h
