import torch.utils.data
import torch
import torch.nn.functional as F
from packaging import version

from utils_flow.flow_and_mapping_operations import unormalise_and_convert_mapping_to_flow
from models.base_matching_net import pre_process_image_glunet


def no_processing(data):
    return data


def normalize_image_with_imagenet_weights(source_img):
    # img has shape bx3xhxw
    b, _, h_scale, w_scale = source_img.shape
    mean_vector = [0.485, 0.456, 0.406]
    std_vector = [0.229, 0.224, 0.225]

    # original resolution
    source_img_copy = source_img.float().div(255.0)
    mean = torch.as_tensor(mean_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
    std = torch.as_tensor(std_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
    source_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])
    return source_img_copy


class GLUNetBatchPreprocessing:
    """ Class responsible for processing the mini-batch to create the desired training inputs for GLU-Net based networks.
    Particularly, from the source and target images at original resolution as well as the corresponding ground-truth
    flow field, needs to create the source, target and flow at resolution 256x256 for training the L-Net.
    """

    def __init__(self, settings, apply_mask=False, apply_mask_zero_borders=False, sparse_ground_truth=False,
                 mapping=False):
        """
        Args:
            settings: settings
            apply_mask: apply ground-truth correspondence mask for loss computation?
            apply_mask_zero_borders: apply mask zero borders (equal to 0 at black borders in target image) for loss
                                     computation? This is specifically needed when the target image was computed
                                     by warping another image with a synthetic transformation.
                                     It can have many black borders due to the warp. That can cause instability
                                     during training, if the loss is applied also in the black areas (where the network
                                     cannot infer any correct predictions).
            sparse_ground_truth: is ground-truth sparse? Important for downscaling/upscaling of the flow field
            mapping: load correspondence map instead of flow field?
        """

        assert not (apply_mask and apply_mask_zero_borders), \
            'apply_mask and apply_mask_zero_borders cannot both be applied at the same time, choose only one'

        self.apply_mask = apply_mask
        self.apply_mask_zero_borders = apply_mask_zero_borders
        self.sparse_ground_truth = sparse_ground_truth

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.mapping = mapping

    def __call__(self, mini_batch, *args, **kwargs):
        """
        args:
            mini_batch: The input data, should contain the following fields:
                        'source_image', 'target_image', 'correspondence_mask'
                        'flow_map' if self.mapping is False else 'correspondence_map_pyro'
                        'mask_zero_borders' if self.apply_mask_zero_borders
        returns:
            TensorDict: output data block with following fields:
                        'source_image', 'target_image', 'source_image_256', 'target_image_256', flow_map',
                        'flow_map_256', 'mask', 'mask_256', 'correspondence_mask'

                        'flow_map' is the ground-truth flow relating the target to the source. 'mask' is the mask
                        where the loss will be applied (in coordinate system of the target).
                        Similar for the 256x256 tensors.

        """
        source_image, source_image_256 = pre_process_image_glunet(mini_batch['source_image'], self.device)
        target_image, target_image_256 = pre_process_image_glunet(mini_batch['target_image'], self.device)

        # At original resolution
        if self.sparse_ground_truth:
            flow_gt_original = mini_batch['flow_map'][0].to(self.device)
            flow_gt_256 = mini_batch['flow_map'][1].to(self.device)
            if flow_gt_original.shape[1] != 2:
                # shape is bxhxwx2
                flow_gt_original = flow_gt_original.permute(0, 3, 1, 2)
            if flow_gt_256.shape[1] != 2:
                # shape is bxhxwx2
                flow_gt_256 = flow_gt_256.permute(0, 3, 1, 2)
        else:
            if self.mapping:
                mapping_gt_original = mini_batch['correspondence_map_pyro'][-1].to(self.device)
                flow_gt_original = unormalise_and_convert_mapping_to_flow(mapping_gt_original.permute(0,3,1,2))
            else:
                flow_gt_original = mini_batch['flow_map'].to(self.device)
            if flow_gt_original.shape[1] != 2:
                # shape is bxhxwx2
                flow_gt_original = flow_gt_original.permute(0, 3, 1, 2)
            bs, _, h_original, w_original = flow_gt_original.shape

            # now we have flow everywhere, at 256x256 resolution, b, _, 256, 256
            flow_gt_256 = F.interpolate(flow_gt_original, (256, 256), mode='bilinear', align_corners=False)
            flow_gt_256[:, 0, :, :] *= 256.0/float(w_original)
            flow_gt_256[:, 1, :, :] *= 256.0/float(h_original)

        bs, _, h_original, w_original = flow_gt_original.shape
        bs, _, h_256, w_256 = flow_gt_256.shape

        # defines the mask to use during training
        mask = None
        mask_256 = None
        if self.apply_mask_zero_borders:
            assert 'mask_zero_borders' in mini_batch
            # make mask to remove all black borders
            mask = mini_batch['mask_zero_borders'].to(self.device)  # bxhxw, torch.uint8

            mask_256 = F.interpolate(mask.unsqueeze(1).float(), (256, 256), mode='bilinear',
                                     align_corners=False).squeeze(1).floor()  # bx256x256, flooring
            mask_256 = mask_256.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask_256.byte()
        elif self.apply_mask:
            if self.sparse_ground_truth:
                mask = mini_batch['correspondence_mask'][0].to(self.device)
                mask_256 = mini_batch['correspondence_mask'][1].to(self.device)
            else:
                mask = mini_batch['correspondence_mask'].to(self.device)  # bxhxw, torch.uint8
                mask_256 = F.interpolate(mask.unsqueeze(1).float(), (256, 256), mode='bilinear',
                                         align_corners=False).squeeze(1).floor()   # bx256x256, flooring
                mask_256 = mask_256.bool() if version.parse(torch.__version__) >= version.parse("1.1") \
                    else mask_256.byte()

        mini_batch['source_image'] = source_image
        mini_batch['target_image'] = target_image
        mini_batch['source_image_256'] = source_image_256
        mini_batch['target_image_256'] = target_image_256
        mini_batch['flow_map'] = flow_gt_original
        mini_batch['flow_map_256'] = flow_gt_256
        mini_batch['mask'] = mask
        mini_batch['mask_256'] = mask_256
        if self.sparse_ground_truth:
            mini_batch['correspondence_mask'] = mini_batch['correspondence_mask'][0].to(self.device)
        else:
            mini_batch['correspondence_mask'] = mini_batch['correspondence_mask'].to(self.device)
        return mini_batch


class GLOCALNetBatchPreprocessing:
    """  Class responsible for processing the mini-batch to create the desired training inputs for GLOCALNet/BaseNet
    based networks.
    """

    def __init__(self, settings, apply_mask=False, apply_mask_zero_borders=False, mapping=False):
        """
        Args:
            settings: settings
            apply_mask: apply ground-truth correspondence mask for loss computation?
            apply_mask_zero_borders: apply mask zero borders (equal to 0 at black borders in target image) for loss
                                     computation? This is specifically needed when the target image was computed
                                     by warping another image with a synthetic transformation.
                                     It can have many black borders due to the warp. That can cause instability
                                     during training, if the loss is applied also in the black areas (where the network
                                     cannot infer any correct predictions).
            mapping: load correspondence map instead of flow field?
        """

        assert not (apply_mask and apply_mask_zero_borders), \
            'apply_mask and apply_mask_zero_borders cannot both be applied at the same time, choose only one'
        self.apply_mask = apply_mask
        self.apply_mask_zero_borders = apply_mask_zero_borders

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.mapping = mapping

    def __call__(self, mini_batch, *args, **kwargs):
        """
        args:
            mini_batch: The input data, should contain the following fields:
                        'source_image', 'target_image', 'correspondence_mask'
                        'flow_map' if self.mapping is False else 'correspondence_map_pyro'
                        'mask_zero_borders' if self.apply_mask_zero_borders
        returns:
            TensorDict: output data block with following fields:
                        'source_image', 'target_image', 'flow_map','mask',  'correspondence_mask'

                        'flow_map' is the ground-truth flow relating the target to the source. 'mask' is the mask
                        where the loss will be applied (in coordinate system of the target)
        """

        if self.mapping:
            mapping_gt = mini_batch['correspondence_map_pyro'][-1].to(self.device)
            flow_gt = unormalise_and_convert_mapping_to_flow(mapping_gt.permute(0, 3, 1, 2))
        else:
            flow_gt = mini_batch['flow_map'].to(self.device)
        if flow_gt.shape[1] != 2:
            flow_gt.permute(0, 3, 1, 2)
        bs, _, h, w = flow_gt.shape

        # BaseNet/GLOCALNet input needs to be 256x256
        assert h == 256 and w == 256

        mask = None
        if self.apply_mask_zero_borders:
            assert 'mask_zero_borders' in mini_batch
            # make mask to remove all black borders
            mask = mini_batch['mask_zero_borders'].to(self.device)  # bxhxw, torch.uint8
        elif self.apply_mask:
            mask = mini_batch['correspondence_mask'].to(self.device)  # bxhxw, torch.uint8
        if mask is not None and (mask.shape[1] != h or mask.shape[2] != w):
            # mask_gt does not have the proper shape
            mask = F.interpolate(mask.float().unsqueeze(1), (h, w), mode='bilinear',
                                 align_corners=False).squeeze(1).floor()  # bxhxw
            mask = mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask.byte()

        mini_batch['source_image'] = normalize_image_with_imagenet_weights(mini_batch['source_image']).to(self.device)
        mini_batch['target_image'] = normalize_image_with_imagenet_weights(mini_batch['target_image']).to(self.device)
        mini_batch['mask'] = mask
        mini_batch['flow_map'] = flow_gt
        mini_batch['correspondence_mask'] = mini_batch['correspondence_mask'].to(self.device)
        return mini_batch


class ImageNetNormalizationBatchPreprocessing:
    """  Class responsible for processing the mini-batch and particularly for normalizing the images with ImageNet
    weights.
    """

    def __init__(self, settings, apply_mask=False, apply_mask_zero_borders=False, mapping=False):
        """
        Args:
            settings: settings
            apply_mask: apply ground-truth correspondence mask for loss computation?
            apply_mask_zero_borders: apply mask zero borders (equal to 0 at black borders in target image) for loss
                                     computation? This is specifically needed when the target image was computed
                                     by warping another image with a synthetic transformation.
                                     It can have many black borders due to the warp. That can cause instability
                                     during training, if the loss is applied also in the black areas (where the network
                                     cannot infer any correct predictions).
            mapping: load correspondence map instead of flow field?
        """
        assert not (apply_mask and apply_mask_zero_borders), \
            'apply_mask and apply_mask_zero_borders cannot both be applied at the same time, choose only one'

        self.apply_mask = apply_mask
        self.apply_mask_zero_borders = apply_mask_zero_borders
        self.mapping = mapping

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu")

    def __call__(self, mini_batch, *args, **kwargs):
        """
        args:
            mini_batch: The input data, should contain the following fields:
                        'source_image', 'target_image', 'correspondence_mask'
                        'flow_map' if self.mapping is False else 'correspondence_map_pyro'
                        'mask_zero_borders' if self.apply_mask_zero_borders
        returns:
            TensorDict: output data block with following fields:
                        'source_image', 'target_image', 'flow_map','mask',  'correspondence_mask'

                        'flow_map' is the ground-truth flow relating the target to the source. 'mask' is the mask
                        where the loss will be applied (in coordinate system of the target)
        """

        if self.mapping:
            mapping_gt = mini_batch['correspondence_map_pyro'][-1].to(self.device)
            flow_gt = unormalise_and_convert_mapping_to_flow(mapping_gt.permute(0, 3, 1, 2))
        else:
            flow_gt = mini_batch['flow_map'].to(self.device)
        if flow_gt.shape[1] != 2:
            flow_gt.permute(0, 3, 1, 2)
        bs, _, h, w = flow_gt.shape

        mask = None
        if self.apply_mask_zero_borders:
            assert 'mask_zero_borders' in mini_batch

            # make mask to remove all black borders
            mask = mini_batch['mask_zero_borders'].to(self.device)  # bxhxw, torch.uint8
        elif self.apply_mask:
            mask = mini_batch['correspondence_mask'].to(self.device)  # bxhxw, torch.uint8
        if mask is not None and (mask.shape[1] != h or mask.shape[2] != w):
            # mask_gt does not have the proper shape
            mask = F.interpolate(mask.float().unsqueeze(1), (h, w), mode='bilinear',
                                 align_corners=False).squeeze(1).floor()  # bxhxw
            mask = mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask.byte()

        mini_batch['source_image'] = normalize_image_with_imagenet_weights(mini_batch['source_image']).to(self.device)
        mini_batch['target_image'] = normalize_image_with_imagenet_weights(mini_batch['target_image']).to(self.device)
        mini_batch['mask'] = mask
        mini_batch['flow_map'] = flow_gt
        mini_batch['correspondence_mask'] = mini_batch['correspondence_mask'].to(self.device)
        return mini_batch
