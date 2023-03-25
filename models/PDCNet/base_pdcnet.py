import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from packaging import version


from models.base_matching_net import BaseGLUMultiScaleMatchingNet
from models.inference_utils import estimate_homography_and_inliers, estimate_homography_and_correspondence_map, \
    estimate_mask, matches_from_flow, from_homography_to_pixel_wise_mapping
from models.PDCNet.mod_uncertainty import estimate_probability_of_confidence_interval_of_mixture_density, \
    estimate_average_variance_of_mixture_density, estimate_probability_of_confidence_interval_of_unimodal_density
from models.modules.local_correlation import correlation
from utils_flow.pixel_wise_mapping import warp, warp_with_mapping
from utils_flow.flow_and_mapping_operations import convert_mapping_to_flow, convert_flow_to_mapping


def pad_to_size(im, size):
    # size first h then w
    if not isinstance(size, tuple):
        size = (size, size)
    # pad to same shape
    if im.shape[0] < size[0]:
        pad_y_1 = size[0] - im.shape[0]
    else:
        pad_y_1 = 0
    if im.shape[1] < size[1]:
        pad_x_1 = size[1] - im.shape[1]
    else:
        pad_x_1 = 0

    im = cv2.copyMakeBorder(im, 0, pad_y_1, 0, pad_x_1, cv2.BORDER_CONSTANT)
    return im


class UncertaintyPredictionInference(nn.Module):
    """
    Common to all uncertainty predicting architectures
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.estimate_uncertainty = True
        inference_parameters_default = {'R': 1.0, 'ransac_thresh': 1.0, 'multi_stage_type': 'direct',
                                        'mask_type': 'proba_interval_1_above_5',
                                        # for multi-scale
                                        'homography_visibility_mask': True,
                                        'list_resizing_ratios': [0.5, 0.6, 0.88, 1, 1.33, 1.66, 2],
                                        # [1, 0.5, 0.88, 1.0 / 0.5, 1.0 / 0.88],
                                        'min_inlier_threshold_for_multi_scale': 0.2,
                                        'min_nbr_points_for_multi_scale': 70,
                                        'compute_cyclic_consistency_error': False}
        self.inference_parameters = inference_parameters_default

    def set_inference_parameters(self, confidence_R=1.0,
                                 ransac_thresh=1.0, multi_stage_type='direct',
                                 mask_type_for_2_stage_alignment='proba_interval_1_above_5',
                                 homography_visibility_mask=True,
                                 list_resizing_ratios=[0.5, 0.6, 0.88, 1, 1.33, 1.66, 2],
                                 min_inlier_threshold_for_multi_scale=0.2, min_nbr_points_for_multi_scale=70,
                                 compute_cyclic_consistency_error=False):
        """Sets the inference parameters required for PDCNet.
        inference_parameters_default = {'R': 1.0, 'ransac_thresh': 1.0, 'multi_stage_type': 'direct',
                                        'mask_type': 'proba_interval_1_above_5',
                                        # for multi-scale
                                        'homography_visibility_mask': True,
                                        'list_resizing_ratios': [0.5, 0.6, 0.88, 1, 1.33, 1.66, 2],
                                        'min_inlier_threshold_for_multi_scale': 0.2,
                                        'min_nbr_points_for_multi_scale': 70,
                                        'compute_cyclic_consistency_error': False}

        """
        inference_parameters = {'R': confidence_R, 'ransac_thresh': ransac_thresh, 'multi_stage_type': multi_stage_type,
                                'mask_type': mask_type_for_2_stage_alignment,
                                'homography_visibility_mask': homography_visibility_mask,
                                'list_resizing_ratios': list_resizing_ratios,
                                'min_inlier_threshold_for_multi_scale': min_inlier_threshold_for_multi_scale,
                                'min_nbr_points_for_multi_scale': min_nbr_points_for_multi_scale,
                                'compute_cyclic_consistency_error': compute_cyclic_consistency_error
                                }
        self.inference_parameters = inference_parameters

    def use_global_corr_layer(self, c_target, c_source):
        """
        Computes global correlation from target and source feature maps.
        similar to DGC-Net, usually features are first normalized with L2 norm and the output cost volume is
        relued, followed by L2 norm.
        Args:
            c_target: B, c, h_t, w_t
            c_source: B, c, h_s, w_s

        Returns:
            input_corr_uncertainty_dec: B, h_s*w_s, h_t, w_t
        """
        if self.params.normalize_features:
            corr_uncertainty = self.corr_module_for_corr_uncertainty_decoder(self.l2norm(c_source),
                                                                             self.l2norm(c_target))
        else:
            corr_uncertainty = self.corr_module_for_corr_uncertainty_decoder(c_source, c_target)
        input_corr_uncertainty_dec = self.l2norm(F.relu(corr_uncertainty))
        return input_corr_uncertainty_dec

    def use_local_corr_layer(self, c_target, c_source):
        """
        Computes local correlation from target and source feature maps.
        similar to PWC-Net, usually features are not normalized with L2 norm and the output cost volume is
        processed with leaky-relu.
        Args:
            c_target: B, c, h_t, w_t
            c_source: B, c, h_s, w_s

        Returns:
            input_corr_uncertainty_dec: B, h_s*w_s, h_t, w_t
        """
        input_corr_uncertainty_dec = correlation.FunctionCorrelation(reference_features=c_target, query_features=c_source)
        input_corr_uncertainty_dec = self.leakyRELU(input_corr_uncertainty_dec)
        return input_corr_uncertainty_dec

    @staticmethod
    def constrain_large_log_var_map(var_min, var_max, large_log_var_map):
        """
        Constrains variance parameter between var_min and var_max, returns log of the variance. Here large_log_var_map
        is the unconstrained variance, outputted by the network
        Args:
            var_min: min variance, corresponds to parameter beta_minus in paper
            var_max: max variance, corresponds to parameter beta_plus in paper
            large_log_var_map: value to constrain

        Returns:
            larger_log_var_map: log of variance parameter
        """
        if var_min > 0 and var_max > 0:
            large_log_var_map = torch.log(var_min +
                                          (var_max - var_min) * torch.sigmoid(large_log_var_map - torch.log(var_max)))
        elif var_max > 0:
            large_log_var_map = torch.log((var_max - var_min) * torch.sigmoid(large_log_var_map - torch.log(var_max)))
        elif var_min > 0:
            # large_log_var_map = torch.log(var_min + torch.exp(large_log_var_map))
            max_exp = large_log_var_map.detach().max() - 10.0
            large_log_var_map = torch.log(var_min / max_exp.exp() + torch.exp(large_log_var_map - max_exp)) + max_exp
        return large_log_var_map

    def estimate_flow_and_confidence_map(self, source_img, target_img, output_shape=None,
                                         scaling=1.0, mode='channel_first'):
        """
        Returns the flow field and corresponding confidence map relating the target to the source image.
        Returned flow has output_shape if provided, otherwise the same dimension than the target image.
        If scaling is provided, the output shape is the target image dimension multiplied by this scaling factor.
        Args:
            source_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            target_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            output_shape: int or list of int, or None, output shape of the returned flow field
            scaling: float, scaling factor applied to target image shape, to obtain the outputted flow field dimensions
                     if output_shape is None
            mode: if channel_first, flow has shape b, 2, H, W. Else shape is b, H, W, 2

        Returns:
            flow_est: estimated flow field relating the target to the reference image, resized and scaled to
                      output_shape (can be defined by scaling parameter)
            uncertainty_est: dict with keys 'p_r', 'inference_parameters', 'variance' and
                            'cyclic_consistency_error' if self.inference_parameters['compute_cyclic_consistency_error']
                             is True.
                             if multimodal density, also contains fields 'log_var_map', 'weight_map' with
                             shapes (B, K, H, W)
                             Otherwise (single mode), contains 'log_var_map', with shape (B, 1, H, W)


        """
        flow_est, uncertainty_est = self.estimate_flow_and_confidence_map_(source_img, target_img, output_shape,
                                                                           scaling, mode)

        if self.inference_parameters['compute_cyclic_consistency_error']:
            flow_est_backward, uncertainty_est_backward = self.estimate_flow_and_confidence_map_(
                target_img, source_img, output_shape, scaling, mode)
            cyclic_consistency_error = torch.norm(flow_est + self.warp(flow_est_backward, flow_est), dim=1,
                                                  keepdim=True)
            uncertainty_est['cyclic_consistency_error'] = cyclic_consistency_error
            uncertainty_est['inv_cyclic_consistency_error'] = 1.0 / (1.0 + cyclic_consistency_error)
        return flow_est, uncertainty_est

    def estimate_flow_and_confidence_map_(self, source_img, target_img, output_shape=None,
                                          scaling=1.0, mode='channel_first'):
        # will be chosen when defining the network by calling 'set_inference_parameters()'
        inference_parameters = self.inference_parameters

        # define output_shape
        if output_shape is None and scaling != 1.0:
            b, _, h_ori, w_ori = target_img.shape
            output_shape = (int(h_ori * scaling), int(w_ori * scaling))

        inference_type = inference_parameters['multi_stage_type']
        if inference_type == 'direct' or inference_type.lower() == 'd':
            return self.estimate_flow_and_confidence_map_direct(source_img, target_img, inference_parameters,
                                                                output_shape=output_shape, mode=mode)

        elif inference_type == 'homography_from_last_level_uncertainty':
            return self.estimate_flow_and_confidence_map_with_homo(source_img, target_img, inference_parameters,
                                                                   scaling=1.0, output_shape=output_shape, mode=mode)

        elif inference_type == 'homography_from_quarter_resolution_uncertainty' or inference_type.lower() == 'h':
            return self.estimate_flow_and_confidence_map_with_homo(source_img, target_img, inference_parameters,
                                                                   scaling=1.0 / 4.0, output_shape=output_shape,
                                                                   mode=mode)

        elif inference_type == 'homography_from_L_Net_upsampled_to_quarter_reso':
            raise NotImplementedError

        elif inference_type == 'multiscale_homo_from_quarter_resolution_uncertainty' or inference_type.lower() == 'ms':
            return self.estimate_flow_and_confidence_map_with_multiscale(source_img, target_img, inference_parameters,
                                                                         scaling=1.0 / 4.0, output_shape=output_shape,
                                                                         mode=mode)
        else:
            raise NotImplementedError

    def estimate_flow_and_confidence_map_with_homo(self, source_img, target_img, inference_parameters,
                                                   inter_shape=None, scaling=1.0,
                                                   output_shape=None, mode='channel_first'):
        """
        Returns the flow field and corresponding confidence map relating the target to the source image, using the
        PDCNet multi-stage approach.
        Returned flow has output_shape if provided, otherwise the same dimension than the target image.
        If scaling is provided, the output shape is the target image dimension multiplied by this scaling factor.
        Args:
            source_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            target_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            inference_parameters: dict with inference parameters
                                  inference_parameters_default =
                                  {'R': 1.0, 'ransac_thresh': 1.0, 'multi_stage_type': 'direct',
                                  'mask_type': 'proba_interval_1_above_5', 'homography_visibility_mask': True,
                                  'list_resizing_ratios': [0.5, 0.6, 0.88, 1, 1.33, 1.66, 2],
                                  'min_inlier_threshold_for_multi_scale': 0.2, 'min_nbr_points_for_multi_scale': 70,
                                  'compute_cyclic_consistency_error': False}
            inter_shape: list of int, shape of outputted flow for homography computation. If None, use target image
                         resolution
            output_shape: int or list of int, or None, output shape of the returned flow field
            scaling: float, scaling factor applied to target image shape, to obtain the outputted flow field dimensions
                     if output_shape is None
            mode: if channel_first, flow has shape b, 2, H, W. Else shape is b, H, W, 2

        Returns:
            flow_est: estimated flow field relating the target to the reference image, resized and scaled to
                      output_shape (can be defined by scaling parameter)
            uncertainty_est: dict with keys 'p_r', 'inference_parameters', 'variance'
                             if multimodal density, also contains fields 'log_var_map', 'weight_map' with
                             shapes (B, K, H, W)
                             Otherwise (single mode), contains 'log_var_map', with shape (B, 1, H, W)
        """
        b, _, h_ori, w_ori = target_img.shape
        image_shape = (h_ori, w_ori)
        if output_shape is None:
            output_shape = image_shape

        # inter shape is the shape of output flow from first forward pass, before computing the homography
        if inter_shape is None:
            inter_shape = [int(image_shape_ * scaling) for image_shape_ in image_shape]
        flow_est, uncertainty_est = self.estimate_flow_and_confidence_map_direct(source_img, target_img,
                                                                                 inference_parameters,
                                                                                 output_shape=inter_shape)

        # do multi-stage by estimating homography from confident matches
        mask_pre = estimate_mask(inference_parameters['mask_type'], uncertainty_est)
        H_image_size, mapping_from_homography = estimate_homography_and_correspondence_map(
            flow_est, mask_pre, original_shape=image_shape, mapping_output_shape=output_shape,
            scaling=np.float32(image_shape)[::-1] / np.float32(inter_shape)[::-1],
            # scaling in horizontal then vertical direction
            ransac_thresh=inference_parameters['ransac_thresh'], min_nbr_points=200)

        if mapping_from_homography is not None:
            flow_est_first = self.resize_and_rescale_flow(flow_est, output_shape)

            Is_remapped_with_homo = cv2.warpPerspective(source_img.squeeze().permute(1, 2, 0).cpu().numpy(),
                                                        H_image_size, image_shape[::-1])
            source_img = torch.Tensor(Is_remapped_with_homo).permute(2, 0, 1).unsqueeze(0)
            # warper = tgm.HomographyWarper(h_ori, w_ori)
            # source_img = warper(source_img, torch.from_numpy(H_image_size).view(1,3,3))  # NxCxHxW

            flow_est_second, uncertainty_est = self.estimate_flow_and_confidence_map_direct(source_img, target_img,
                                                                                            inference_parameters,
                                                                                            output_shape=output_shape,
                                                                                            mode=mode)

            # final flow is composition of mapping from homography and flow_est_second
            mapping_estimated_final = warp(mapping_from_homography.to(self.device), flow_est_second)
            flow_est = convert_mapping_to_flow(mapping_estimated_final)

            # mask = warp_with_mapping(torch.ones(output_shape).unsqueeze(0).unsqueeze(0).float().to(self.device),
            #                          mapping_estimated_final).ge(0.99).squeeze(1)
            mask = warp(torch.ones(output_shape).unsqueeze(0).unsqueeze(0).float().to(self.device),
                        flow_est_second).ge(0.98).squeeze(1)
            warping_mask = warp_with_mapping(torch.ones(output_shape).unsqueeze(0).unsqueeze(0).float().to(self.device),
                                             mapping_from_homography).ge(0.98).squeeze(1)
            if inference_parameters['homography_visibility_mask']:
                mask = mask * warping_mask
            uncertainty_est['warping_mask'] = mask

            flow_est = flow_est * mask.float() * warping_mask.float() + \
                       flow_est_first * (~(mask & warping_mask)).float()

        else:
            if inter_shape[0] != output_shape[0] or inter_shape[1] != output_shape[1]:
                # recompute so the output is the proper size
                flow_est, uncertainty_est = self.estimate_flow_and_confidence_map_direct(source_img, target_img,
                                                                                         inference_parameters,
                                                                                         output_shape=output_shape)
        if mode == 'channel_first':
            return flow_est, uncertainty_est
        else:
            return flow_est.permute(0, 2, 3, 1), uncertainty_est

    def estimate_flow_and_confidence_map_with_multiscale(self, source_img, target_img, inference_parameters,
                                                         scaling=1.0, output_shape=None, mode='channel_first'):
        """
        Returns the flow field and corresponding confidence map relating the target to the source image, using the
        PDCNet multi-scale approach.
        Returned flow has output_shape if provided, otherwise the same dimension than the target image.
        If scaling is provided, the output shape is the target image dimension multiplied by this scaling factor.
        Args:
            source_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            target_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            inference_parameters: dict with inference parameters
                                  inference_parameters_default =
                                  {'R': 1.0, 'ransac_thresh': 1.0, 'multi_stage_type': 'direct',
                                  'mask_type': 'proba_interval_1_above_5', 'homography_visibility_mask': True,
                                  'list_resizing_ratios': [0.5, 0.6, 0.88, 1, 1.33, 1.66, 2],
                                  'min_inlier_threshold_for_multi_scale': 0.2, 'min_nbr_points_for_multi_scale': 70,
                                  'compute_cyclic_consistency_error': False}
            output_shape: int or list of int, or None, output shape of the returned flow field
            scaling: float, scaling factor applied to target image shape, to obtain the outputted flow field dimensions
                     if output_shape is None
            mode: if channel_first, flow has shape b, 2, H, W. Else shape is b, H, W, 2

        Returns:
            flow_est: estimated flow field relating the target to the reference image, resized and scaled to
                      output_shape (can be defined by scaling parameter)
            uncertainty_est: dict with keys 'p_r', 'inference_parameters', 'variance'
                             if multimodal density, also contains fields 'log_var_map', 'weight_map' with
                             shapes (B, K, H, W)
                             Otherwise (single mode), contains 'log_var_map', with shape (B, 1, H, W)
        """

        b, _, h_ori, w_ori = target_img.shape
        image_shape = (h_ori, w_ori)
        if output_shape is None:
            output_shape = image_shape

        resizing_ratio_list = inference_parameters['list_resizing_ratios']
        H_image_size, mapping_from_homography, flow_est_first = self.estimate_homo_through_multiscale(
            source_img, target_img, resizing_factor_for_inter_flow=scaling,
            resizing_ratio_list=resizing_ratio_list,
            min_nbr_points=inference_parameters['min_nbr_points_for_multi_scale'],
            output_shape=output_shape, inference_parameters=inference_parameters,
            min_inlier_threshold=inference_parameters['min_inlier_threshold_for_multi_scale'])
        # output shape must be the size of the flow, while H_image_size corresponds to the image_size

        if mapping_from_homography is not None:
            flow_est_first = self.resize_and_rescale_flow(flow_est_first, output_shape)

            Is_remapped_with_homo = cv2.warpPerspective(source_img.squeeze().permute(1, 2, 0).cpu().numpy(),
                                                        H_image_size, image_shape[::-1])
            source_img = torch.Tensor(Is_remapped_with_homo).permute(2, 0, 1).unsqueeze(0)

            flow_est_second, uncertainty_est = self.estimate_flow_and_confidence_map_direct(source_img, target_img,
                                                                                            inference_parameters,
                                                                                            output_shape=output_shape,
                                                                                            mode=mode)

            # final flow is composition of mapping from homography and flow_est_second
            mapping_estimated_final = warp(mapping_from_homography.to(self.device), flow_est_second)
            flow_est = convert_mapping_to_flow(mapping_estimated_final)

            # mask = warp_with_mapping(torch.ones(output_shape).unsqueeze(0).unsqueeze(0).float().to(self.device),
            #                          mapping_estimated_final).ge(0.99).squeeze(1)
            mask = warp(torch.ones(output_shape).unsqueeze(0).unsqueeze(0).float().to(self.device),
                        flow_est_second).ge(0.98).squeeze(1)
            warping_mask = warp_with_mapping(torch.ones(output_shape).unsqueeze(0).unsqueeze(0).float().to(self.device),
                                             mapping_from_homography).ge(0.98).squeeze(1)
            if inference_parameters['homography_visibility_mask']:
                mask = mask * warping_mask
            uncertainty_est['warping_mask'] = mask

            flow_est = flow_est * mask.float() * warping_mask.float() + flow_est_first * (~(mask & warping_mask)).float()

            if mode == 'channel_first':
                return flow_est, uncertainty_est
            else:
                return flow_est.permute(0, 2, 3, 1), uncertainty_est
        else:
            return self.estimate_flow_and_confidence_map_direct(source_img, target_img, inference_parameters,
                                                                output_shape=output_shape, mode=mode)

    def estimate_homo_through_multiscale(self, image_source_original_padded_torch, image_target_original_padded_torch,
                                         resizing_factor_for_inter_flow,
                                         output_shape, inference_parameters, resizing_ratio_list, min_nbr_points=100,
                                         min_inlier_threshold=0.6):

        image_source_original_padded = image_source_original_padded_torch.permute(0, 2, 3, 1)[0].cpu().numpy()
        image_target_original_padded = image_target_original_padded_torch.permute(0, 2, 3, 1)[0].cpu().numpy()
        h_t, w_t = image_target_original_padded.shape[:2]
        h_s, w_s = image_source_original_padded.shape[:2]

        list_of_H_source = []
        list_of_H_target = []
        list_of_normalization_value = []
        list_of_padded_source_images = []
        list_of_padded_target_images = []

        inter_shape = [int(image_shape_ * resizing_factor_for_inter_flow) for image_shape_
                       in [h_t, w_t]]
        # compute scaling, in case the original size was not dividable by the original scaling parameter
        # first dimension is horizontal, then vertical.
        scaling = np.float32([h_t, w_t][::-1]) / np.float32(inter_shape[::-1])

        if 1.0 not in resizing_ratio_list:
            resizing_ratio_list.append(1.0)

        index_of_original_resolution = resizing_ratio_list.index(1.0)

        for ratio in resizing_ratio_list:
            if ratio == 1.0:
                list_of_H_target.append(np.eye(3))
                list_of_H_source.append(np.eye(3))
                list_of_normalization_value.append(float(h_t * w_t / scaling[0] * scaling[1]))
                list_of_padded_target_images.append(np.expand_dims(image_target_original_padded, 0))
                list_of_padded_source_images.append(np.expand_dims(image_source_original_padded, 0))
            elif ratio < 1.0:
                # resize the target image
                h_resized, w_resized = int(h_t * ratio), int(w_t * ratio)
                ratio_h = float(h_resized) / float(h_t)
                ratio_w = float(w_resized) / float(w_t)
                H_target_resized = np.array([[ratio_w, 0, 0], [0, ratio_h, 0], [0, 0, 1]])
                list_of_H_target.append(H_target_resized)
                list_of_H_source.append(np.eye(3))
                list_of_normalization_value.append(float(w_resized * h_resized / scaling[0] * scaling[1]))

                image_target_resized = cv2.warpPerspective(image_target_original_padded, H_target_resized,
                                                           (w_resized, h_resized))
                image_target_resized_padded = pad_to_size(image_target_resized, (h_t, w_t))
                list_of_padded_target_images.append(np.expand_dims(image_target_resized_padded, 0))
                list_of_padded_source_images.append(np.expand_dims(image_source_original_padded, 0))

            else:
                ratio = 1.0 / ratio  # for the source image
                # resize the source image
                h_resized, w_resized = int(h_s * ratio), int(w_s * ratio)
                ratio_h = float(h_resized) / float(h_s)
                ratio_w = float(w_resized) / float(w_s)
                H_source_resized = np.array([[ratio_w, 0, 0], [0, ratio_h, 0], [0, 0, 1]])
                list_of_H_source.append(H_source_resized)
                list_of_H_target.append(np.eye(3))
                list_of_normalization_value.append(float(h_t * w_t / scaling[0] * scaling[1]))

                image_source_resized = cv2.warpPerspective(image_source_original_padded, H_source_resized,
                                                           (w_resized, h_resized))
                image_source_resized_padded = pad_to_size(image_source_resized, (h_t, w_t))
                list_of_padded_target_images.append(np.expand_dims(image_target_original_padded, 0))
                list_of_padded_source_images.append(np.expand_dims(image_source_resized_padded, 0))
                # resize the target

        target_images = np.concatenate(list_of_padded_target_images, axis=0)
        source_images = np.concatenate(list_of_padded_source_images, axis=0)

        target_images_torch = torch.Tensor(target_images).permute(0, 3, 1, 2)
        source_images_torch = torch.Tensor(source_images).permute(0, 3, 1, 2)

        flow_est_pre, uncertainty_est_pre = self.estimate_flow_and_confidence_map_direct(source_images_torch,
                                                                                         target_images_torch,
                                                                                         inference_parameters,
                                                                                         output_shape=inter_shape)

        flow_est_first_original_resolution = flow_est_pre[index_of_original_resolution].unsqueeze(0)

        # do multi-stage by estimating homography from confident matches
        mask_pre = estimate_mask(inference_parameters['mask_type'], uncertainty_est_pre)

        list_H_padded_reso = []
        list_inliers = []
        # then process each image one at a time:
        for ind in range(mask_pre.shape[0]):
            mask_ = mask_pre[ind].unsqueeze(0)
            flow_ = flow_est_pre[ind].unsqueeze(0)
            H, inliers_sum = estimate_homography_and_inliers(flow_, mask_, scaling=scaling,
                                                             min_nbr_points=min_nbr_points)
            if H is not None:
                H_final = np.linalg.inv(list_of_H_target[ind]) @ H @ list_of_H_source[ind]
                list_H_padded_reso.append(H_final)
                list_inliers.append(float(inliers_sum) / list_of_normalization_value[ind])

            else:
                list_H_padded_reso.append(np.eye(3))
                list_inliers.append(0.0)

        H_final = None
        index_max_inlier = np.argmax(list_inliers)
        max_inlier = list_inliers[index_max_inlier] * 100
        if max_inlier > min_inlier_threshold:
            # to remove the completely shitty homographies
            H_final = list_H_padded_reso[index_max_inlier]

        if not np.all(H_final == np.eye(3)) and (H_final is not None):
            mapping_from_homography_x, mapping_from_homography_y = from_homography_to_pixel_wise_mapping(
                (h_t, w_t), np.linalg.inv(H_final))
            mapping_from_homography_numpy = np.dstack((mapping_from_homography_x, mapping_from_homography_y))
            mapping_from_homography_torch = torch.from_numpy(mapping_from_homography_numpy).unsqueeze(0) \
                .permute(0, 3, 1, 2)

            if output_shape is not None:
                mapping_from_homography_torch = torch.nn.functional.interpolate(
                    input=mapping_from_homography_torch.to(self.device),
                    size=output_shape, mode='bilinear',
                    align_corners=False)
                mapping_from_homography_torch[:, 0] *= float(output_shape[1]) / float(w_t)
                mapping_from_homography_torch[:, 1] *= float(output_shape[0]) / float(h_t)
        else:
            mapping_from_homography_torch = None
        return H_final, mapping_from_homography_torch, flow_est_first_original_resolution

    def get_matches_and_confidence(self, source_img, target_img, scaling=1.0/4.0,
                                   confident_mask_type='proba_interval_1_above_10', min_number_of_pts=200):
        """
        Computes matches and corresponding confidence value.
        Confidence value is obtained with forward-backward cyclic consistency.
        Args:
            source_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            target_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            scaling: float, scaling factor applied to target_img image shape, to obtain the outputted flow field dimensions,
                     where the matches are extracted
            confident_mask_type: default is 'proba_interval_1_above_10' for PDCNet.
                                 See inference_utils/estimate_mask for more details
            min_number_of_pts: below that number, we discard the retrieved matches (little blobs in cyclic
                               consistency mask)


        Returns:
            dict with keys 'kp_source', 'kp_target', 'confidence_value', 'flow' and 'mask'
            flow and mask are torch tensors

        """
        flow_estimated, uncertainty_est = self.estimate_flow_and_confidence_map(
            source_img, target_img, scaling=scaling)

        mask = estimate_mask(confident_mask_type, uncertainty_est)
        if 'warping_mask' in list(uncertainty_est.keys()):
            # get mask from internal multi stage alignment, if it took place
            mask = mask * uncertainty_est['warping_mask']
        mapping_estimated = convert_flow_to_mapping(flow_estimated)
        # remove point that lead to outside the source_img image
        mask = mask & mapping_estimated[:, 0].ge(0) & mapping_estimated[:, 1].ge(0) & \
            mapping_estimated[:, 0].le(source_img.shape[-1] * scaling - 1) & \
            mapping_estimated[:, 1].le(source_img.shape[-2] * scaling - 1)

        # get corresponding keypoints
        scaling_kp = np.float32(target_img.shape[-2:]) / np.float32(flow_estimated.shape[-2:])  # h, w
        mkpts_s, mkpts_t = matches_from_flow(flow_estimated, mask, scaling=scaling_kp[::-1])
        confidence_values = uncertainty_est['p_r'].squeeze()[mask.squeeze()].cpu().numpy()
        sort_index = np.argsort(np.array(confidence_values)).tolist()[::-1]  # from highest to smallest
        confidence_values = np.array(confidence_values)[sort_index]
        mkpts_s = np.array(mkpts_s)[sort_index]
        mkpts_t = np.array(mkpts_t)[sort_index]

        if len(mkpts_s) < min_number_of_pts:
            mkpts_s = np.empty([0, 2], dtype=np.float32)
            mkpts_t = np.empty([0, 2], dtype=np.float32)
            confidence_values = np.empty([0], dtype=np.float32)

        pred = {'kp_source': mkpts_s, 'kp_target': mkpts_t, 'confidence_value': confidence_values,
                'flow': self.resize_and_rescale_flow(flow_estimated, target_img.shape[-2:]),
                'mask': F.interpolate(input=mask.unsqueeze(1).float(), size=target_img.shape[-2:], mode='bilinear',
                                      align_corners=False).squeeze(1)}
        return pred

    def perform_matching(self, data_source, data_target, cfg, segNet=None):
        """
        Utils function to get flow and matching confidence mask relating target image to source image.
        Args:
            data_source: dict with keys 'image_original', 'size_original', 'image_resized', 'size_resized',
                         'image_resized_padded', 'size_resized_padded', 'image_resized_padded_torch'
            data_target: dict with keys 'image_original', 'size_original', 'image_resized', 'size_resized',
                         'image_resized_padded', 'size_resized_padded', 'image_resized_padded_torch'
            cfg: config with default
                 {'estimate_at_quarter_resolution: True, 'use_segnet': False,
                  'mask_type_for_pose_estimation': 'proba_interval_1_above_10'}
            segNet: segmentation network initialized. If not used, None

        Returns:
            flow, confidence_map and mask: torch tensors of shapes (b, 2, h, w), (b, h, w) and (b, h, w) respectively

        """
        if cfg.estimate_at_quarter_resolution:
            scaling = 4.0
            size_of_flow_padded = [int(image_shape_ // 4) for image_shape_ in data_target['size_resized_padded']]
            size_of_flow = [int(image_shape_ // 4) for image_shape_ in data_target['size_resized']]
            size_of_source = [int(image_shape_ // 4) for image_shape_ in data_source['size_resized']]
        else:
            scaling = 1.0
            size_of_flow_padded = data_target['size_resized_padded']
            size_of_flow = data_target['size_resized']
            size_of_source = data_source['size_resized']

        target_padded_numpy = data_target['image_resized_padded']
        if cfg.use_segnet:
            mask_building = segNet.getSky(target_padded_numpy)
            mask_padded = torch.from_numpy(mask_building.astype(np.float32)).unsqueeze(0)
            # might have a different shape
            mask_padded = torch.nn.functional.interpolate(input=mask_padded.to(self.device).unsqueeze(1),
                                                          size=size_of_flow_padded, mode='bilinear',
                                                          align_corners=False).squeeze(1).byte()
        else:
            mask_padded = torch.ones(size_of_flow_padded).unsqueeze(0).byte().to(self.device)
        mask_padded = mask_padded.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask_padded.byte()

        # scaling defines the final outputted shape by the network.
        source_padded_torch = data_source['image_resized_padded_torch']
        target_padded_torch = data_target['image_resized_padded_torch']
        flow_estimated_padded, uncertainty_est_padded = self.estimate_flow_and_confidence_map(
            source_padded_torch, target_padded_torch, scaling=1.0 / scaling)

        if 'warping_mask' in list(uncertainty_est_padded.keys()):
            # get mask from internal multi stage alignment, if it took place
            mask_padded = uncertainty_est_padded['warping_mask'] * mask_padded

        # get the mask according to uncertainty estimation
        mask_padded = estimate_mask(cfg.mask_type_for_pose_estimation, uncertainty_est_padded) * mask_padded

        # remove the padding
        flow = flow_estimated_padded[:, :, :size_of_flow[0], :size_of_flow[1]]
        mask = mask_padded[:, :size_of_flow[0], :size_of_flow[1]]
        mapping_estimated = convert_flow_to_mapping(flow)
        # remove point that lead to outside the source image
        mask = mask & mapping_estimated[:, 0].ge(0) & mapping_estimated[:, 1].ge(0) & \
            mapping_estimated[:, 0].le(size_of_source[1] - 1) & mapping_estimated[:, 1].le(size_of_source[0] - 1)

        confidence_map = uncertainty_est_padded['p_r'][:, :, :size_of_flow[0], :size_of_flow[1]].squeeze(1)
        return flow, confidence_map, mask


class ProbabilisticGLU(BaseGLUMultiScaleMatchingNet, UncertaintyPredictionInference):
    """Base class for probabilistic matching networks."""

    def __init__(self, params, pyramid=None, pyramid_256=None, *args, **kwargs):
        super().__init__(params=params, pyramid=pyramid, pyramid_256=pyramid_256, *args, **kwargs)
        self.estimate_one_mode = False  # will be overwritten
        self.laplace_distr = True  # Laplace distributions?

    def estimate_flow_and_confidence_map_direct(self, source_img, target_img, inference_parameters,
                                                output_shape=None, mode='channel_first'):
        """
        Returns the flow field and corresponding confidence map relating the target to the source image, using the
        PDCNet direct approach (single forward pass).
        Returned flow has output_shape if provided, otherwise the same dimension than the target image.
        If scaling is provided, the output shape is the target image dimension multiplied by this scaling factor.
        Args:
            source_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            target_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            inference_parameters: dict with inference parameters
                                  inference_parameters_default =
                                  {'R': 1.0, 'ransac_thresh': 1.0, 'multi_stage_type': 'direct',
                                  'mask_type': 'proba_interval_1_above_5', 'homography_visibility_mask': True,
                                  'list_resizing_ratios': [0.5, 0.6, 0.88, 1, 1.33, 1.66, 2],
                                  'min_inlier_threshold_for_multi_scale': 0.2, 'min_nbr_points_for_multi_scale': 70}
            output_shape: int or list of int, or None, output shape of the returned flow field
            mode: if channel_first, flow has shape b, 2, H, W. Else shape is b, H, W, 2

        Returns:
            flow_est: estimated flow field relating the target to the reference image, resized and scaled to
                      output_shape (can be defined by scaling parameter)
            uncertainty_est: dict with keys 'p_r', 'inference_parameters', 'variance'
                             if multimodal density, also contains fields 'log_var_map', 'weight_map' with
                             shapes (B, K, H, W)
                             Otherwise (single mode), contains 'log_var_map', with shape (B, 1, H, W)
        """
        w_scale = target_img.shape[3]
        h_scale = target_img.shape[2]

        # pre-process the images
        source_img, target_img, source_img_256, target_img_256, ratio_x, ratio_y \
            = self.pre_process_data(source_img, target_img)

        if output_shape is None:
            output_shape = (h_scale, w_scale)
        else:
            ratio_x *= float(output_shape[1]) / float(w_scale)
            ratio_y *= float(output_shape[0]) / float(h_scale)

        flow_est, uncertainty_est = self.compute_flow_and_uncertainty(source_img, target_img,
                                                                      source_img_256, target_img_256,
                                                                      output_shape, inference_parameters,
                                                                      ratio_x=ratio_x, ratio_y=ratio_y)

        if mode == 'channel_first':
            return flow_est, uncertainty_est
        else:
            return flow_est.permute(0, 2, 3, 1), uncertainty_est

    def compute_flow_and_uncertainty(self, source_img, target_img, source_img_256, target_img_256, output_shape,
                                     inference_parameters, ratio_x=1.0, ratio_y=1.0):
        """
        Returns the flow field and uncertainty estimation dictionary relating the target to the source image, using the
        a single forward pass of the network.
        Returned flow has output_shape.
        Args:
            source_img: torch tensor, bx3xHxW (size dividable by 16), normalized with imagenet weights
            target_img: torch tensor, bx3xHxW (size dividable by 16), normalized with imagenet weights
            source_img_256: torch tensor, bx3x256x256, normalized with imagenet weights
            target_img_256: torch tensor, bx3x256x256, normalized with imagenet weights
            output_shape: int or list of int, or None, output shape of the returned flow field
            inference_parameters: dict with inference parameters
                                  inference_parameters_default =
                                  {'R': 1.0, 'ransac_thresh': 1.0, 'multi_stage_type': 'direct',
                                  'mask_type': 'proba_interval_1_above_5', 'homography_visibility_mask': True,
                                  'list_resizing_ratios': [0.5, 0.6, 0.88, 1, 1.33, 1.66, 2],
                                  'min_inlier_threshold_for_multi_scale': 0.2, 'min_nbr_points_for_multi_scale': 70}
            ratio_x: ratio to apply to the horizontal coordinate of the ouputted flow field.
            ratio_y: ratio to apply to the vertical coordinate of the ouputted flow field.

        Returns:
            flow_est: estimated flow field relating the target to the reference image, resized and scaled to
                      output_shape (can be defined by scaling parameter)
            uncertainty_est: dict with keys 'p_r', 'inference_parameters', 'variance'
                             if multimodal density, also contains fields 'log_var_map', 'weight_map' with
                             shapes (B, K, H, W)
                             Otherwise (single mode), contains 'log_var_map', with shape (B, 1, H, W)
        """
        # define output shape and scaling ratios
        output_256, output = self.forward(target_img, source_img, target_img_256, source_img_256)
        flow_est_list = output['flow_estimates']
        flow_est = flow_est_list[-1]
        uncertainty_list = output['uncertainty_estimates'][-1]
        # contains log_var_map and weight_map if multi-modal, only log_var_map if unimodal

        # get the flow field
        flow_est = torch.nn.functional.interpolate(input=flow_est, size=output_shape, mode='bilinear',
                                                   align_corners=False)
        flow_est[:, 0, :, :] *= ratio_x
        flow_est[:, 1, :, :] *= ratio_y

        # get the confidence value
        if self.estimate_one_mode:
            log_var_map = torch.nn.functional.interpolate(input=uncertainty_list, size=output_shape,
                                                          mode='bilinear', align_corners=False)
            p_r = estimate_probability_of_confidence_interval_of_unimodal_density\
                (log_var_map=log_var_map, R=inference_parameters['R'],
                 gaussian=not self.laplace_distr)
            variance = torch.exp(log_var_map)
            uncertainty_est = {'log_var_map': log_var_map}
        else:
            log_var_map = torch.nn.functional.interpolate(input=uncertainty_list[0], size=output_shape,
                                                          mode='bilinear', align_corners=False)
            weight_map = torch.nn.functional.interpolate(input=uncertainty_list[1], size=output_shape,
                                                         mode='bilinear', align_corners=False)
            p_r = estimate_probability_of_confidence_interval_of_mixture_density\
                (weight_map, log_var_map, R=inference_parameters['R'],
                 gaussian=not self.laplace_distr)
            variance = estimate_average_variance_of_mixture_density(weight_map, log_var_map)
            uncertainty_est = {'log_var_map': log_var_map, 'weight_map': weight_map}
        uncertainty_est.update({'p_r': p_r, 'inference_parameters': inference_parameters, 'variance': variance})
        return flow_est, uncertainty_est
