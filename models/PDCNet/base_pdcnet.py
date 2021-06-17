import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from ..base_matching_net import BaseGLUMultiScaleMatchingNet
from .inference_utils import estimate_probability_of_confidence_interval_of_mixture_density, \
    estimate_average_variance_of_mixture_density, estimate_mask, estimate_homography_and_correspondence_map, \
    estimate_homography_and_inliers, matches_from_flow, from_homography_to_pixel_wise_mapping
from ..modules.local_correlation import correlation
from utils_flow.pixel_wise_mapping import warp, warp_with_mapping
from utils_flow.flow_and_mapping_operations import convert_mapping_to_flow
from utils_flow.util import pad_to_size


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
                                        'min_nbr_points_for_multi_scale': 70}
        self.inference_parameters = inference_parameters_default

    def set_inference_parameters(self, confidence_R=1.0, ransac_thresh=1.0, multi_stage_type='direct',
                                 mask_type_for_2_stage_alignment='proba_interval_1_above_5',
                                 homography_visibility_mask=True,
                                 list_resizing_ratios=[0.5, 0.6, 0.88, 1, 1.33, 1.66, 2],
                                 min_inlier_threshold_for_multi_scale=0.2, min_nbr_points_for_multi_scale=70):
        """Sets the inference parameters required for PDCNet.
        Args:
            inference_parameters_default = {'R': 1.0, 'ransac_thresh': 1.0,
                                            'multi_stage_type': 'direct', 'mask_type': 'proba_interval_1_above_5',
                                            'homography_visibility_mask': True,
                                            'list_resizing_ratios': [0.5, 0.6, 0.88, 1, 1.33, 1.66, 2],
                                            'min_inlier_threshold_for_multi_scale': 0.2,
                                            'min_nbr_points_for_multi_scale': 70}
            confidence_R:
            ransac_thresh:
            mask_type_for_2_stage_alignment:
            homography_visibility_mask:
            list_resizing_ratios:
            min_inlier_threshold_for_multi_scale:
            min_nbr_points_for_multi_scale:
        """
        inference_parameters = {'R': confidence_R, 'ransac_thresh': ransac_thresh, 'multi_stage_type': multi_stage_type,
                                'mask_type': mask_type_for_2_stage_alignment,
                                'homography_visibility_mask': homography_visibility_mask,
                                'list_resizing_ratios': list_resizing_ratios,
                                'min_inlier_threshold_for_multi_scale': min_inlier_threshold_for_multi_scale,
                                'min_nbr_points_for_multi_scale': min_nbr_points_for_multi_scale
                                }
        self.inference_parameters = inference_parameters

    def use_global_corr_layer(self, c_target, c_source):
        if self.params.normalize_features:
            corr_uncertainty = self.corr_module_for_corr_uncertainty_decoder(self.l2norm(c_source),
                                                                             self.l2norm(c_target))
        else:
            corr_uncertainty = self.corr_module_for_corr_uncertainty_decoder(c_source, c_target)
        input_corr_uncertainty_dec = self.l2norm(F.relu(corr_uncertainty))
        return input_corr_uncertainty_dec

    def use_local_corr_layer(self, c_t, c_s):
        input_corr_uncertainty_dec = correlation.FunctionCorrelation(reference_features=c_t, query_features=c_s)
        input_corr_uncertainty_dec = self.leakyRELU(input_corr_uncertainty_dec)
        return input_corr_uncertainty_dec

    @staticmethod
    def constrain_large_log_var_map(var_min, var_max, large_log_var_map):
        """
        Constrains variance parameter between var_min and var_max, returns log of the variance. Here large_log_var_map
        if the unconstrained variance, outputted by the network
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
            large_log_var_map = torch.log(var_min + torch.exp(large_log_var_map))
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
            uncertainty_est: dict with keys 'log_var_map', 'weight_map', 'p_r', 'R', 'variance'
        """
        # will be chosen when defining the network by calling 'set_inference_parameters()'
        inference_parameters = self.inference_parameters

        # define output_shape
        if output_shape is None and scaling != 1.0:
            b, _, h_ori, w_ori = target_img.shape
            output_shape = (int(h_ori * scaling), int(w_ori * scaling))

        if inference_parameters['multi_stage_type'] == 'direct':
            return self.estimate_flow_and_confidence_map_direct(source_img, target_img, inference_parameters,
                                                                output_shape=output_shape, mode=mode)

        elif inference_parameters['multi_stage_type'] == 'homography_from_last_level_uncertainty':
            return self.estimate_flow_and_confidence_map_with_homo(source_img, target_img, inference_parameters,
                                                                   scaling=1.0, output_shape=output_shape, mode=mode)

        elif inference_parameters['multi_stage_type'] == 'homography_from_quarter_resolution_uncertainty':
            return self.estimate_flow_and_confidence_map_with_homo(source_img, target_img, inference_parameters,
                                                                   scaling=1.0 / 4.0, output_shape=output_shape,
                                                                   mode=mode)

        elif inference_parameters['multi_stage_type'] == 'homography_from_L_Net_upsampled_to_quarter_reso':
            raise NotImplementedError

        elif inference_parameters['multi_stage_type'] == 'multiscale_homo_from_quarter_resolution_uncertainty':
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
                                  'min_inlier_threshold_for_multi_scale': 0.2, 'min_nbr_points_for_multi_scale': 70}
            inter_shape: list of int, shape of outputted flow for homography computation. If None, use target image
                         resolution
            output_shape: int or list of int, or None, output shape of the returned flow field
            scaling: float, scaling factor applied to target image shape, to obtain the outputted flow field dimensions
                     if output_shape is None
            mode: if channel_first, flow has shape b, 2, H, W. Else shape is b, H, W, 2

        Returns:
            flow_est: estimated flow field relating the target to the reference image, resized and scaled to
                      output_shape (can be defined by scaling parameter)
            uncertainty_est: dict with keys 'log_var_map', 'weight_map', 'p_r', 'R', 'variance'
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
        mask_pre = estimate_mask(inference_parameters['mask_type'], uncertainty_est, list_item=-1)
        H_image_size, mapping_from_homography = estimate_homography_and_correspondence_map(
            flow_est, mask_pre, original_shape=image_shape, mapping_output_shape=output_shape,
            scaling=np.float32(image_shape)[::-1] / np.float32(inter_shape)[::-1],
            # scaling in horizontal then vertical direction
            ransac_thresh=inference_parameters['ransac_thresh'], min_nbr_points=200)

        if mapping_from_homography is not None:
            flow_est_first = self.resize(flow_est, output_shape)

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
                # recompute so the output is the proper load_size
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
                                  'min_inlier_threshold_for_multi_scale': 0.2, 'min_nbr_points_for_multi_scale': 70}
            output_shape: int or list of int, or None, output shape of the returned flow field
            scaling: float, scaling factor applied to target image shape, to obtain the outputted flow field dimensions
                     if output_shape is None
            mode: if channel_first, flow has shape b, 2, H, W. Else shape is b, H, W, 2

        Returns:
            flow_est: estimated flow field relating the target to the reference image, resized and scaled to
                      output_shape (can be defined by scaling parameter)
            uncertainty_est: dict with keys 'log_var_map', 'weight_map', 'p_r', 'R', 'variance'
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
        # output shape must be the load_size of the flow, while H_image_size corresponds to the image_size

        if mapping_from_homography is not None:
            flow_est_first = self.resize(flow_est_first, output_shape)

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

            flow_est = flow_est * mask.float() * warping_mask.float() + \
                       flow_est_first * (~(mask & warping_mask)).float()

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
        mask_pre = estimate_mask(inference_parameters['mask_type'], uncertainty_est_pre, list_item=-1)

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


class ProbabilisticGLU(BaseGLUMultiScaleMatchingNet, UncertaintyPredictionInference):
    """Base class for probabilistic matching networks."""

    def __init__(self, params, pyramid=None, pyramid_256=None, *args, **kwargs):
        super().__init__(params=params, pyramid=pyramid, pyramid_256=pyramid_256, *args, **kwargs)

    def compute_flow_and_uncertainty(self, source_img, target_img, source_img_256, target_img_256, output_shape,
                                     inference_parameters, ratio_x=1.0, ratio_y=1.0, list_item=-1):
        # define output shape and scaling ratios
        output_256, output = self.forward(target_img, source_img, target_img_256, source_img_256)
        flow_est_list = output['flow_estimates']
        flow_est = flow_est_list[-1]
        uncertainty_list = output['uncertainty_estimates'][-1]  # contains log_var_map and weight_map

        # get the flow field
        flow_est = torch.nn.functional.interpolate(input=flow_est, size=output_shape, mode='bilinear',
                                                   align_corners=False)
        flow_est[:, 0, :, :] *= ratio_x
        flow_est[:, 1, :, :] *= ratio_y

        # get the confidence value
        if isinstance(uncertainty_list[0], list):
            # estimate multiple uncertainty maps per level
            log_var_map = torch.nn.functional.interpolate(input=uncertainty_list[0][list_item], size=output_shape,
                                                          mode='bilinear', align_corners=False)
            weight_map = torch.nn.functional.interpolate(input=uncertainty_list[1][list_item], size=output_shape,
                                                         mode='bilinear', align_corners=False)
        else:
            log_var_map = torch.nn.functional.interpolate(input=uncertainty_list[0], size=output_shape,
                                                          mode='bilinear', align_corners=False)
            weight_map = torch.nn.functional.interpolate(input=uncertainty_list[1], size=output_shape,
                                                         mode='bilinear', align_corners=False)
        p_r = estimate_probability_of_confidence_interval_of_mixture_density(weight_map, log_var_map,
                                                                             R=inference_parameters['R'])
        variance = estimate_average_variance_of_mixture_density(weight_map, log_var_map)
        uncertainty_est = {'log_var_map': log_var_map, 'weight_map': weight_map,
                           'p_r': p_r, 'R': inference_parameters['R'], 'variance': variance}
        return flow_est, uncertainty_est

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
            uncertainty_est: dict with keys 'log_var_map', 'weight_map', 'p_r', 'R', 'variance'
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
