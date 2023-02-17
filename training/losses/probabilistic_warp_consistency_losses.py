import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

from models.modules.feature_correlation_layer import compute_global_correlation
from models.modules.feature_correlation_layer import featureL2Norm
from training.losses.cost_volume_losses.cost_volume_geometry import getBlurredGT_multiple_kp
from training.losses.cost_volume_losses.losses_on_matching_and_non_matching_pairs import SupervisionStrategy
from training.losses.cross_entropy_supervised import (BinaryCrossEntropy, OneHotCrossEntropy,
                                                      OneHotBinaryCrossEntropy, SmoothCrossEntropy)
from utils_flow.flow_and_mapping_operations import convert_flow_to_mapping
from utils_flow.flow_and_mapping_operations import get_gt_correspondence_mask
from utils_flow.correlation_to_matches_utils import cost_volume_to_probabilistic_mapping


class ProbabilisticWarpConsistencyForGlobalCorr(nn.Module):
    """Main module to compute the Probabilistic Warp Consistency loss, on an input cost volume. """
    def __init__(self, name_of_loss, activation='softmax', temperature=1.0, balance_losses=True,
                 loss_name='LogProb', loss_name_warp_sup=None,  occlusion_handling=False, h=None, w=None,
                 apply_loss_on_top=False, top_percent=0.2, apply_mask_warpsup=False):
        """
        Args:
            name_of_loss: 'pw_bipath_and_pwarp_supervision' , 'pw_bipath', 'pwarp_supervision'
            activation: function to apply on the the cost volume to convert it to probabilistic mapping
            temperature: to apply to the cost volume scores before the softmax function
            balance_losses: pw_bipath and pwarp_supervision then amount for the same
            loss_name: final loss module, like cross-entropy. If loss_name_warp_sup not specified, this is used for
                       both pw_bipath and pwarp_supervision
            loss_name_warp_sup: final loss module, like cross-entropy, used when computing the pwarp_supervision loss.
            occlusion_handling: bool, unmatched state prediction?
            h:
            w:
            apply_loss_on_top: bool, compute visibility mask by taking the top x percent per batch of the probabilities
                               of the composition, according to the known synthetic flow.
            top_percent: which percent to use for apply_loss_on_top, or apply_loss_on_top_non_occluded, or
                         apply_loss_on_top_non_occluded_and_best
            apply_mask_warpsup: Apply the same visibility mask to the pwarp_supervision objective.
        """
        super(ProbabilisticWarpConsistencyForGlobalCorr, self).__init__()
        self.name_of_loss = name_of_loss

        self.h, self.w = h, w
        self.balance_losses = balance_losses
        self.loss_module = self.select_loss(loss_name)
        if loss_name_warp_sup is None:
            self.loss_module_warp_sup = self.loss_module
        else:
            self.loss_module_warp_sup = self.select_loss(loss_name_warp_sup)

        self.activation = activation
        self.temperature = temperature

        self.occlusion_handling = occlusion_handling

        self.apply_loss_on_top = apply_loss_on_top
        self.top_percent = top_percent
        self.apply_mask_warpsup = apply_mask_warpsup

    @staticmethod
    def select_loss(loss_name):
        if loss_name == 'LogProb':
            loss_module = OneHotCrossEntropy(reduction='mean')
        elif loss_name == 'SmoothCrossEntropy':
            loss_module = SmoothCrossEntropy(reduction='mean_per_el')
        elif loss_name == 'BCE':
            loss_module = BinaryCrossEntropy(reduction='mean_per_el')
        elif loss_name == 'OneHotBCELoss':
            loss_module = OneHotBinaryCrossEntropy(reduction='mean_per_el')
        else:
            raise ValueError
        return loss_module

    @staticmethod
    def affinity(feature_source, feature_target):
        """
        Args:
            feature_source: B x C x Hs x Ws
            feature_target: B x C x Ht x Wt
        Returns:
            correlation volume, shape B x (Hs*hs) x Ht x Wt
        """
        return compute_global_correlation(feature_source, feature_target, shape='3D')

    def cost_volume_to_probabilistic_mapping(self, A):
        """ Convert cost volume to probabilistic mapping.
        Args:
            A: cost volume, dimension B x C x H x W, matching points are in C
        """
        return cost_volume_to_probabilistic_mapping(A, self.activation, self.temperature)

    def stoch_mat_pw_bipath(self, c_source=None, c_target=None, c_target_prime=None,
                            A_target_prime_to_source=None, A_source_to_target=None):
        """
        Computes the composition according to the PW-bipath constraint
        Args:
            c_source: feature of the source image, in case cost volume is not already computed
            c_target: feature of the target image, in case cost volume is not already computed
            c_target_prime: feature of the target prime image , in case cost volume is not already computed
            A_target_prime_to_source: cost volume from the target prime to the source, b, h_s*w_s, h_tp, w_tp
            A_source_to_target: cost volume from the target prime to the source, b, h_t*w_t, h_s, w_s

        Returns:
            P_W: probabilistic mapping resulting from the composition of P_target_prime_to_source and P_source_to_target
                 according to the PW-bipath constraint. shape is b, h_t*w_t, h_tp, w_tp
            proba_matrices: dict storing intermediate probabilistic mappings.
        """
        proba_matrices = {}

        if A_target_prime_to_source is None:
            if c_source is None or c_target_prime is None:
                raise ValueError
            A_target_prime_to_source = self.affinity(
                feature_source=featureL2Norm(c_source), feature_target=featureL2Norm(c_target_prime))
        # b, h_s*w_s, h_tp, w_tp

        if A_source_to_target is None:
            A_source_to_target = self.affinity(
                feature_source=featureL2Norm(c_target), feature_target=featureL2Norm(c_source))
        # b, h_t*w_t, h_s, w_s

        b, _, h, w = A_target_prime_to_source.shape
        proba_matrices['A_target_to_source'] = self.cost_volume_to_probabilistic_mapping(
            A_source_to_target.detach().view(b, -1, h*w).permute(0, 2, 1).reshape(b, -1, h, w))

        P_target_prime_to_source = torch.flatten(self.cost_volume_to_probabilistic_mapping(A_target_prime_to_source), start_dim=2)
        # b, h_s*w_s, h_tp*w_tp
        P_source_to_target = torch.flatten(self.cost_volume_to_probabilistic_mapping(A_source_to_target), start_dim=2)  # b, h_t*w_t, h_s*w_s
        # now they are b, C, N

        P_W = P_source_to_target @ P_target_prime_to_source

        # should be b, C, N. N is actually h_tp * w_tp and C is h_t * w_t
        proba_matrices['P_target_prime_to_source'] = P_target_prime_to_source.view(b, -1, h, w).detach()
        proba_matrices['P_source_to_target'] = P_source_to_target.view(b, -1, h, w).detach()
        proba_matrices['P_w_bipath'] = P_W.view(b, -1, h, w).detach()

        P_W = P_W.permute(0, 2, 1).contiguous().view(-1, h*w)  # B*h_tp*w_tp, C
        return P_W, proba_matrices

    def stoch_mat_w_bipath_with_bin(self, A_target_prime_to_source=None, A_source_to_target=None):
        """
        Computes the composition according to the PW-bipath constraint when the probabilistic mapping space was
        extended to include an unmatched state.
        Args:
            A_target_prime_to_source: cost volume from the target prime to the source, b, (h_s*w_s + 1), h_tp, w_tp
            A_source_to_target: cost volume from the target prime to the source, b, (h_t*w_t + 1), h_s, w_s

        Returns:
            P_W: probabilistic mapping resulting from the composition of P_target_prime_to_source and P_source_to_target
                 according to the PW-bipath constraint. shape is b, (h_t*w_t + 1), h_tp, w_tp
            proba_matrices: dict storing intermediate probabilitic mappings.
        """
        proba_matrices = {}

        # A_target_prime_to_source b, h_s*w_s +  1, h_tp, w_tp
        # A_source_to_target b, h_t*w_t + 1, h_s, w_s

        b, _, h, w = A_target_prime_to_source.shape

        P_target_prime_to_source = torch.flatten(self.cost_volume_to_probabilistic_mapping(A_target_prime_to_source), start_dim=2)
        # b, h_s*w_s + 1, h_tp*w_tp
        P_source_to_target = torch.flatten(self.cost_volume_to_probabilistic_mapping(A_source_to_target), start_dim=2)
        # b, h_t*w_t + 1, h_s*w_s

        # not really correct anymore cause the matching was done in the other direction
        proba_matrices['A_target_to_source'] = A_source_to_target[:, :h*w]\
            .view(b, -1, h*w).detach().permute(0, 2, 1).view(b, -1, h, w)

        proba_matrices['P_target_prime_to_source'] = P_target_prime_to_source.view(b, -1, h, w).detach()
        proba_matrices['P_source_to_target'] = P_source_to_target.view(b, -1, h, w).detach()

        # need to augment P_source_to_target which hardcoded column to handle P(x_target | x_source, occ)
        # hardcode that P(x_target, occ | x_source, occ) = 1
        occ_proba = torch.zeros(h*w + 1).cuda()
        occ_proba[-1] = 1.0
        occ_proba = occ_proba.reshape(1, -1).repeat(b, 1).unsqueeze(-1)  # b, h*w+1, 1
        P_source_to_target = torch.cat((P_source_to_target, occ_proba), dim=-1)
        # b, h_t*w_t + 1, h_s*w_s + 1

        # dummy example
        assert P_source_to_target[1, -1, -1] == 1.0
        assert P_source_to_target[1, 2, -1] == 0.0

        P_W = P_source_to_target @ P_target_prime_to_source
        # b, h_t*w_t + 1, h_tp*w_tp

        proba_matrices['P_w_bipath'] = P_W.view(b, -1, h, w).detach()

        P_W = P_W.permute(0, 2, 1).contiguous().view(b*h*w, -1)  # B*h_tp*w_tp, h_t*w_t + 1
        return P_W, proba_matrices

    def get_gt_probabilistic_mapping_and_mask(self, flow_gt_full, h, w, mask_valid=None):
        """
        Computes the known probabilistic mapping and valid mask relating the target prime to the target,
        based on the known synthetic flow field.
        Args:
            flow_gt_full: flow relating target prime to index_of_target feature PROVIDED,
                          defined in the target prime coordinate system,
                          shape is in original resolution b, 2, H_tp, W_tp
            h, w: size of the correlation volume (h_tp, w_tp here)
            mask_valid: shape is b, h_tp, w_tp, bool tensor
        Returns:
            index_of_target: coordinates of the mapping from target prime to target (in flattened coordinate),
                             for each point in the target prime image.
                             shape is b*h_tp*w_tp, then N after applying the mask.
            proba_map_gt: ground-truth probabilistic map, with smooth representation, corresponding to the synthetic
                          flow field. shape b*h_tp*w_tp, h_t*w_t or
                          b*h_tp*w_tp, h_t*w_t + 1, if occlusion_handling is True
            mask_for_target: bool tensor indicating if index_of_target is valid. shape is b*h_tp*w_tp
        """
        # resize the ground-truth flow field to correlation dimension and scale accordingly
        b, _, H, W = flow_gt_full.shape
        flow_gt_full = F.interpolate(flow_gt_full, (h, w), mode='bilinear', align_corners=False)
        flow_gt_full[:, 0] *= float(w) / float(W)
        flow_gt_full[:, 1] *= float(h) / float(H)
        mapping_gt = convert_flow_to_mapping(flow_gt_full)

        # check valid mask to exclude computation of the losses there.
        mask_corr = get_gt_correspondence_mask(flow_gt_full)  # must be within the dimensions of cost volume.
        if mask_valid is not None:
            mask_valid = F.interpolate(mask_valid.unsqueeze(1).float(), (h, w), mode='bilinear',
                                       align_corners=False).squeeze(1)
            mask_valid = mask_valid.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask_valid.byte()
            mask_corr = mask_valid & mask_corr

        mapping_gt = mapping_gt.view(b, 2, -1).permute(0, 2, 1).contiguous().view(-1, 2)  # b*h_tp*w_tp, 2

        # here, match is taken as closest integer.
        # just round to closest integer
        mapping_gt_x_rounded = torch.round(mapping_gt[:, 0]).long()  # b*h_tp*w_tp
        mapping_gt_y_rounded = torch.round(mapping_gt[:, 1]).long()  # b*h_tp*w_tp
        index_of_gt_in_image = mapping_gt_x_rounded.lt(w) & mapping_gt_y_rounded.lt(h) & \
                               mapping_gt_x_rounded.ge(0) & mapping_gt_y_rounded.ge(0)  # b*h_tp*w_tp
        # get the index in a flattened array

        mask_for_target = mask_corr.flatten() & index_of_gt_in_image  # b*h_tp*w_tp, sum equal to N
        mapping_gt_x_rounded = mapping_gt_x_rounded[mask_for_target]  # N
        mapping_gt_y_rounded = mapping_gt_y_rounded[mask_for_target]  # N

        # flatten the match position
        index_of_target = mapping_gt_y_rounded * w + mapping_gt_x_rounded  # N

        # also computes known probabilistic mapping, for all pixels of the target prime image. It is computed by
        # considering the distance of the match to the found neighbors.
        # keep only valid points in the index_of_target prime image
        mapping_gt = mapping_gt[mask_for_target]  # N, 2
        proba_map_gt = getBlurredGT_multiple_kp(mapping_gt, [h, w], [h, w])  # N, h_t*w_t

        if self.occlusion_handling:
            # add the ground-truth column corresponding to the bin, ie for these locations, there is a visible mask
            # so the proba to the occluded state should be zero
            proba_map_gt = torch.cat((proba_map_gt, torch.zeros(proba_map_gt.shape[0], 1).cuda()), dim=-1)
        return index_of_target, proba_map_gt, mask_for_target

    def get_loss(self, logits, gt_proba_map, index_of_target, name_of_loss, h, w, stats=None, mask=None):
        """
        Computes the actual loss, based on the known probabilistic mapping and the direct or composition prediction.
        Args:
            logits: estimated probabilistic mapping. Shape is N, h_t*w_t or N, h_t*w_t + 1
            gt_proba_map: known probabilistic mapping, smooth representation. Shape is N, h_t*w_t or N, h_t*w_t + 1
            index_of_target: flattened index of the match (when rounded to the closest integer at the probabilistic
                             mapping resolution). Shape is N
            name_of_loss: 'pw_bipath' or 'pwarp_supervision'
            h, w: dimensions of the probabilistic mappings.
            stats:
            mask: visibility mask, to apply for loss computation

        Returns:
            loss, stats
        """
        if logits.shape[0] == 0:
            loss = torch.zeros(1)
        else:
            if mask is not None:
                logits = logits[mask]
                gt_proba_map = gt_proba_map[mask]
                index_of_target = index_of_target[mask]

            if 'warp_supervision' in name_of_loss:
                loss = self.loss_module_warp_sup(logits=logits, target=gt_proba_map, index_of_target=index_of_target,
                                                 h=h, w=w)
            else:
                loss = self.loss_module(logits=logits, target=gt_proba_map, index_of_target=index_of_target, h=h, w=w)

        if stats is None:
            stats = {}
        stats['{}_loss_{}x{}'.format(name_of_loss, h, w)] = loss.item()
        return loss, stats

    @staticmethod
    def get_stats(logits, index_of_target, name, h, w, stats=None):
        """
        Computes accuracy of the probabilistic direct or composition prediction, according to the ground-truth.
        Args:
            logits: estimated probabilistic mapping. Shape is N, h_t*w_t or N, h_t*w_t + 1
            index_of_target: flattened index of the match (when rounded to the closest integer at the probabilistic
                             mapping resolution). Shape is N
            h, w: dimensions of the probabilistic mappings.
            stats:
        Returns:
            stats
        """
        if stats is None:
            stats = {}
        acc = (torch.argmax(logits.detach(), dim=-1) == index_of_target).float().mean()

        stats['{}_accuracy_{}x{}'.format(name, h, w)] = acc.item()
        return stats

    def compute_visibility_mask(self, proba_map, index_of_target):
        """
        Computes the visibility mask.
        Args:
            proba_map: estimated probabilistic mapping, shape is N, h_t*w_t or N, h_t*w_t + 1
            index_of_target: flattened index of the match (when rounded to the closest integer at the probabilistic
                             mapping resolution). Shape is N
        Returns:
            mask
        """
        proba_map = proba_map.detach()
        index = torch.arange(0, proba_map.shape[0])
        proba = proba_map[index, index_of_target]

        # apply top percent
        indice_for_top_percent = int(len(proba) * self.top_percent)
        value_sorted, indice_sorted = torch.topk(proba, indice_for_top_percent)
        mask = torch.zeros_like(proba)
        mask[indice_sorted] = 1.0
        return mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask.byte()

    def forward(self, flow_gt_full, mask_valid=None, c_source=None, c_target=None, c_target_prime=None,
                A_target_prime_to_source=None, A_source_to_target=None, A_target_prime_to_target=None, *args, **kwargs):
        """
        Args:
            flow_gt_full: flow relating target prime to target feature PROVIDED, defined in the target prime
                          coordinate system. shape is in original resolution b, 2, H_tp, W_tp
            mask_valid: shape is b, h_tp, w_tp, bool tensor
            c_source: features from source image, shape b, d, h_s, w_s
            c_target: features from target image, shape b, d, h_t, w_t
            c_target_prime: features from target prime image, shape b, d, h_tp, w_tp
            A_target_prime_to_source: correlation volume in target prime coordinate system.
                                      shape is b, h_s*w_s, h_tp, w_tp
            A_source_to_target: correlation volume in source coordinate system. shape is b, h_t*w_t, h_s, w_s
            A_target_prime_to_target: correlation volume in target prime coordinate system.
                                      shape is b, h_t*w_t, h_tp, w_tp.
        """

        assert flow_gt_full is not None
        if c_target is not None:
            b, c, h, w = c_target.shape
        else:
            b, _, h, w = A_target_prime_to_source.shape

        target, proba_map_gt, mask_for_target = self.get_gt_probabilistic_mapping_and_mask(flow_gt_full, h, w,
                                                                                           mask_valid=mask_valid)

        return self.compute_probabilistic_warpc(target, proba_map_gt, mask_for_target, c_source, c_target, c_target_prime,
                                                A_target_prime_to_source, A_source_to_target, A_target_prime_to_target)

    def compute_probabilistic_warpc(self, index_of_target, proba_map_gt, mask_for_target, c_source=None,
                                    c_target=None, c_target_prime=None, A_target_prime_to_source=None,
                                    A_source_to_target=None, A_target_prime_to_target=None):
        """
        Main function computing the probabilistic warp consistency objective.
        Args:
            index_of_target: flattened index of the match (when rounded to the closest integer at the probabilistic
                             mapping resolution). Shape is M
            proba_map_gt: known probabilistic mapping, smooth representation. Shape is M, h_t*w_t
            mask_for_target: bool indicating valid coordinates to compute the loss.
                             shape is b, h_tp, w_tp. M positive values
            c_source: features from source image, shape b, d, h_s, w_s
            c_target: features from target image, shape b, d, h_t, w_t
            c_target_prime: features from target prime image, shape b, d, h_tp, w_tp
            A_target_prime_to_source: correlation volume in target prime coordinate system.
                                      shape is b, h_s*w_s, h_tp, w_tp
            A_source_to_target: correlation volume in source coordinate system. shape is b, h_t*w_t, h_s, w_s
            A_target_prime_to_target: correlation volume in target prime coordinate system.
                                      shape is b, h_t*w_t, h_tp, w_tp.

        Returns:
            loss, stats
            dict_proba_matrices: dict containing intermediate probabilistic mapping predictions
        """

        if c_target is not None:
            b, c, h, w = c_target.shape
        else:
            b, _, h, w = A_target_prime_to_source.shape

        # compute the loss
        loss = 0.0
        loss_pw_bipath = 0.0
        loss_pwarp_sup = 0.0
        stats = {}
        dict_proba_matrices = {}
        if 'w_bipath' in self.name_of_loss:
            # get pw_bipath matrix
            if self.occlusion_handling:
                P_w_bipath, dict_proba_matrices_ = self.stoch_mat_w_bipath_with_bin(
                    A_target_prime_to_source=A_target_prime_to_source, A_source_to_target=A_source_to_target)
                stats['avg_proba_of_bin_pos'] = dict_proba_matrices_['P_source_to_target'][:, -1].mean().item()
            else:
                P_w_bipath, dict_proba_matrices_ = self.stoch_mat_pw_bipath(
                    c_source, c_target, c_target_prime, A_target_prime_to_source=A_target_prime_to_source,
                    A_source_to_target=A_source_to_target)

            P_w_bipath = P_w_bipath[mask_for_target]  # take only elements for which we have valid gt

            stats_ = self.get_stats(logits=P_w_bipath, index_of_target=index_of_target, name='pw_bipath', h=h, w=w)

            stats['avg_proba_in_valid_mask_pw_bipath'] = \
                P_w_bipath.detach()[torch.arange(0, P_w_bipath.shape[0]), index_of_target].mean()

            mask = None
            if self.apply_loss_on_top:
                mask = self.compute_visibility_mask(P_w_bipath, index_of_target)
                stats['percent_mask_thresh_pw_bipath'] = mask.sum().item() / len(mask.view(-1))
                stats['avg_proba_in_thresh_mask_pw_bipath'] = \
                    P_w_bipath.detach()[torch.arange(0, P_w_bipath.shape[0]), index_of_target][mask].mean() \
                        if mask.sum() > 0 else 0.0

            loss_, stats_ = self.get_loss(logits=P_w_bipath, gt_proba_map=proba_map_gt, index_of_target=index_of_target,
                                          name_of_loss='pw_bipath', h=h, w=w, mask=mask, stats=stats_)

            loss_pw_bipath += loss_
            loss += loss_pw_bipath
            stats.update(stats_)
            dict_proba_matrices.update(dict_proba_matrices_)

        if 'warp_supervision' in self.name_of_loss:
            # get warp-supervision
            if A_target_prime_to_target is None:
                if c_target is None or c_target_prime is None:
                    raise ValueError
                A_target_prime_to_target = self.affinity(
                    feature_source=featureL2Norm(c_target), feature_target=featureL2Norm(c_target_prime))
            # b, h_t*w_t, h_tp, w_tp

            P_warp_supervision = self.cost_volume_to_probabilistic_mapping(A_target_prime_to_target)
            dict_proba_matrices['P_warp_supervision'] = P_warp_supervision.detach()

            P_warp_supervision = torch.flatten(P_warp_supervision, start_dim=2).permute(0, 2, 1).contiguous()\
                .view(b*h*w, -1)
            # b*h_tp*w_tp, h_t*w_t

            P_warp_supervision = P_warp_supervision[mask_for_target]

            stats_ = self.get_stats(logits=P_warp_supervision, index_of_target=index_of_target,
                                    name='pwarp_supervision', h=h, w=w)

            stats['avg_proba_in_valid_mask_pwarp_sup'] = \
                P_warp_supervision.detach()[torch.arange(0, P_warp_supervision.shape[0]), index_of_target].mean()

            mask = None
            if self.apply_mask_warpsup and self.apply_mask_on_loss:
                mask = self.compute_visibility_mask(P_warp_supervision, index_of_target, h, w)
                stats['percent_mask_thresh_pwarp_sup'] = mask.sum().item() / len(mask.view(-1))
                stats['avg_proba_in_thresh_mask_pwarp_sup'] = \
                    P_warp_supervision.detach()[torch.arange(0, P_warp_supervision.shape[0]),
                                                index_of_target][mask].mean()

            loss_, stats_ = self.get_loss(logits=P_warp_supervision, gt_proba_map=proba_map_gt,
                                          index_of_target=index_of_target,
                                          name_of_loss='pwarp_supervision', h=h, w=w, mask=mask, stats=stats_)
            loss_pwarp_sup += loss_

            w = 1.0
            if self.balance_losses and 'pw_bipath' in self.name_of_loss:
                w = loss_pw_bipath.detach() / (loss_pwarp_sup.detach() + 1e-6)
            loss += w * loss_pwarp_sup
            stats['pwarp_supervision_loss_w_balance'] = (w * loss_pwarp_sup).item()
            stats.update(stats_)

        return loss, stats, dict_proba_matrices


class FullImagePairsCrossEntropyLosses:
    """  Different options for probabilistic losses using the unmatched state. """
    def __init__(self, activation='noactivation', temperature=1.0, label=0.9):
        """
        Args:
            activation: function to apply on the the cost volume to convert it to probabilistic mapping
            temperature: to apply to the cost volume scores before the softmax function
            label:
        """
        self.temperature = temperature
        self.eps = 1e-8
        self.activation = activation
        # usually, it is already a probabilistic mapping. no need to apply further activation
        self.label = label

    def cost_volume_to_probabilistic_mapping(self, A):
        """ Convert cost volume to probabilistic mapping.
        Args:
            A: cost volume, dimension B x C x H x W, matching points are in C
        """
        return cost_volume_to_probabilistic_mapping(A, self.activation, self.temperature)

    # POSITIVE IMAGE PAIRS
    def compute_cross_entropy_loss_mean_per_pixel_pos(self, correlation_matrix_t_to_s):
        """
        Computed between positive image pairs (matching image pairs).
        Mean of probabilistic mapping per pixel of the target, excluding the unmatched state, is enforced to be 1.
        Args:
            correlation_matrix_t_to_s: cost volume from target to source image. shape b, (h_s*w_s+1), h_t, w_t
        """

        b, c, h, w = correlation_matrix_t_to_s.shape
        correlation_matrix_t_to_s = self.cost_volume_to_probabilistic_mapping(correlation_matrix_t_to_s)  # b, h*w +1, h, w
        correlation_matrix_t_to_s = correlation_matrix_t_to_s.permute(0, 2, 3, 1).reshape(b * h * w, -1)
        # b*h*w, h*w + 1

        assert correlation_matrix_t_to_s.shape[-1] == h * w + 1

        avg_score_of_bin = correlation_matrix_t_to_s[:, -1].detach().mean()

        correlation_matrix_t_to_s = correlation_matrix_t_to_s[:, :h * w].mean(dim=1).view(-1, 1)  # remove the bin
        cross_ent = -torch.log(correlation_matrix_t_to_s + self.eps)  # b*h*w, 1

        return cross_ent.mean(), avg_score_of_bin

    def compute_cross_entropy_loss_max_per_pixel_pos(self, correlation_matrix_t_to_s):
        """
        Computed between positive image pairs (matching image pairs).
        Max of probabilistic mapping per pixel of the target, excluding the unmatched state, is enforced to be 1.
        Args:
            correlation_matrix_t_to_s: cost volume from target to source image. shape b, (h_s*w_s+1), h_t, w_t
        """

        b, c, h, w = correlation_matrix_t_to_s.shape
        correlation_matrix_t_to_s = self.cost_volume_to_probabilistic_mapping(correlation_matrix_t_to_s)  # b, h*w +1, h, w
        correlation_matrix_t_to_s = correlation_matrix_t_to_s.permute(0, 2, 3, 1).reshape(b * h * w, -1)  # b*h*w, h*w + 1

        assert correlation_matrix_t_to_s.shape[-1] == h * w + 1

        avg_score_of_bin = correlation_matrix_t_to_s[:, -1].detach().mean()

        correlation_matrix_t_to_s = correlation_matrix_t_to_s[:, :h * w].max(dim=1)[0].view(-1, 1)  # remove the bin
        cross_ent = -torch.log(correlation_matrix_t_to_s + self.eps)  # b*h*w, 1

        return cross_ent.mean(), avg_score_of_bin

    def compute_smooth_cross_entropy_loss_bin_per_pixel_pos(self, correlation_matrix_t_to_s):
        """
        Computed between positive image pairs (matching image pairs).
        Binary cross-entropy. For each pixel of the target, probability of unmatched state is
        enforced to be (1-self.label).
        Args:
            correlation_matrix_t_to_s: cost volume from target to source image. shape b, (h_s*w_s+1), h_t, w_t
        """

        b, c, h, w = correlation_matrix_t_to_s.shape
        correlation_matrix_t_to_s = self.cost_volume_to_probabilistic_mapping(correlation_matrix_t_to_s)  # b, h*w +1, h, w
        correlation_matrix_t_to_s = correlation_matrix_t_to_s.permute(0, 2, 3, 1).reshape(b * h * w,
                                                                                          -1)  # b*h*w, h*w + 1

        avg_score_of_bin = correlation_matrix_t_to_s[:, -1].detach().mean()

        # related to softmax, so sum is 1.
        bin_score = correlation_matrix_t_to_s[:, -1]  # b*h*w, 1
        pos_sum = 1. - bin_score  # b*h*w, 1, same than taking sum there

        cross_ent_bin = -torch.log(bin_score + self.eps)  # b*h*w, 1
        cross_ent_pos = -torch.log(pos_sum + self.eps)  # b*h*w, 1

        cross_ent = self.label * cross_ent_pos + (1. - self.label) * cross_ent_bin

        return cross_ent.mean(), avg_score_of_bin

    def compute_smooth_cross_entropy_loss_bin_per_image_pos(self, correlation_matrix_t_to_s):
        """
        Computed between positive image pairs (matching image pairs).
        Binary cross-entropy. For each image, computes an unmatched score as the mean over all unmatched predictions per
        image. Enforces this unmatched score to be (1-self.label).
        Args:
            correlation_matrix_t_to_s: cost volume from target to source image. shape b, (h_s*w_s+1), h_t, w_t
        """

        b, c, h, w = correlation_matrix_t_to_s.shape
        correlation_matrix_t_to_s = self.cost_volume_to_probabilistic_mapping(correlation_matrix_t_to_s)  # b, h*w +1, h, w
        correlation_matrix_t_to_s = correlation_matrix_t_to_s.permute(0, 2, 3, 1).reshape(b, h * w, -1)
        # b, h*w, h*w + 1

        avg_score_of_bin = correlation_matrix_t_to_s[:, :, -1].detach().mean()

        # related to softmax, so sum is 1.
        bin_score = correlation_matrix_t_to_s[:, :, -1]  # b, h*w, 1
        pos = correlation_matrix_t_to_s[:, :, :-1]  # b, h*w, h*w

        bin_score_per_image = bin_score.mean(1)  # b
        pos_score_per_image = pos.reshape(b, -1).sum(1) / float(h*w)  # b

        cross_ent_bin = -torch.log(bin_score_per_image + self.eps)  # b
        cross_ent_pos = -torch.log(pos_score_per_image + self.eps)  # b

        cross_ent = self.label * cross_ent_pos + (1. - self.label) * cross_ent_bin

        return cross_ent.mean(), avg_score_of_bin

    # NEGATIVE IMAGE PAIRS
    def compute_cross_entropy_loss_bin_per_pixel_neg(self, correlation_matrix_t_to_s):
        """
        Computed between negative image pairs (non-matching image pairs).
        Probability of the unmatched state, is enforced to be 1 for all pixels of the target.
        Args:
            correlation_matrix_t_to_s: cost volume from target to source image. shape b, (h_s*w_s+1), h_t, w_t
        """

        b, c, h, w = correlation_matrix_t_to_s.shape
        correlation_matrix_t_to_s = self.cost_volume_to_probabilistic_mapping(correlation_matrix_t_to_s)  # b, h*w +1, h, w
        correlation_matrix_t_to_s = correlation_matrix_t_to_s.permute(0, 2, 3, 1).reshape(b * h * w, -1)  # b*h*w, h*w + 1

        avg_score_of_bin = correlation_matrix_t_to_s[:, -1].detach().mean()

        cross_ent = -torch.log(correlation_matrix_t_to_s + self.eps)  # b*h*w, h*w+1

        assert cross_ent.shape[-1] == h * w + 1

        cross_ent = cross_ent[range(b * h * w), -1]

        return cross_ent.mean(), avg_score_of_bin

    def compute_smooth_cross_entropy_loss_bin_per_pixel_neg(self, correlation_matrix_t_to_s):
        """
        Computed between negative image pairs (non-matching image pairs). Binary cross entropy.
        Probability of the unmatched state, is enforced to be self.label for all pixels of the target.
        Args:
            correlation_matrix_t_to_s: cost volume from target to source image. shape b, (h_s*w_s+1), h_t, w_t
        """

        b, c, h, w = correlation_matrix_t_to_s.shape
        correlation_matrix_t_to_s = self.cost_volume_to_probabilistic_mapping(correlation_matrix_t_to_s)  # b, h*w +1, h, w
        correlation_matrix_t_to_s = correlation_matrix_t_to_s.permute(0, 2, 3, 1).reshape(b * h * w,
                                                                                          -1)  # b*h*w, h*w + 1

        avg_score_of_bin = correlation_matrix_t_to_s[:, -1].detach().mean()

        # related to softmax, so sum is 1.
        bin_score = correlation_matrix_t_to_s[:, -1]  # b*h*w, 1
        pos_sum = 1. - bin_score  # b*h*w, 1, same than taking sum there

        cross_ent_bin = - torch.log(bin_score + self.eps)  # b*h*w, 1
        cross_ent_pos = - torch.log(pos_sum + self.eps)  # b*h*w, 1

        cross_ent = self.label * cross_ent_bin + (1. - self.label) * cross_ent_pos

        return cross_ent.mean(), avg_score_of_bin

    def compute_smooth_cross_entropy_loss_bin_per_pixel_neg_per_pixel_pos(self, correlation_matrix_t_to_s):
        """
        Computed between negative image pairs (non-matching image pairs).
        Probability of the unmatched state, is enforced to be self.label for all pixels of the target.
        For each target pixel, all other probability scores are enforced to be (1-self.label)/(w*h).
        Args:
            correlation_matrix_t_to_s: cost volume from target to source image. shape b, (h_s*w_s+1), h_t, w_t
        """

        b, c, h, w = correlation_matrix_t_to_s.shape
        correlation_matrix_t_to_s = self.cost_volume_to_probabilistic_mapping(correlation_matrix_t_to_s)  # b, h*w +1, h, w
        correlation_matrix_t_to_s = correlation_matrix_t_to_s.permute(0, 2, 3, 1).reshape(b * h * w, -1)
        # b*h*w, h*w + 1

        avg_score_of_bin = correlation_matrix_t_to_s[:, -1].detach().mean()

        # related to softmax, so sum is 1.
        bin_score = correlation_matrix_t_to_s[:, -1]  # b*h*w, 1
        pos = correlation_matrix_t_to_s[:, :-1]  # b*h*w, h*w

        cross_ent_bin = -torch.log(bin_score + self.eps)  # b*h*w, 1
        cross_ent_pos = -torch.log(pos + self.eps)  # b*h*w, h*w

        cross_ent = self.label * cross_ent_bin + ((1. - self.label) / float(h*w) * cross_ent_pos).sum(1)

        return cross_ent.mean(), avg_score_of_bin

    def compute_smooth_cross_entropy_loss_bin_per_image_neg(self, correlation_matrix_t_to_s):
        """
        Computed between negative image pairs (non-matching image pairs).
        Binary cross-entropy. For each image, computes an unmatched score as the mean over all unmatched predictions per
        image. Enforces this unmatched score to be self.label.
        Args:
            correlation_matrix_t_to_s: cost volume from target to source image. shape b, (h_s*w_s+1), h_t, w_t
        """

        b, c, h, w = correlation_matrix_t_to_s.shape
        correlation_matrix_t_to_s = self.cost_volume_to_probabilistic_mapping(correlation_matrix_t_to_s)  # b, h*w +1, h, w
        correlation_matrix_t_to_s = correlation_matrix_t_to_s.permute(0, 2, 3, 1).reshape(b, h * w, -1)
        # b, h*w, h*w + 1

        avg_score_of_bin = correlation_matrix_t_to_s[:, :, -1].detach().mean()

        bin_score = correlation_matrix_t_to_s[:, :, -1]  # b, h*w, 1
        pos = correlation_matrix_t_to_s[:, :, :-1]  # b, h*w, h*w

        bin_score_per_image = bin_score.mean(1)  # b
        pos_score_per_image = pos.reshape(b, -1).sum(1) / float(h*w)  # b, same than 1-bin_score_per_image
        # average over the image of the sum of positive (per pixel)

        cross_ent_bin = - torch.log(bin_score_per_image + self.eps)  # b
        cross_ent_pos = - torch.log(pos_score_per_image + self.eps)  # b

        cross_ent = self.label * cross_ent_bin + (1. - self.label) * cross_ent_pos

        return cross_ent.mean(), avg_score_of_bin

    def compute_fro_norm_loss_bin_per_pixel_neg(self, correlation_matrix_t_to_s):
        """
        Computed between negative image pairs (non-matching image pairs).
        Probability of the unmatched state, is enforced to be 1 for all pixels of the target with norm.
        Args:
            correlation_matrix_t_to_s: cost volume from target to source image. shape b, (h_s*w_s+1), h_t, w_t
        """

        b, c, h, w = correlation_matrix_t_to_s.shape
        correlation_matrix_t_to_s = self.cost_volume_to_probabilistic_mapping(correlation_matrix_t_to_s)  # b, h*w +1, h, w
        correlation_matrix_t_to_s = correlation_matrix_t_to_s.permute(0, 2, 3, 1).reshape(b * h * w, -1)
        # b*h*w, h*w + 1

        avg_score_of_bin = correlation_matrix_t_to_s[:, -1].detach().mean()

        target = torch.zeros_like(correlation_matrix_t_to_s)
        target[torch.torch.arange(0, target.shape[0]), -1] = 1.0  # the bin is 1.0

        loss = torch.norm(correlation_matrix_t_to_s - target, dim=1, p='fro')

        return loss.mean(), avg_score_of_bin

    def compute_smooth_fro_norm_loss_bin_per_pixel_neg(self, correlation_matrix_t_to_s):
        """
        Computed between negative image pairs (non-matching image pairs).
        Probability of the unmatched state, is enforced to be self.label for all pixels of the target. And probability
        of non-unmatched state is enforced to (1-self.label) for all pixels of the target. Using the norm.
        Args:
            correlation_matrix_t_to_s: cost volume from target to source image. shape b, (h_s*w_s+1), h_t, w_t
        """

        b, c, h, w = correlation_matrix_t_to_s.shape
        correlation_matrix_t_to_s = self.cost_volume_to_probabilistic_mapping(correlation_matrix_t_to_s)  # b, h*w +1, h, w
        correlation_matrix_t_to_s = correlation_matrix_t_to_s.permute(0, 2, 3, 1).reshape(b * h * w,
                                                                                          -1)  # b*h*w, h*w + 1

        avg_score_of_bin = correlation_matrix_t_to_s[:, -1].detach().mean()

        # related to softmax, so sum is 1.
        bin_score = correlation_matrix_t_to_s[:, -1]  # b*h*w, 1
        pos_sum = 1. - bin_score  # b*h*w, 1, same than taking sum there

        loss = torch.norm(bin_score - self.label, dim=1, p='fro') + \
               torch.norm(pos_sum - (1 - self.label), dim=1, p='fro')

        return loss.mean(), avg_score_of_bin


class NegProbabilisticBin(SupervisionStrategy):
    """ Using our explicit occlusion modeling, where the probability of unmatched is encoded in the probabilistic
    mapping, this module computes a probabilistic loss relating non-matching image pairs. """
    def __init__(self, activation='noactivation', temperature=1.0, label_for_smooth_ce=0.9,
                 name_of_loss='max_per_pixel'):
        self.num_negatives = 0
        self.name_of_loss = name_of_loss
        self.temperature = temperature
        self.eps = 1e-8
        self.activation = activation
        self.loss_module = FullImagePairsCrossEntropyLosses(activation, temperature=temperature,
                                                            label=label_for_smooth_ce)

    def get_image_pair(self, batch, *args):
        """Forms positive/negative image paris for weakly-supervised training"""
        training = args[0]
        self.bsz = len(batch['source_image'])

        if training:
            shifted_idx = np.roll(np.arange(self.bsz), -1)
            trg_img_neg = batch['target_image'][shifted_idx].clone()
            trg_cls_neg = batch['category_id'][shifted_idx].clone()
            neg_subidx = (batch['category_id'] - trg_cls_neg) != 0

            src_img = torch.cat((batch['source_image'], batch['source_image'][neg_subidx]), dim=0)
            trg_img = torch.cat((batch['target_image'], trg_img_neg[neg_subidx]), dim=0)
            self.num_negatives = neg_subidx.sum()
        else:
            src_img, trg_img = batch['source_image'], batch['target_image']
            self.num_negatives = 0

        return src_img, trg_img

    def get_correlation(self, correlation_matrix):
        """Returns correlation matrices of 'POSITIVE PAIRS' in a batch"""
        return correlation_matrix[:self.bsz].clone().detach()

    def compute_cross_entropy_loss(self, correlation_matrix_t_to_s):
        if self.name_of_loss == 'max_per_pixel':
            return self.loss_module.compute_cross_entropy_loss_bin_per_pixel_neg(correlation_matrix_t_to_s)
        elif self.name_of_loss == 'smooth_max_per_pixel':
            return self.loss_module.compute_smooth_cross_entropy_loss_bin_per_pixel_neg(correlation_matrix_t_to_s)
        elif self.name_of_loss == 'smooth_max_per_pixel_neg_pos':
            return self.loss_module.compute_smooth_cross_entropy_loss_bin_per_pixel_neg_per_pixel_pos(correlation_matrix_t_to_s)
        elif self.name_of_loss == 'smooth_max_per_image':
            return self.loss_module.compute_smooth_cross_entropy_loss_bin_per_image_neg(correlation_matrix_t_to_s)
        elif self.name_of_loss == 'fronorm_per_pixel':
            return self.loss_module.compute_fro_norm_loss_bin_per_pixel_neg(correlation_matrix_t_to_s)
        elif self.name_of_loss == 'smooth_fronorm_per_pixel':
            return self.loss_module.compute_smooth_fro_norm_loss_bin_per_pixel_neg(correlation_matrix_t_to_s)
        else:
            raise ValueError

    def compute_loss(self, correlation_matrix, *args, **kwargs):
        """Weakly-supervised matching loss"""

        # b, h*w +1, h*w or b, h*w +1, h, w
        stats = {}

        if len(correlation_matrix.shape) == 3:
            b, c, hw = correlation_matrix.shape
            h = w = int(math.sqrt(hw))
        else:
            b, c, h, w = correlation_matrix.shape
            hw = h*w

        correlation_matrix = correlation_matrix.view(b, -1, h, w)  # b, h*w +1, h, w

        if self.num_negatives > 0:
            loss_net, avg_score_of_bin = self.compute_cross_entropy_loss(correlation_matrix[self.bsz:])
        else:
            loss_net, avg_score_of_bin = torch.as_tensor(0.0), torch.as_tensor(0.0)  # only for the negatives

        stats['avg_max_score_pos'] = torch.max(correlation_matrix[:self.bsz, :hw].detach(), dim=1)[0].mean()
        if self.num_negatives > 0:
            stats['avg_max_score_neg'] = torch.max(correlation_matrix[self.bsz:, :hw].detach(), dim=1)[0].mean()
            stats['avg_proba_of_bin_neg'] = avg_score_of_bin.item()

        stats['Loss_neg'] = loss_net.item()
        stats['Loss_pos_neg/total'] = loss_net.item()
        return loss_net, stats

