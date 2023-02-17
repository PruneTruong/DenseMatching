import math
from packaging import version
import torch

import training.losses.cost_volume_losses.cost_volume_geometry as geometry
from ..multiscale_loss import MultiScaleFlow
from ..basic_losses import EPE
from .losses_on_matching_and_non_matching_pairs import SupervisionStrategy
from ..cross_entropy_supervised import build_one_hot
from utils_flow.correlation_to_matches_utils import cost_volume_to_probabilistic_mapping


class SparseSupervisedSmoothGT(SupervisionStrategy):
    """
    Based on keypoint annotations, creates a ground-truth smooth probability map for each. From the estimated
    cost volume, computes the loss between estimated and ground-truth probability map for each key-point.
    """
    def __init__(self, objective, activation='softmax', temperature=1.0, interpolate_position=True,
                 enforce_gt_sum_to_1=False, apply_gaussian_kernel=True):
        self.num_negatives = 0
        self.objective = objective
        self.activation = activation
        self.target_rate = 0.5
        self.temperature = temperature
        self.loss_func = objective  # treats each entry as a binary classification
        self.interpolate_position = interpolate_position
        self.enforce_gt_sum_to_1 = enforce_gt_sum_to_1
        self.apply_gaussian_kernel = apply_gaussian_kernel

    def get_image_pair(self, batch, *args):
        """Only requires positive image pairs"""
        self.bsz = len(batch['source_image'])
        return batch['source_image'], batch['target_image']

    def get_correlation(self, correlation_matrix):
        """Returns correlation matrices of 'POSITIVE PAIRS' in a batch"""
        return correlation_matrix[:self.bsz].clone().detach()

    def cost_volume_to_probabilistic_mapping(self, A):
        """ Convert cost volume to probabilistic mapping.
        Args:
            A: cost volume, dimension B x C x H x W, matching points are in C
        """
        return cost_volume_to_probabilistic_mapping(A, self.activation, self.temperature)

    def cross_entropy_loss2d(self, correlation_matrix_t_to_s, source_kp, target_kp, n_valid_pts,
                             original_shape):
        """
        Args:
            correlation_matrix_t_to_s: correlation volume in target coordinate system. shape is b, h_s*w_s, h_t, w_t.
            target_kp: kp in target, shape B, N, 2. only n_pts of valid ones, the others have coordinates below 0.
            source_kp: kp in source, shape B, N, 2. only n_pts of valid ones, the others have coordinates below 0.
            original_shape: at which the keypoints where extracted (H, W)
        Returns:
            loss
            stats
        """

        assert len(correlation_matrix_t_to_s.shape) == 4
        b, c, h, w = correlation_matrix_t_to_s.shape

        stats = {}
        featsShape = [h, w]

        loss = 0

        # per image in the batch
        for i in range(b):

            kps_src = source_kp[i][:n_valid_pts[i]]
            kps_trg = target_kp[i][:n_valid_pts[i]]

            # resize to this
            kps_src[:, 0] *= w / original_shape[1]
            kps_src[:, 1] *= h / original_shape[0]
            kps_trg[:, 0] *= w / original_shape[1]
            kps_trg[:, 1] *= h / original_shape[0]

            if self.interpolate_position:
                # check that the keypoints are within the array
                mask_corr_valid = kps_src[:, 0].ge(0) & kps_trg[:, 0].ge(0) & \
                                  kps_src[:, 1].ge(0) & kps_trg[:, 1].ge(0) & \
                                  kps_src[:, 0].lt(w) & kps_trg[:, 0].lt(w) & \
                                  kps_src[:, 1].lt(h) & kps_trg[:, 1].lt(h)
            else:
                # check that the keypoints are within the array
                mask_corr_valid = kps_src[:, 0].ge(0) & kps_trg[:, 0].ge(0) & \
                                  kps_src[:, 1].ge(0) & kps_trg[:, 1].ge(0) & \
                                  torch.round(kps_src[:, 0]).lt(w) & torch.round(kps_trg[:, 0]).lt(w) & \
                                  torch.round(kps_src[:, 1]).lt(h) & torch.round(kps_trg[:, 1]).lt(h)

            kps_src = kps_src[mask_corr_valid]
            kps_trg = kps_trg[mask_corr_valid]

            if kps_src.shape[0] > 0:
                # there are valid kp
                kps_src = torch.t(kps_src).cuda()  # 2, N
                kps_trg = torch.t(kps_trg).cuda()
                weights = torch.zeros(kps_src.shape[1], c).cuda()
                # targets = torch.zeros(kps_src.shape[1], c).cuda()

                for j in range(0, kps_src.shape[1]):
                    if self.interpolate_position:
                        weights[j] = geometry.BilinearInterpolate(
                            [kps_trg[0, j], kps_trg[1, j]], correlation_matrix_t_to_s[i], featsShape)
                    else:
                        correlation_matrix_t_to_s_at_j = correlation_matrix_t_to_s[i].view(-1, h, w).permute(1, 2, 0)
                        # h_t, w_t, h_s*w_s
                        weights[j] = correlation_matrix_t_to_s_at_j[torch.round(kps_trg[1, j]).long(),
                                                                    torch.round(kps_trg[0, j]).long()]

                    # the probabilisty map corresponding to the source kp ==> gt for prediction at target kp.
                    # targets[j] = geometry.getBlurredGT(
                    #     [kps_src[0, j], kps_src[1, j]], featsShape, featsShape)
                    # does not sum to 1 because gaussian kernel applied there

                targets = geometry.getBlurredGT_multiple_kp(torch.t(kps_src), featsShape, featsShape,
                                                            apply_gaussian_kernel=self.apply_gaussian_kernel)
                if self.enforce_gt_sum_to_1:
                    targets = targets / targets.sum(axis=1, keepdim=True)
                if weights.shape[-1] == h*w + 1:
                    # include the gt for the bin
                    targets = torch.cat([targets, torch.zeros(weights.shape[0], 1).cuda()], dim=-1)
                loss += self.loss_func(logits=self.cost_volume_to_probabilistic_mapping(weights), target=targets)

        loss = loss / b
        stats['kp_loss_{}x{}'.format(h, w)] = loss.item()
        return loss, stats

    def compute_loss(self, correlation_matrix, t_to_s=True, batch=None, *args, **kwargs):
        """
        Args:
            correlation_matrix: correlation volume, can be in target or source coordinate system.
            t_to_s: bool, if Tue, correlation_matrix has shape b, h_s*w_s, h_t, w_t.
            batch: must at least contain the fields 'target_kps', 'source_kps', 'n_pts', 'target_image'.
        Returns:
            loss
            stats
        """

        if len(correlation_matrix.shape) == 3:
            hw = correlation_matrix.shape[-1]
            h = w = int(math.sqrt(hw))
        else:
            h, w = correlation_matrix.shape[-2:]
        b = correlation_matrix.shape[0]
        correlation_matrix = correlation_matrix.view(b, -1, h, w)

        if t_to_s:
            return self.cross_entropy_loss2d(correlation_matrix, target_kp=batch['target_kps'].clone(),
                                             source_kp=batch['source_kps'].clone(),
                                             n_valid_pts=batch['n_pts'],
                                             original_shape=batch['target_image'].shape[-2:])
        else:
            return self.cross_entropy_loss2d(correlation_matrix, source_kp=batch['target_kps'].clone(),
                                             target_kp=batch['source_kps'].clone(), n_valid_pts=batch['n_pts'],
                                             original_shape=batch['target_image'].shape[-2:])


class SparseSupervisedOneHotGT(SupervisionStrategy):
    """
    Based on keypoint annotations, creates a ground-truth one-hot probability map for each. From the estimated
    cost volume, computes the loss between estimated and ground-truth probability map for each key-point.
    """
    def __init__(self, objective, activation='softmax', temperature=1.0, interpolate_position=True):
        self.num_negatives = 0
        self.objective = objective
        self.activation = activation
        self.target_rate = 0.5
        self.temperature = temperature
        self.loss_func = objective  # treats each entry as a binary classification
        self.interpolate_position = interpolate_position

    def get_image_pair(self, batch, *args):
        """Only requires positive image pairs"""
        self.bsz = len(batch['source_image'])
        return batch['source_image'], batch['target_image']

    def get_correlation(self, correlation_matrix):
        """Returns correlation matrices of 'POSITIVE PAIRS' in a batch"""
        return correlation_matrix[:self.bsz].clone().detach()

    def cost_volume_to_probabilistic_mapping(self, A):
        """ Convert cost volume to probabilistic mapping.
        Args:
            A: cost volume, dimension B x C x H x W, matching points are in C
        """
        return cost_volume_to_probabilistic_mapping(A, self.activation, self.temperature)

    def cross_entropy_loss2d(self, correlation_matrix_t_to_s, source_kp, target_kp, n_valid_pts,
                             original_shape):
        """
        Args:
            correlation_matrix_t_to_s: correlation volume in target coordinate system. shape is b, h_s*w_s, h_t, w_t.
            target_kp: kp in target, shape B, N, 2. only n_pts of valid ones, the others have coordinates below 0.
            source_kp: kp in source, shape B, N, 2. only n_pts of valid ones, the others have coordinates below 0.
            original_shape: at which the keypoints where extracted (H, W)
        Returns:
            loss
            stats
        """
        assert len(correlation_matrix_t_to_s.shape) == 4
        b, c, h, w = correlation_matrix_t_to_s.shape

        stats = {}
        featsShape = [h, w]

        loss = 0.

        # per image in the batch
        for i in range(b):

            kps_src = source_kp[i][:n_valid_pts[i]]
            kps_trg = target_kp[i][:n_valid_pts[i]]

            # resize to this
            kps_src[:, 0] *= w / original_shape[1]
            kps_src[:, 1] *= h / original_shape[0]
            kps_trg[:, 0] *= w / original_shape[1]
            kps_trg[:, 1] *= h / original_shape[0]

            # check that the keypoints are within the array
            mask_corr_valid = kps_src[:, 0].ge(0) & kps_trg[:, 0].ge(0) & \
                              kps_src[:, 1].ge(0) & kps_trg[:, 1].ge(0) & \
                              torch.round(kps_src[:, 0]).lt(w) & torch.round(kps_trg[:, 0]).lt(w) & \
                              torch.round(kps_src[:, 1]).lt(h) & torch.round(kps_trg[:, 1]).lt(h)
            kps_src = kps_src[mask_corr_valid]
            kps_trg = kps_trg[mask_corr_valid]

            if kps_src.shape[0] > 0:
                # there are some valid kp
                kps_src = torch.t(kps_src).cuda()  # 2, N
                kps_trg = torch.t(kps_trg).cuda()
                weights = torch.zeros(kps_src.shape[1], c).cuda()
                targets = torch.zeros(kps_src.shape[1], c).cuda()
                indexes = torch.zeros(kps_src.shape[1]).long().cuda()

                for j in range(0, kps_src.shape[1]):
                    if self.interpolate_position:
                        weights[j] = geometry.BilinearInterpolate(
                            [kps_trg[0, j], kps_trg[1, j]], correlation_matrix_t_to_s[i], featsShape)
                    else:
                        correlation_matrix_t_to_s_at_j = correlation_matrix_t_to_s[i].view(-1, h, w).permute(1, 2, 0)
                        # h_t, w_t, h_s*w_s
                        weights[j] = correlation_matrix_t_to_s_at_j[torch.round(kps_trg[1, j]).long(),
                                                                    torch.round(kps_trg[0, j]).long()]

                    # the probabilisty map corresponding to the source kp ==> gt for prediction at target kp.
                    index = torch.round(kps_src[1, j]) * w + torch.round(kps_src[0, j])
                    indexes[j] = index
                    targets[j] = build_one_hot(index.long(), c)
                loss += self.loss_func(logits=self.cost_volume_to_probabilistic_mapping(weights),
                                       target=targets, index_of_target=indexes)

        loss = loss / b
        stats['kp_loss_{}x{}'.format(h, w)] = loss.item()
        return loss, stats

    def compute_loss(self, correlation_matrix, t_to_s=True, batch=None, *args, **kwargs):
        """
        Args:
            correlation_matrix: correlation volume, can be in target or source coordinate system.
            t_to_s: bool, if Tue, correlation_matrix has shape b, h_s*w_s, h_t, w_t.
            batch: must at least contain the fields 'target_kps', 'source_kps', 'n_pts', 'target_image'.
        Returns:
            loss
            stats
        """

        if len(correlation_matrix.shape) == 3:
            hw = correlation_matrix.shape[-1]
            h = w = int(math.sqrt(hw))
        else:
            h, w = correlation_matrix.shape[-2:]
        b = correlation_matrix.shape[0]
        correlation_matrix = correlation_matrix.view(b, -1, h, w)

        if t_to_s:
            return self.cross_entropy_loss2d(correlation_matrix, target_kp=batch['target_kps'].clone(),
                                             source_kp=batch['source_kps'].clone(),
                                             n_valid_pts=batch['n_pts'],
                                             original_shape=batch['target_image'].shape[-2:])
        else:
            return self.cross_entropy_loss2d(correlation_matrix, source_kp=batch['target_kps'].clone(),
                                             target_kp=batch['source_kps'].clone(), n_valid_pts=batch['n_pts'],
                                             original_shape=batch['target_image'].shape[-2:])


class SparseSupervisedOneHotCE(SupervisionStrategy):
    """
    Based on keypoint annotations, estimates the label for each keypoint. From the estimated
    cost volume, computes the loss on the estimated probability map for each key-point.
    Similar than SparseSupervisedOneHotGT, when using the cross-entropy loss.
    """
    def __init__(self, objective,  activation='softmax', temperature=1.0):
        self.num_negatives = 0
        self.objective = objective
        self.activation = activation
        self.target_rate = 0.5
        self.temperature = temperature

    def get_image_pair(self, batch, *args):
        """Only requires positive image pairs"""
        self.bsz = len(batch['source_image'])
        return batch['source_image'], batch['target_image']

    def get_correlation(self, correlation_matrix):
        """Returns correlation matrices of 'POSITIVE PAIRS' in a batch"""
        return correlation_matrix[:self.bsz].clone().detach()

    def cost_volume_to_probabilistic_mapping(self, A):
        """ Convert cost volume to probabilistic mapping.
        Args:
            A: cost volume, dimension B x C x H x W, matching points are in C
        """
        return cost_volume_to_probabilistic_mapping(A, self.activation, self.temperature)

    @staticmethod
    def get_gt_mapping_and_mask_from_keypoints(target_kp, source_kp, original_shape, h, w):
        """
        Args:
            target_kp: kp in target, shape B, N, 2. only n_pts of valid ones, the others have coordinates below 0.
            source_kp: kp in source, shape B, N, 2. only n_pts of valid ones, the others have coordinates below 0.
            original_shape: at which the keypoints where extracted (H, W)
            h, w: size of the correlation volume (h_tp, w_tp here)
        Returns:
            target: coordinate of mapping from target to source (in flattened coordinate).
                    shape is M (after applying the mask)
            mask: bool tensor indicating if target is valid. shape is b*h_tp*w_tp
        """
        # resize the ground-truth to correlation dimension and scale accordingly
        target_kp = target_kp.clone().cuda()  # b,N,2
        source_kp = source_kp.clone().cuda()  # b,N,2
        b, N = source_kp.shape[:2]

        # remove kp for which value is -1
        kp_mask = source_kp[:, :, 1].ge(0) & source_kp[:, :, 0].ge(0) & \
                  target_kp[:, :, 1].ge(0) & target_kp[:, :, 0].ge(0)  # b, N

        # rescale the kp to correlation size
        target_kp[:, :, 0] *= w / original_shape[1]
        target_kp[:, :, 1] *= h / original_shape[0]
        source_kp[:, :, 0] *= w / original_shape[1]
        source_kp[:, :, 1] *= h / original_shape[0]

        kp_mask = kp_mask.view(-1)  # B*N, M positive
        source_kp_rounded = torch.round(source_kp).long().view(-1, 2)[kp_mask]  # B*N*2, then M*2
        target_kp_rounded = torch.round(target_kp).long().view(-1, 2)[kp_mask]  # B*N*2, then M*2

        mask_corr_valid = source_kp_rounded[:, 0].ge(0) & target_kp_rounded[:, 0].ge(0) & \
                    source_kp_rounded[:, 1].ge(0) & target_kp_rounded[:, 1].ge(0) & \
                    source_kp_rounded[:, 0].lt(w) & target_kp_rounded[:, 0].lt(w) & \
                    source_kp_rounded[:, 1].lt(h) & target_kp_rounded[:, 1].lt(h)

        source_kp_rounded = source_kp_rounded[mask_corr_valid]
        target_kp_rounded = target_kp_rounded[mask_corr_valid]

        # store corresponding batch index
        batch_index = torch.arange(0, b)
        batch_index = batch_index.reshape(b, 1).repeat(1, N).flatten()  # b*N
        batch_index = batch_index[kp_mask][mask_corr_valid]  # only the valid ones, M
        M = len(batch_index)

        # need to create the mask!
        mask = torch.zeros(b, h, w).cuda()
        mapping = torch.zeros(b, h, w, 2).long().cuda()
        mask[batch_index, target_kp_rounded[range(M), 1], target_kp_rounded[range(M), 0]] = 1
        mask = mask.view(-1).bool() if version.parse(torch.__version__) >= version.parse("1.1") else \
            mask.view(-1).byte()  # b*h_tp*w_tp

        mapping[batch_index, target_kp_rounded[range(M), 1], target_kp_rounded[range(M), 0]] = \
            source_kp_rounded[range(M)]

        mapping_flattened = mapping.view(-1, 2)[mask]  # M, 2
        target = mapping_flattened[:, 1] * w + mapping_flattened[:, 0]  # M

        assert len(target) == mask.sum()

        return target, mask

    @staticmethod
    def get_stats(logits, target, name, h, w, stats=None):
        if stats is None:
            stats = {}
        acc = (torch.argmax(logits.detach(), dim=-1) == target).float().mean()

        stats['{}_accuracy_{}x{}'.format(name, h, w)] = acc.item()
        return stats

    def compute_loss(self, correlation_matrix, t_to_s=True, batch=None, *args, **kwargs):
        """
        Args:
            correlation_matrix: correlation volume, can be in target or source coordinate system.
            t_to_s: bool, if Tue, correlation_matrix has shape b, h_s*w_s, h_t, w_t.
            batch: must at least contain the fields 'target_kps', 'source_kps', 'n_pts', 'target_image'.
        Returns:
            loss
            stats
        """

        if t_to_s:
            return self.compute_cross_entropy_with_gt_kp(correlation_matrix, target_kp=batch['target_kps'].clone(),
                                                         source_kp=batch['source_kps'].clone(),
                                                         original_shape=batch['target_image'].shape[-2:])
        else:
            return self.compute_cross_entropy_with_gt_kp(correlation_matrix, source_kp=batch['target_kps'].clone(),
                                                         target_kp=batch['source_kps'].clone(),
                                                         original_shape=batch['target_image'].shape[-2:])

    def compute_cross_entropy_with_gt_kp(self, correlation_matrix_t_to_s, target_kp, source_kp, original_shape):
        """
        Args:
            correlation_matrix_t_to_s: correlation volume in target coordinate system. shape is b, h_s*w_s, h_t, w_t.
            target_kp: kp in target, shape B, N, 2. only n_pts of valid ones, the others have coordinates below 0.
            source_kp: kp in source, shape B, N, 2. only n_pts of valid ones, the others have coordinates below 0.
            original_shape: at which the keypoints where extracted (H, W)
        Returns:
            loss
            stats
        """

        if len(correlation_matrix_t_to_s.shape) == 3:
            hw = correlation_matrix_t_to_s.shape[-1]
            h = w = int(math.sqrt(hw))
        else:
            h, w = correlation_matrix_t_to_s.shape[-2:]
        b = correlation_matrix_t_to_s.shape[0]
        correlation_matrix_t_to_s = correlation_matrix_t_to_s.view(b, -1, h, w)
        # b, h_s*w_s, h_t, w_t

        labels, mask_for_labels = self.get_gt_mapping_and_mask_from_keypoints(target_kp, source_kp,
                                                                              original_shape, h, w)

        P_warp_supervision = self.cost_volume_to_probabilistic_mapping(correlation_matrix_t_to_s)

        P_warp_supervision = torch.flatten(P_warp_supervision, start_dim=2).permute(0, 2, 1).contiguous()\
            .view(b*h*w, -1)
        # b*h_t*w_t, h_s*w_s

        # supervise positive
        P_warp_supervision = P_warp_supervision[mask_for_labels]

        stats = self.get_stats(logits=P_warp_supervision, target=labels, name='super', h=h, w=w)

        stats['avg_proba_in_valid_mask_super'] = \
            P_warp_supervision.detach()[torch.arange(0, P_warp_supervision.shape[0]), labels].mean()

        # cross entropy loss
        logits = torch.log(P_warp_supervision + 1e-7)
        index = torch.arange(0, logits.shape[0])
        loss = - logits[index, labels]
        loss = loss.mean()
        stats['kp_loss_{}x{}'.format(h, w)] = loss.item()

        return loss, stats


class SparseSupervisedOneHotEMD(SupervisionStrategy):
    """
    Based on keypoint annotations, creates a ground-truth one-hot probability map for each. From the estimated
    cost volume, computes the EMD loss between estimated and ground-truth probability map for each key-point.
    """
    def __init__(self, objective=None, activation='softmax', temperature=1.0, interpolate_position=True):
        self.num_negatives = 0
        self.objective = objective
        self.activation = activation
        self.target_rate = 0.5
        self.temperature = temperature
        self.loss_func = objective  # treats each entry as a binary classification
        self.interpolate_position = interpolate_position

    def get_image_pair(self, batch, *args):
        """Only requires positive image pairs"""
        self.bsz = len(batch['source_image'])
        return batch['source_image'], batch['target_image']

    def get_correlation(self, correlation_matrix):
        """Returns correlation matrices of 'POSITIVE PAIRS' in a batch"""
        return correlation_matrix[:self.bsz].clone().detach()

    @staticmethod
    def cost_volume_to_probabilistic_mapping(self, A):
        """ Convert cost volume to probabilistic mapping.
        Args:
            A: cost volume, dimension B x C x H x W, matching points are in C
        """
        return cost_volume_to_probabilistic_mapping(A, self.activation, self.temperature)

    def cross_entropy_loss2d(self, correlation_matrix_t_to_s, source_kp, target_kp, n_valid_pts,
                             original_shape):
        """
        Args:
            correlation_matrix_t_to_s: correlation volume in target coordinate system. shape is b, h_s*w_s, h_t, w_t.
            target_kp: kp in target, shape B, N, 2. only n_pts of valid ones, the others have coordinates below 0.
            source_kp: kp in source, shape B, N, 2. only n_pts of valid ones, the others have coordinates below 0.
            original_shape: at which the keypoints where extracted (H, W)
        Returns:
            loss
            stats
        """

        assert len(correlation_matrix_t_to_s.shape) == 4

        b, c, h, w = correlation_matrix_t_to_s.shape

        stats = {}
        featsShape = [h, w]

        loss = 0.
        xx = torch.arange(0, w).view(1, -1).repeat(h, 1)
        yy = torch.arange(0, h).view(-1, 1).repeat(1, w)
        xx = xx.view(h, w, 1)
        yy = yy.view(h, w, 1)
        grid = torch.cat((xx, yy), -1).float().cuda().view(-1, 2)  # h*w, 2
        # per image in the batch
        for i in range(b):

            kps_src = source_kp[i][:n_valid_pts[i]]
            kps_trg = target_kp[i][:n_valid_pts[i]]

            # resize to this
            kps_src[:, 0] *= w / original_shape[1]
            kps_src[:, 1] *= h / original_shape[0]
            kps_trg[:, 0] *= w / original_shape[1]
            kps_trg[:, 1] *= h / original_shape[0]

            # check that the keypoints are within the array
            if self.interpolate_position:
                # check that the keypoints are within the array
                mask_corr_valid = kps_src[:, 0].ge(0) & kps_trg[:, 0].ge(0) & \
                                  kps_src[:, 1].ge(0) & kps_trg[:, 1].ge(0) & \
                                  kps_src[:, 0].lt(w) & kps_trg[:, 0].lt(w) & \
                                  kps_src[:, 1].lt(h) & kps_trg[:, 1].lt(h)
            else:
                # check that the keypoints are within the array
                mask_corr_valid = kps_src[:, 0].ge(0) & kps_trg[:, 0].ge(0) & \
                                  kps_src[:, 1].ge(0) & kps_trg[:, 1].ge(0) & \
                                  torch.round(kps_src[:, 0]).lt(w) & torch.round(kps_trg[:, 0]).lt(w) & \
                                  torch.round(kps_src[:, 1]).lt(h) & torch.round(kps_trg[:, 1]).lt(h)

            kps_src = kps_src[mask_corr_valid]
            kps_trg = kps_trg[mask_corr_valid]

            if kps_src.shape[0] > 0:
                # there are some valid kp
                kps_src = torch.t(kps_src).cuda()
                kps_trg = torch.t(kps_trg).cuda()
                weights = torch.zeros(kps_src.shape[1], c).cuda()
                distance_maps = torch.zeros(kps_src.shape[1], c).cuda()

                kps_src_t = torch.t(kps_src)
                for j in range(0, kps_src.shape[1]):
                    if self.interpolate_position:
                        weights[j] = geometry.BilinearInterpolate(
                            [kps_trg[0, j], kps_trg[1, j]], correlation_matrix_t_to_s[i], featsShape)
                        # h*w, could be h*w+1
                    else:
                        correlation_matrix_t_to_s_at_j = correlation_matrix_t_to_s[i].view(-1, h, w).permute(1, 2, 0)
                        # h_t, w_t, h_s*w_s
                        weights[j] = correlation_matrix_t_to_s_at_j[torch.round(kps_trg[1, j]).long(),
                                                                    torch.round(kps_trg[0, j]).long()]

                    # the probabilisty map corresponding to the source kp ==> gt for prediction at target kp.
                    distance_map_j = grid - kps_src_t[j].view(1, -1)
                    distance_map_j = torch.norm(distance_map_j, dim=-1, p=2).view(1, -1)  # 1, h*w
                    if c == h*w + 1:
                        # there is also a bin, the distance is a constant, equal to max dim
                        distance_map_j = torch.cat([distance_map_j, torch.as_tensor(max(h, w)).cuda().view(1, -1)], dim=-1)
                    distance_maps[j] = distance_map_j.view(-1)

                loss += (self.cost_volume_to_probabilistic_mapping(weights) * distance_maps).sum() / kps_src.shape[1]

        loss = loss / b
        stats['kp_loss_{}x{}'.format(h, w)] = loss.item()
        return loss, stats

    def compute_loss(self, correlation_matrix, t_to_s=True, batch=None, *args, **kwargs):
        """
        Args:
            correlation_matrix: correlation volume, can be in target or source coordinate system.
            t_to_s: bool, if Tue, correlation_matrix has shape b, h_s*w_s, h_t, w_t.
            batch: must at least contain the fields 'target_kps', 'source_kps', 'n_pts', 'target_image'.
        Returns:
            loss
            stats
        """

        if len(correlation_matrix.shape) == 3:
            hw = correlation_matrix.shape[-1]
            h = w = int(math.sqrt(hw))
        else:
            h, w = correlation_matrix.shape[-2:]
        b = correlation_matrix.shape[0]
        correlation_matrix = correlation_matrix.view(b, -1, h, w)

        if t_to_s:
            return self.cross_entropy_loss2d(correlation_matrix, target_kp=batch['target_kps'].clone(),
                                             source_kp=batch['source_kps'].clone(),
                                             n_valid_pts=batch['n_pts'],
                                             original_shape=batch['target_image'].shape[-2:])
        else:
            return self.cross_entropy_loss2d(correlation_matrix, source_kp=batch['target_kps'].clone(),
                                             target_kp=batch['source_kps'].clone(), n_valid_pts=batch['n_pts'],
                                             original_shape=batch['target_image'].shape[-2:])


class SparseSupervisedFlowEPE(SupervisionStrategy):
    """
    Based on keypoint annotations, computes the EPE loss.
    """
    def __init__(self, loss_module):
        if loss_module is None:
            objective = EPE()
            weights_level_loss = [0.01]
            loss_module = MultiScaleFlow(level_weights=weights_level_loss, loss_function=objective,
                                         downsample_gt_flow=False)
        self.loss_module = loss_module

    def get_image_pair(self, batch, *args):
        """Only requires positive image pairs"""
        self.bsz = len(batch['source_image'])
        return batch['source_image'], batch['target_image']

    def get_correlation(self, correlation_matrix):
        """Returns correlation matrices of 'POSITIVE PAIRS' in a batch"""
        return correlation_matrix[:self.bsz].clone().detach()

    def compute_loss(self, est_flow_t_to_s, batch=None, *args, **kwargs):
        """
        Args:
            est_flow_t_to_s: estimated flow from target to source.
            batch:
            *args:
            **kwargs:

        Returns:

        """
        if batch is None:
            raise ValueError
        if est_flow_t_to_s is None:
            raise ValueError

        mask_gt_target_to_source = batch['correspondence_mask_target_to_source']
        flow_gt_target_to_source = batch['flow_map_target_to_source']
        return self.loss_module(est_flow_t_to_s, flow_gt_target_to_source, mask=mask_gt_target_to_source)
