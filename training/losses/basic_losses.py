import torch
import torch.nn.functional as F


def sparse_max_pool(input, size):
    '''Downsample the input by considering 0 values as invalid.
    Unfortunately, no generic interpolation mode can resize a sparse map correctly,
    the strategy here is to use max pooling for positive values and "min pooling"
    for negative values, the two results are then summed.
    This technique allows sparsity to be minized, contrary to nearest interpolation,
    which could potentially lose information for isolated data points.'''

    positive = (input > 0).float()
    negative = (input < 0).float()
    output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
    return output


def realEPE(output, target, mask_gt, ratio_x=None, ratio_y=None):
    """
    Computes real EPE: the network output (quarter of original resolution) is upsampled to the size of
    the target (and scaled if necessary by ratio_x and ratio_y), it should be equal to target flow.
    Args:
        output: estimated flow field, shape (b,2,h,w)
        target: ground-truth flow field, shape (b,2,H,W)
        mask_gt: valid mask according to which the average end point error is computed, shape (b,H,W)
        ratio_x: multiplicative factor
        ratio_y: multiplicative factor

    Returns:
        EPE: Average End Point Error over all valid pixels of the batch
    """
    # mask_gt in shape bxhxw, can be torch.byte or torch.uint8 or torch.int
    b, _, h, w = target.size()
    if ratio_x is not None and ratio_y is not None:
        upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
        upsampled_output[:, 0, :, :] *= ratio_x
        upsampled_output[:, 1, :, :] *= ratio_y
    else:
        upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
    # output interpolated to original size (supposed to be in the right range then)

    flow_target_x = target.permute(0, 2, 3, 1)[:, :, :, 0]
    flow_target_y = target.permute(0, 2, 3, 1)[:, :, :, 1]
    flow_est_x = upsampled_output.permute(0, 2, 3, 1)[:, :, :, 0]  # BxH_xW_
    flow_est_y = upsampled_output.permute(0, 2, 3, 1)[:, :, :, 1]

    flow_target = \
        torch.cat((flow_target_x[mask_gt].unsqueeze(1),
                   flow_target_y[mask_gt].unsqueeze(1)), dim=1)
    flow_est = \
        torch.cat((flow_est_x[mask_gt].unsqueeze(1),
                   flow_est_y[mask_gt].unsqueeze(1)), dim=1)
    EPE = torch.norm(flow_est-flow_target, 2, 1)
    return EPE.mean()


def real_metrics(output, target, mask_gt, ratio_x=None, ratio_y=None, thresh_1=1.0, thresh_2=3.0, thresh_3=5.0):
    """
    Computes real EPE, PCK-1, PCK-3 and PCK-3:
    the network output (quarter of original resolution) is upsampled to the size of
    the target (and scaled if necessary by ratio_x and ratio_y), it should be equal to target flow.
    Args:
        output: estimated flow field, shape (b,2,h,w)
        target: ground-truth flow field, shape (b,2,H,W)
        mask_gt: valid mask according to which the average end point error is computed, shape (b,H,W)
        ratio_x: multiplicative factor
        ratio_y: multiplicative factor

    Returns:
        EPE: Average End Point Error over all valid pixels of the batch
        PCK-1: Percentage of Correct Correspondences for a pixel threshold of 1, over all valid pixels of the batch
        PCK-3: Percentage of Correct Correspondences for a pixel threshold of 3, over all valid pixels of the batch
        PCK-5: Percentage of Correct Correspondences for a pixel threshold of 5, over all valid pixels of the batch
    """
    # mask_gt in shape bxhxw, can be torch.byte or torch.uint8 or torch.int
    b, _, h, w = target.size()
    if ratio_x is not None and ratio_y is not None:
        upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
        upsampled_output[:, 0, :, :] *= ratio_x
        upsampled_output[:, 1, :, :] *= ratio_y
    else:
        upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
    # output interpolated to originale size (supposed to be in the right range then)

    flow_target_x = target.permute(0, 2, 3, 1)[:, :, :, 0]
    flow_target_y = target.permute(0, 2, 3, 1)[:, :, :, 1]
    flow_est_x = upsampled_output.permute(0, 2, 3, 1)[:, :, :, 0]  # BxH_xW_
    flow_est_y = upsampled_output.permute(0, 2, 3, 1)[:, :, :, 1]

    flow_target = \
        torch.cat((flow_target_x[mask_gt].unsqueeze(1),
                   flow_target_y[mask_gt].unsqueeze(1)), dim=1)
    flow_est = \
        torch.cat((flow_est_x[mask_gt].unsqueeze(1),
                   flow_est_y[mask_gt].unsqueeze(1)), dim=1)
    EPE = torch.norm(flow_est-flow_target, 2, 1)
    PCK_1 = EPE.le(thresh_1).float()
    PCK_3 = EPE.le(thresh_2).float()
    PCK_5 = EPE.le(thresh_3).float()
    return EPE.mean(), PCK_1.mean(), PCK_3.mean(), PCK_5.mean()


class EPE:
    """Compute EPE loss. """
    def __init__(self, reduction='weighted_sum_normalized'):
        """
        Args:
            reduction: str, type of reduction to apply to loss
        """
        super().__init__()
        self.reduction = reduction

    def __call__(self, gt_flow, est_flow, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            mask: valid mask, where the loss is computed. shape (b, 1, H, W)
        """
        b, _, h, w = gt_flow.shape
        EPE_map = torch.norm(gt_flow-est_flow, 2, 1, keepdim=True)
    
        if mask is not None:
            mask = ~torch.isnan(EPE_map.detach()) & ~torch.isinf(EPE_map.detach()) & mask
        else:
            mask = ~torch.isnan(EPE_map.detach()) & ~torch.isinf(EPE_map.detach())

        if self.reduction == 'mean':
            if mask is not None:
                loss = torch.masked_select(EPE_map, mask).mean()
            else:
                loss = EPE_map.mean()
            return loss
        elif 'weighted_sum' in self.reduction:
            if mask is not None:
                EPE_map = EPE_map * mask.float()
                L = 0
                for bb in range(0, b):
                    norm_const = float(h*w) / (mask[bb, ...].sum().float() + 1e-6)
                    L += EPE_map[bb][mask[bb, ...] != 0].sum() * norm_const
                if 'normalized' in self.reduction:
                    return L / b
                return L

            if 'normalized' in self.reduction:
                return EPE_map.sum()/b
            return EPE_map
        else:
            raise ValueError


class L1:
    """ Computes L1 loss. """
    def __init__(self, reduction='weighted_sum_normalized'):
        """
        Args:
            reduction: str, type of reduction to apply to loss
        """
        super().__init__()
        self.reduction = reduction

    def __call__(self, gt_flow, est_flow, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            mask: valid mask, where the loss is computed. shape (b, 1, H, W)
        """
        b, _, h, w = gt_flow.shape
        L1 = torch.sum(torch.abs(est_flow-gt_flow), 1, keepdim=True)

        if mask is not None:
            mask = ~torch.isnan(L1.detach()) & ~torch.isinf(L1.detach()) & mask
        else:
            mask = ~torch.isnan(L1.detach()) & ~torch.isinf(L1.detach())

        if self.reduction == 'mean':
            if mask is not None:
                loss = torch.masked_select(L1, mask).mean()
            else:
                loss = L1.mean()
            return loss
        elif 'weighted_sum' in self.reduction:
            if mask is not None:
                L1 = L1 * mask.float()
                L = 0
                for bb in range(0, b):
                    norm_const = float(h*w) / (mask[bb, ...].sum().float() + 1e-6)
                    L += L1[bb][mask[bb, ...] != 0].sum() * norm_const
                if 'normalized' in self.reduction:
                    return L / b
                return L

            if 'normalized' in self.reduction:
                return L1.sum()/b
            return L1
        else:
            raise ValueError


class L1Charbonnier:
    """Computes L1 Charbonnier loss. """
    def __init__(self, reduction='weighted_sum_normalized'):
        """
        Args:
            reduction: str, type of reduction to apply to loss
        """
        super().__init__()
        self.reduction = reduction

    def __call__(self, gt_flow, est_flow, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            mask: valid mask, where the loss is computed. shape (b, 1, H, W)
        """
        b, _, h, w = gt_flow.shape
        epsilon = 0.01
        alpha = 0.4
        L1 = torch.sum(torch.abs(est_flow - gt_flow), 1, keepdim=True)
        norm = torch.pow(L1 + epsilon, alpha)

        if mask is not None:
            mask = ~torch.isnan(norm.detach()) & ~torch.isinf(norm.detach()) & mask
        else:
            mask = ~torch.isnan(norm.detach()) & ~torch.isinf(norm.detach())

        if self.reduction == 'mean':
            if mask is not None:
                loss = torch.masked_select(norm, mask).mean()
            else:
                loss = norm.mean()
            return loss
        elif 'weighted_sum' in self.reduction:
            if mask is not None:
                norm = norm * mask.float()
                L = 0
                for bb in range(0, b):
                    norm_const = float(h*w) / (mask[bb, ...].sum().float() + 1e-6)
                    L = L + norm[bb][mask[bb, ...] != 0].sum() * norm_const
                if 'normalized' in self.reduction:
                    return L / b
                return L

            if 'normalized' in self.reduction:
                return norm.sum() / b
            return norm
        else:
            raise ValueError
