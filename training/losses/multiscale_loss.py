import torch.nn.functional as F
import torch
from packaging import version


class MultiScaleFlow:
    """ Module for multi-scale matching loss computation.
    The loss is computed at all estimated flow resolutions and weighted according to level_weights. """

    def __init__(self, level_weights, loss_function, downsample_gt_flow: bool):
        """
        Args:
            level_weights: weights to apply to computed loss at each level (from coarsest to finest pyramid level)
            loss_function: actual loss computation module, used for all levels
            downsample_gt_flow: bool, downsample gt flow to estimated flow resolution? otherwise, the estimated flow
                                of each level is instead up-sampled to ground-truth resolution for loss computation
        """
        self.level_weights = level_weights
        self.loss_function = loss_function
        self.downsample_gt_flow = downsample_gt_flow

    def one_scale(self, est_flow, gt_flow, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            mask: valid mask, where the loss is computed. shape (b, H, W)
        """
        if self.downsample_gt_flow:
            b, _, h, w = est_flow.size()
            gt_flow = F.interpolate(gt_flow, (h, w), mode='bilinear', align_corners=False)
            if mask is not None:
                mask = mask.unsqueeze(1)
                if mask.shape[2] != h or mask.shape[3] != w:
                    mask = F.interpolate(mask.float(), (h, w), mode='bilinear', align_corners=False).floor()
                    # floor not to include the borders
                    mask = mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask.byte()
        else:
            b, _, h, w = gt_flow.shape
            # upsample output to ground truth flow size
            est_flow = F.interpolate(est_flow, (h, w), mode='bilinear', align_corners=False)
            if mask is not None:
                mask = mask.unsqueeze(1)
                if mask.shape[2] != h or mask.shape[3] != w:
                    mask = F.interpolate(mask.float(), (h, w), mode='bilinear', align_corners=False).floor()
                    # floor not to include the borders
                    mask = mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask.byte()

        return self.loss_function(est_flow, gt_flow, mask=mask)

    def __call__(self, network_output, gt_flow, mask=None):
        """
        Args:
            network_output: network predictions, can either be a dictionary, where network_output['flow_estimates']
                            is a list containing the estimated flow fields at all levels. or directly the list 
                            of flow fields.
            gt_flow: ground-truth flow field
            mask: bool tensor, valid mask, 1 indicates valid pixels where the loss is computed.

        Returns:
            loss: computed loss
            stats: dict with stats from the loss computation

        """

        if isinstance(network_output, dict):
            # it is a dictionary, extract the flow estimates
            flow_output = network_output['flow_estimates']
        else:
            flow_output = network_output  # the flow was directly given
        if type(flow_output) not in [tuple, list]:
            flow_output = [flow_output]

        assert(len(self.level_weights) == len(flow_output))

        loss = 0
        stats = {}
        for i, (flow, weight) in enumerate(zip(flow_output, self.level_weights)):
            # from smallest size to biggest size (last one is a quarter of input image size
            b, _, h, w = flow.shape
            if mask is not None and isinstance(mask, list):
                mask_used = mask[i]
            else:
                mask_used = mask
            level_loss = weight * self.one_scale(flow, gt_flow, mask=mask_used)
            stats['loss_reso_{}x{}/loss_level'.format(h, w)] = level_loss.item()
            loss += level_loss
        return loss, stats


class MultiScaleSingleDensity:
    """ Module for multi-scale matching loss computation, when the flow regression is modeled with a
    single probability density (ex: Gaussian or Laplace density).
    The loss is computed at all estimated flow resolutions and weighted according to level_weights. """

    def __init__(self, level_weights, loss_function, downsample_gt_flow):
        """
        Args:
            level_weights: weights to apply to computed loss at each level (from coarsest to finest pyramid level)
            loss_function: actual loss computation module, used for all levels
            downsample_gt_flow: bool, downsample gt flow to estimated flow resolution? otherwise, the estimated flow
                                of each level is instead up-sampled to ground-truth resolution for loss computation
        """
        self.level_weights = level_weights
        self.loss_function = loss_function
        self.downsample_gt_flow = downsample_gt_flow

    def one_scale(self, est_flow, gt_flow, log_var_uncertainty, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            log_var_uncertainty: estimated log variance, shape (b, 1, H, W)
            mask: valid mask, where the loss is computed. shape (b, H, W)
        """
        if self.downsample_gt_flow:
            b, _, h, w = est_flow.size()
            gt_flow = F.interpolate(gt_flow, (h, w), mode='bilinear', align_corners=False)
            if mask is not None:
                mask = F.interpolate(mask.float().unsqueeze(1), (h, w), mode='bilinear', align_corners=False).floor()
                mask = mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask.byte()
        else:
            b, _, h, w = gt_flow.shape
            # upsample output to ground truth flow size
            est_flow = F.interpolate(est_flow, (h, w), mode='bilinear', align_corners=False)
            log_var_uncertainty = F.interpolate(log_var_uncertainty, (h, w), mode='bilinear', align_corners=False)
            if mask is not None:
                mask = mask.unsqueeze(1)
        return self.loss_function(est_flow, gt_flow, log_var_uncertainty, mask=mask)

    def __call__(self, network_output, gt_flow, mask=None):
        """
        Args:
            network_output: network predictions, it is a dictionary. must contain fields 'flow_estimates' and
                            'uncertainty_estimates'
            gt_flow: ground-truth flow field
            mask: bool tensor, valid mask, 1 indicates valid pixels where the loss is computed.

        Returns:
            loss: computed loss
            stats: dict with stats from the loss computation

        """
        if isinstance(network_output, dict):
            # it is a dictionary, extract the flow estimates
            flow_output = network_output['flow_estimates']
        else:
            flow_output = network_output  # the flow was directly given
        if type(flow_output) not in [tuple, list]:
            flow_output = [flow_output]
        assert(len(self.level_weights) == len(flow_output))
        uncertainty_estimate = network_output['uncertainty_estimates']

        stats = {}
        loss = 0
        for i, (flow, weight, uncertainty) in enumerate(zip(flow_output, self.level_weights, uncertainty_estimate)):
            b, _, h, w = flow.shape
            if mask is not None and isinstance(mask, list):
                mask_used = mask[i]
            else:
                mask_used = mask
            level_loss = weight * self.one_scale(flow, gt_flow, uncertainty, mask=mask_used)
            stats['loss_reso_{}x{}/loss_level'.format(h, w)] = level_loss.item()
            loss += level_loss
        return loss, stats


class MultiScaleMixtureDensity:
    """ Module for multi-scale matching loss computation, when the flow regression is modeled with a
    mixture probability density.
    The loss is computed at all estimated flow resolutions and weighted according to level_weights. """
    def __init__(self, level_weights, loss_function, downsample_gt_flow):
        """
        Args:
            level_weights: weights to apply to computed loss at each level (from coarsest to finest pyramid level)
            loss_function: actual loss computation module, used for all levels
            downsample_gt_flow: bool, downsample gt flow to estimated flow resolution ? otherwise, the estimated flow
                                of each level is instead up-sampled to ground-truth resolution for loss computation
        """
        self.level_weights = level_weights
        self.loss_function = loss_function
        self.downsample_gt_flow = downsample_gt_flow

    def one_scale(self, est_flow, gt_flow, log_var_map, weight_map, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            log_var_map: estimated log variances, shape (b, nbr_components, H, W)
            weight_map: estimated weights map (alpha) for the mixture model, each corresponding to a log variance,
                        shape (b, nbr_components, H, W)
            mask: valid mask, where the loss is computed. shape (b, H, W)
        """
        if self.downsample_gt_flow:
            b, _, h, w = est_flow.size()
            gt_flow = F.interpolate(gt_flow, (h, w), mode='bilinear', align_corners=False)
            if mask is not None:
                mask = F.interpolate(mask.float().unsqueeze(1), (h, w), mode='bilinear', align_corners=False).floor()
                mask = mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask.byte()
        else:
            b, _, h, w = gt_flow.shape
            # upsample output to ground truth flow size
            est_flow = F.interpolate(est_flow, (h, w), mode='bilinear', align_corners=False)
            log_var_map = F.interpolate(log_var_map, (h, w), mode='bilinear', align_corners=False)
            weight_map = F.interpolate(weight_map, (h, w), mode='bilinear', align_corners=False)
            if mask is not None:
                mask = mask.unsqueeze(1)
        return self.loss_function(est_flow, gt_flow, log_var_map, weight_map, mask=mask)

    def __call__(self, network_output, gt_flow, mask=None):
        """
        Args:
            network_output: network predictions, it is a dictionary. must contain fields 'flow_estimates' and
                            'uncertainty_estimates'
            gt_flow: ground-truth flow field
            mask: bool tensor, valid mask, 1 indicates valid pixels where the loss is computed.

        Returns:
            loss: computed loss
            stats: dict with stats from the loss computation

        """

        if isinstance(network_output, dict):
            # it is a dictionary, extract the flow estimates
            flow_output = network_output['flow_estimates']
        else:
            flow_output = network_output  # the flow was directly given
        if type(flow_output) not in [tuple, list]:
            flow_output = [flow_output]

        assert(len(self.level_weights) == len(flow_output))
        uncertainty_estimate = network_output['uncertainty_estimates']

        stats = {}
        loss = 0
        for level, (flow, weight, uncertainty_list) in enumerate(zip(flow_output, self.level_weights,
                                                                     uncertainty_estimate)):
            log_var_map = uncertainty_list[0]
            weight_map = uncertainty_list[1]
            b, _, h, w = flow.shape
            if mask is not None and isinstance(mask, list):
                mask_used = mask[level]
            else:
                mask_used = mask
            level_loss = weight * self.one_scale(flow, gt_flow, log_var_map, weight_map, mask=mask_used)

            stats['loss_reso_{}x{}/loss_level'.format(h, w)] = level_loss.item()
            loss += level_loss
        return loss, stats

