import torch.nn.functional as F
import torch


class MultiScaleFlow:
    """ Module for multi-scale matching loss computation.
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
                    mask = F.interpolate(mask.float(), (h, w), mode='bilinear', align_corners=False).byte()
                    # round not to include the borders
                    mask = mask.bool() if float(torch.__version__[:3]) >= 1.1 else mask.byte()
        else:
            b, _, h, w = gt_flow.shape
            # upsample output to ground truth flow load_size
            est_flow = F.interpolate(est_flow, (h, w), mode='bilinear', align_corners=False)
            if mask is not None:
                mask = mask.unsqueeze(1)
                if mask.shape[2] != h or mask.shape[3] != w:
                    mask = F.interpolate(mask.float(), (h, w), mode='bilinear', align_corners=False).byte()
                    # round not to include the borders
                    mask = mask.bool() if float(torch.__version__[:3]) >= 1.1 else mask.byte()

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
            # from smallest load_size to biggest load_size (last one is a quarter of input image load_size
            b, _, h, w = flow.shape
            if mask is not None and isinstance(mask, list):
                mask_used = mask[i]
            else:
                mask_used = mask
            level_loss = weight * self.one_scale(flow, gt_flow, mask=mask_used)
            stats['loss_reso_{}x{}/loss_level'.format(h, w)] = level_loss
            loss += level_loss
        return loss, stats


class MultiScaleSingleDensity:
    """ Module for multi-scale matching loss computation, when the flow regression is modeled with a
    single probability density (ex: Gaussian or Laplace density).
    The loss is computed at all estimated flow resolutions and weighted according to level_weights. """

    def __init__(self, level_weights, loss_function, downsample_gt_flow, supervise_uncertainty_number=[0, 1, 2]):
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
        self.supervise_uncertainty_number = supervise_uncertainty_number

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
                mask = F.interpolate(mask.float().unsqueeze(1), (h, w), mode='bilinear', align_corners=False).byte()
                mask = mask.bool() if float(torch.__version__[:3]) >= 1.1 else mask.byte()
        else:
            b, _, h, w = gt_flow.shape
            # upsample output to ground truth flow load_size
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
        for flow, weight, uncertainty in zip(flow_output, self.level_weights, uncertainty_estimate):
            b, _, h, w = flow.shape
            level_loss = weight * self.one_scale(flow, gt_flow, uncertainty, mask=mask)
            stats['loss_reso_{}x{}/loss_level'.format(h, w)] = level_loss
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
                mask = F.interpolate(mask.float().unsqueeze(1), (h, w), mode='bilinear', align_corners=False).byte()
                mask = mask.bool() if float(torch.__version__[:3]) >= 1.1 else mask.byte()
        else:
            b, _, h, w = gt_flow.shape
            # upsample output to ground truth flow load_size
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
            level_loss = weight * self.one_scale(flow, gt_flow, log_var_map, weight_map, mask=mask)

            stats['loss_reso_{}x{}/loss_level'.format(h, w)] = level_loss
            loss += level_loss
        return loss, stats


class MultiScaleMultipleMixtureDensity:
    """ Module for multi-scale matching loss computation, when the flow regression is modeled with a
    mixture probability density.
    The loss is computed at all estimated flow resolutions and weighted according to level_weights. """
    def __init__(self, level_weights, loss_function, downsample_gt_flow, supervise_uncertainty_number=[0, 1, 2]):
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
        self.supervise_uncertainty_number = supervise_uncertainty_number

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
                mask = F.interpolate(mask.float().unsqueeze(1), (h, w), mode='bilinear', align_corners=False).byte()
                mask = mask.bool() if float(torch.__version__[:3]) >= 1.1 else mask.byte()
        else:
            b, _, h, w = gt_flow.shape
            # upsample output to ground truth flow load_size
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
            log_var_map = uncertainty_list[0] # this is a list containing multiple uncertainties
            weight_map = uncertainty_list[1]

            level_loss = 0
            for index in self.supervise_uncertainty_number:
                l_level = self.one_scale(flow, gt_flow, log_var_map[index], weight_map[index], mask=mask)
                level_loss += l_level
                stats['loss_reso_{}x{}_index_{}/loss_level'.format(h, w, index)] = weight * l_level

            b, _, h, w = flow.shape
            stats['loss_reso_{}x{}/loss_level'.format(h, w)] = weight * level_loss
            loss += weight * level_loss

        return loss, stats


'''
def multiscaleOcc(estimated_occlusion, gt_occlusion, sparse=False, mask=None, weights=None):
    def one_scale(estimated_occlusion, gt_occlusion, sparse,  mask=None, mean=False):
        b, _, h, w = estimated_occlusion.load_size()
        if sparse:
            gt_occlusion_scaled = sparse_max_pool(gt_occlusion.float(), (h, w))

            if mask is not None:
                mask = sparse_max_pool(mask.float().unsqueeze(1), (h, w)).byte()
        else:
            gt_occlusion_scaled = F.adaptive_avg_pool2d(gt_occlusion.float(), [h, w])
            if mask is not None:
                mask = F.interpolate(mask.float().unsqueeze(1), (h, w), mode='bilinear').byte()
        if mask is not None:
            return f1_score_bal_loss(estimated_occlusion[mask], gt_occlusion_scaled[mask])
        else:
            return f1_score_bal_loss(estimated_occlusion, gt_occlusion_scaled)

    occ_loss = 0
    for output_occlusion, weight in zip(estimated_occlusion, weights):
        output_occlusion = F.sigmoid(output_occlusion)
        occ_loss += weight * one_scale(output_occlusion,  gt_occlusion, sparse, mask=mask)
    return occ_loss



def multiscale_loss_upsampling_flow(network_output, target_flow, robust_L1_loss=False, mask=None, weights=None, ratio_x=None, ratio_y=None,
                                    sparse=False, mean=False):

    def one_scale(output, target, sparse, ratio_x=None, ratio_y=None, robust_L1_loss=False, mask=None, mean=False):
        # mask is bxhxw, unsqueeze then is bx1xhxw
        b, _, h, w = target.load_size()
        if ratio_x is not None and ratio_y is not None:
            upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
            upsampled_output[:, 0, :, :] *= ratio_x
            upsampled_output[:, 1, :, :] *= ratio_y
        else:
            upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
        # mask = F.interpolate(mask.float().unsqueeze(1), (h, w), mode='bilinear').byte()
        mask = mask.float().unsqueeze(1) # before it was b,h,w
        if robust_L1_loss:
            if mask is not None:
                return L1_charbonnier_loss(upsampled_output * mask.float(), target * mask.float(), sparse,
                                           mean=mean, sum=False)
            else:
                return L1_charbonnier_loss(upsampled_output, target, sparse, mean=mean, sum=False)
        else:
            if mask is not None:
                return EPE(upsampled_output * mask.float(), target * mask.float(), sparse, mean=mean, sum=False)
            else:
                return EPE(upsampled_output, target, sparse, mean=mean, sum=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [1.0, 1.0, 1.0, 1.0]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        # from smallest load_size to biggest load_size (last one is a quarter of input image load_size
        loss += weight * one_scale(output, target_flow, sparse, ratio_x=ratio_x, ratio_y=ratio_y,
                                   robust_L1_loss=robust_L1_loss, mask=mask, mean=mean)
    return loss


class MultiScaleEPEAndOcc:
    def __init__(self):
    
    def __call__(self, *args, **kwargs):
        occ_loss = multiscaleOcc(occlusion_mask_list, gt_occlusion_mask, weights=loss_grid_weights,
                                 sparse=False)
        f_loss = Loss.detach()
        o_loss = occ_loss.detach()
        if (f_loss.data > o_loss.data).numpy:
            f_l_w = 1
            o_l_w = f_loss / o_loss / ratio_occlusion_loss
        else:
            f_l_w = o_loss / f_loss * ratio_occlusion_loss
            o_l_w = 1

        Loss = (Loss * f_l_w + occ_loss * o_l_w)
'''