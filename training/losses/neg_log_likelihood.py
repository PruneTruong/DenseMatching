import torch
import numpy as np
import math


class NLLMixtureLaplace:
    """ Computes Negative Log Likelihood loss for a Mixture of Laplace distributions. """
    def __init__(self, sum_normalized=True, ratio=1.0):
        """
        Args:
            sum_normalized: bool, compute the sum over tensor and divide by number of image pairs per batch?
            ratio:
        """
        super().__init__()
        self.sum_normalized = sum_normalized
        self.ratio = ratio

    def __call__(self, gt_flow, est_flow, log_var, weight_map, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            log_var: estimated log variances, shape (b, nbr_components, H, W)
            weight_map: estimated weights map (alpha) for the mixture model, each corresponding to a log variance,
                        shape (b, nbr_components, H, W)
            mask: valid mask, where the loss is computed. shape (b, 1, H, W)
        """
        b, _, h, w = gt_flow.shape
        l1 = torch.logsumexp(weight_map, 1, keepdim=True)
        # shape will be b,1,h,w

        reg = math.sqrt(2)*torch.sum(torch.abs((gt_flow - est_flow)*self.ratio), 1, keepdim=True)  # shape will be b,1,h,w
        exponent = weight_map - math.log(2) - log_var - reg * torch.exp(-0.5*log_var)
        l2 = torch.logsumexp(exponent, 1, keepdim=True)

        loss = l1 - l2

        if mask is not None:
            mask = ~torch.isnan(loss.detach()) & ~torch.isinf(loss.detach()) & mask
        else:
            mask = ~torch.isnan(loss.detach()) & ~torch.isinf(loss.detach())

        if torch.isnan(loss.detach()).sum().ge(1) or torch.isinf(loss.detach()).sum().ge(1):
            # debugging
            print('mask or inf in the loss ! ')

        if mask is not None:
            loss = loss * mask.float()
            L = 0
            for bb in range(0, b):
                loss[bb, ...][mask[bb, ...] == 0] = loss[bb, ...][mask[bb, ...] == 0].detach()
                norm_const = float(h)*float(w) / (mask[bb, ...].sum().float() + 1e-6)
                L = L + loss[bb][mask[bb, ...] != 0].sum() * norm_const
            if self.sum_normalized:
                return L / b
            return L

        if self.sum_normalized:
            return loss.sum() / b
        return loss.sum()


class NLLMixtureGaussian:
    """ Computes Negative Log Likelihood loss for a Mixture of Gaussian distributions. """
    def __init__(self, sum_normalized=True, ratio=1.0):
        """
        Args:
            sum_normalized: bool, compute the sum over tensor and divide by number of image pairs per batch ?
            ratio:
        """
        super().__init__()
        self.sum_normalized = sum_normalized
        self.ratio = ratio

    def __call__(self, gt_flow, est_flow, log_var, weight_map, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            log_var: estimated log variances, shape (b, nbr_components, H, W)
            weight_map: estimated weights map (alpha) for the mixture model, each corresponding to a log variance,
                        shape (b, nbr_components, H, W)
            mask: valid mask, where the loss is computed. shape (b, 1, H, W)
        """
        b, _, h, w = gt_flow.shape
        PI = torch.tensor(np.pi).cuda()
        l1 = torch.logsumexp(weight_map, 1, keepdim=True)

        reg = 0.5 * torch.sum((gt_flow - est_flow) ** 2, 1, keepdim=True)
        exponent = weight_map - 0.5 * torch.log(2 * PI) - 0.5 * log_var - reg * torch.exp(-log_var)
        l2 = torch.logsumexp(exponent, 1, keepdim=True)
        loss = l1 - l2

        if mask is not None:
            loss = loss * mask.float()
            L = 0
            for bb in range(0, b):
                loss[bb, ...][mask[bb, ...] == 0] = loss[bb, ...][mask[bb, ...] == 0].detach()
                norm_const = float(h*w) / (mask[bb, ...].sum().float() + 1e-6)
                L = L + loss[bb][mask[bb, ...] != 0].sum() * norm_const
            if self.sum_normalized:
                return L / b
            else:
                return L

        if self.sum_normalized:
            return loss.sum() / b
        else:
            return loss.sum()
        
        
class NLLLaplace:
    """ Computes Negative Log Likelihood loss for a (single) Laplace distribution. """
    def __init__(self, sum_normalized=True, ratio=1.0):
        """
        Args:
            sum_normalized: bool, compute the sum over tensor and divide by number of image pairs per batch ?
            ratio:
        """
        super().__init__()
        self.sum_normalized = sum_normalized
        self.ratio = ratio

    def __call__(self, gt_flow, est_flow, log_var, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            log_var: estimated log variance, shape (b, 1, H, W)
            mask: valid mask, where the loss is computed. shape (b, 1, H, W)
        """
        b, _, h, w = gt_flow.shape
        loss1 = math.sqrt(2) * torch.exp(-0.5 * log_var) * torch.abs(gt_flow - est_flow)
        # each dimension is multiplied
        loss2 = 0.5 * log_var
        loss = loss1 + loss2

        if mask is not None:
            mask = ~torch.isnan(loss.detach()) & ~torch.isinf(loss.detach()) & mask
        else:
            mask = ~torch.isnan(loss.detach()) & ~torch.isinf(loss.detach())

        if torch.isnan(loss.detach()).sum().ge(1) or torch.isinf(loss.detach()).sum().ge(1):
            print('mask or inf in the loss ! ')

        if mask is not None:
            loss = loss * mask.float()
            L = 0
            for bb in range(0, b):
                loss[bb, ...][mask[bb, ...] == 0] = loss[bb, ...][mask[bb, ...] == 0].detach()
                norm_const = float(h) * float(w) / (mask[bb, ...].sum().float() + 1e-6)
                L = L + loss[bb][mask[bb, ...] != 0].sum() * norm_const
            if self.sum_normalized:
                return L / b
            else:
                return L

        if self.sum_normalized:
            return loss.sum() / b
        else:
            return loss


class NLLGaussian:
    """ Computes Negative Log Likelihood loss for a (single) Gaussian distribution. """
    def __init__(self, sum_normalized=True, ratio=1.0):
        """
        Args:
            sum_normalized: bool, compute the sum over tensor and divide by number of image pairs per batch ?
            ratio:
        """
        super().__init__()
        self.sum_normalized = sum_normalized
        self.ratio = ratio

    def __call__(self, gt_flow, est_flow, log_var, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            log_var: estimated log variance, shape (b, 1, H, W)
            mask: valid mask, where the loss is computed. shape (b, 1, H, W)
        """
        b, _, h, w = gt_flow.shape
        loss1 = torch.mul(torch.exp(-log_var), (gt_flow - est_flow) ** 2)
        loss2 = log_var
        loss = .5 * (loss1 + loss2)

        if mask is not None:
            loss = loss * mask.float()
            L = 0
            for bb in range(0, b):
                loss[bb, ...][mask[bb, ...] == 0] = loss[bb, ...][mask[bb, ...] == 0].detach()
                norm_const = float(h*w) / (mask[bb, ...].sum().float() + 1e-6)
                L = L + loss[bb][mask[bb, ...] != 0].sum() * norm_const
            if self.sum_normalized:
                return L / b
            else:
                return L

        if self.sum_normalized:
            return loss.sum() / b
        else:
            return loss.sum()


