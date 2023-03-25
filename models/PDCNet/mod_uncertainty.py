import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.modules.batch_norm import BatchNorm


def estimate_average_variance_of_mixture_density(weight_map, log_var_map):
    # Computes variance of the mixture
    proba_map = torch.nn.functional.softmax(weight_map, dim=1)

    avg_variance = torch.sum(proba_map * torch.exp(log_var_map), dim=1, keepdim=True) # shape is b,1,  h, w
    return avg_variance


def estimate_probability_of_confidence_interval_of_mixture_density(weight_map, log_var_map, R=1.0, gaussian=False):
    """Computes P_R of a mixture of probability distributions (with K components). See PDC-Net.
    Args:
        weight_map: weight maps of each component of the mixture. They are not softmaxed yet. (B, K, H, W)
        log_var_map: log variance corresponding to each component, (B, K, H, W)
        R: radius for the confidence interval
        gaussian: Mixture of Gaussian or Laplace densities?
    """
    # compute P_R of the mixture
    proba_map = torch.nn.functional.softmax(weight_map, dim=1)

    if gaussian:
        var_map = torch.exp(log_var_map)
        p_r = torch.sum(proba_map * (1 - torch.exp(-R ** 2 / (2 * var_map))), dim=1, keepdim=True)
    else:
        # laplace distribution
        var_map = torch.exp(log_var_map)
        p_r = torch.sum(proba_map * (1 - torch.exp(- math.sqrt(2)*R/torch.sqrt(var_map)))**2, dim=1, keepdim=True)
    return p_r


def estimate_probability_of_confidence_interval_of_unimodal_density(log_var_map, R=1.0, gaussian=False):
    """Computes P_R of a unimodal probability distribution.
    Args:
        log_var_map: log variance of the distribution, (B, 1, H, W)
        R: radius for the confidence interval
        gaussian: Mixture of Gaussian or Laplace densities?
    """
    # NOTE: ONLY FOR GAUSSIAN
    assert log_var_map.shape[1] == 1
    var_map = torch.exp(log_var_map)

    if gaussian:
        p_r = 1.0 - torch.exp(-R ** 2 / (2 * var_map))
    else:
        p_r = (1 - torch.exp(- math.sqrt(2)*R/torch.sqrt(var_map)))**2
    return p_r


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False):
    if batch_norm:
        return nn.Sequential(
                            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, bias=True),
                            BatchNorm(out_planes),
                            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(
                            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, bias=True),
                            nn.LeakyReLU(0.1))


class MixtureDensityEstimatorFromCorr(nn.Module):
    def __init__(self, in_channels, batch_norm, search_size, output_channels=3, estimate_small_variance=False,
                 concatenate_with_flow=False, nbr_channels_concatenated_flow=2, output_all_channels_together=False):
        super(MixtureDensityEstimatorFromCorr, self).__init__()
        # 9
        self.estimate_small_variance=estimate_small_variance
        self.concatenate_with_flow=concatenate_with_flow
        self.search_size = search_size
        self.output_all_channels_together = output_all_channels_together
        if self.search_size == 9:
            self.conv_0 = conv(in_channels, 32, kernel_size=3, stride=1, padding=0, batch_norm=batch_norm)
            self.conv_1 = conv(32, 32, kernel_size=3, stride=1, padding=0, batch_norm=batch_norm)
            self.conv_2 = conv(32, 16, kernel_size=3, stride=1, padding=0, batch_norm=batch_norm)
            if self.estimate_small_variance:
                self.predict_uncertainty = nn.Conv2d(16, output_channels + 1, kernel_size=3, stride=1, padding=0, bias=True)
            else:
                self.predict_uncertainty = nn.Conv2d(16, output_channels, kernel_size=3, stride=1, padding=0, bias=True)
        elif search_size == 16:
            self.conv_0 = conv(in_channels, 32, kernel_size=3, stride=1, padding=0, batch_norm=batch_norm)
            self.maxpool = nn.MaxPool2d((2, 2))
            self.conv_1 = conv(32, 32, kernel_size=3, stride=1, padding=0, batch_norm=batch_norm)
            self.conv_2 = conv(32, 16, kernel_size=3, stride=1, padding=0, batch_norm=batch_norm)
            if self.estimate_small_variance:
                self.predict_uncertainty = nn.Conv2d(16, output_channels+1, kernel_size=3, stride=1, padding=0, bias=True)
            else:
                self.predict_uncertainty = nn.Conv2d(16, output_channels, kernel_size=3, stride=1, padding=0, bias=True)

        if self.concatenate_with_flow:
            self.conv_3 = conv(4 + nbr_channels_concatenated_flow, 32, kernel_size=3, stride=1, padding=1, batch_norm=batch_norm)
            self.conv_4 = conv(32, 16, kernel_size=3, stride=1, padding=1, batch_norm=batch_norm)
            if self.estimate_small_variance:
                self.predict_uncertainty_final = nn.Conv2d(16, output_channels+1, kernel_size=3, stride=1, padding=1, bias=True)
            else:
                self.predict_uncertainty_final = nn.Conv2d(16, output_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, previous_uncertainty=None, flow=None, x_second_corr=None):
        # x shape is b, s*s, h, w
        b, _, h, w = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(b*h*w, self.search_size, self.search_size).unsqueeze(1).contiguous()
        # x is now shape b*h*w, 1, s, s

        if x_second_corr is not None:
            x_second_corr = x_second_corr.permute(0, 2, 3, 1).contiguous().view(b * h * w, self.search_size, self.search_size).unsqueeze(1).contiguous()
            # x_second_corr is now shape b*h*w, 1, s, s
            x = torch.cat((x, x_second_corr), 1)
            # x is now shape b*h*w, 2, s, s

        if previous_uncertainty is not None:
            # shape is b, 4, h, w
            previous_uncertainty = previous_uncertainty.permute(0, 2, 3, 1).contiguous().view(b * h * w, -1).unsqueeze(2).unsqueeze(2)
            previous_uncertainty = previous_uncertainty.repeat(1, 1, self.search_size, self.search_size)
            x = torch.cat((x, previous_uncertainty), 1)

        if self.search_size == 9:
            x = self.conv_2(self.conv_1(self.conv_0(x)))
            uncertainty_corr = self.predict_uncertainty(x)
        elif self.search_size == 16:
            x = self.conv_0(x)
            x = self.maxpool(x)
            x = self.conv_2(self.conv_1(x))
            uncertainty_corr = self.predict_uncertainty(x)

        if self.concatenate_with_flow:
            if self.estimate_small_variance:
                # shape is b*h*w, 4, 1, 1
                uncertainty_corr = uncertainty_corr.squeeze().view(b, h, w, 4).permute(0, 3, 1, 2)
            else:
                # shape is b*h*w, 3, 1, 1
                uncertainty_corr = uncertainty_corr.squeeze().view(b, h, w, 3).permute(0, 3, 1, 2)
                # shape is b, 3, h, w
                log_var_map = uncertainty_corr[:, 0].unsqueeze(1)
                proba_map = uncertainty_corr[:, 1:]
                uncertainty_corr = torch.cat((log_var_map, torch.zeros_like(log_var_map, requires_grad=False), proba_map), 1)
                # now shape is b, 4, h, w # based only on the correlation here

            uncertainty_and_flow = torch.cat((uncertainty_corr, flow), 1)
            x = self.conv_4(self.conv_3(uncertainty_and_flow))
            uncertainty = self.predict_uncertainty_final(x)
            if self.output_all_channels_together:
                return uncertainty
            else:
                if self.estimate_small_variance:
                    # shape is b*h*w, 4, 1, 1
                    large_log_var = uncertainty[:, 0].unsqueeze(1)
                    small_var = uncertainty[:, 1].unsqueeze(1)
                    small_log_var = F.logsigmoid(small_var) # constraining the small var to 0 and 1 and putting sigmoid to it
                    proba_map = uncertainty[:, 2:]
                    return large_log_var, small_log_var, proba_map
                else:
                    # shape is b, 3, h, w
                    log_var_map = uncertainty[:, 0].unsqueeze(1)
                    proba_map = uncertainty[:, 1:]
                    return log_var_map, proba_map
        else:
            uncertainty_corr = uncertainty_corr.squeeze().view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            if self.output_all_channels_together:
                return uncertainty_corr
            else:
                if self.estimate_small_variance:
                    # shape is b*h*w, 4, 1, 1

                    large_log_var = uncertainty_corr[:, 0].unsqueeze(1)
                    small_log_var = uncertainty_corr[:, 1].unsqueeze(1)
                    # small_log_var = F.logsigmoid(small_var)
                    # constraining the small var to 0 and 1 and putting sigmoid to it
                    proba_map = uncertainty_corr[:, 2:]
                    return large_log_var, small_log_var, proba_map
                else:
                    # shape is b*h*w, 3, 1, 1
                    # shape is b, 3, h, w
                    log_var_map = uncertainty_corr[:, 0].unsqueeze(1)
                    proba_map = uncertainty_corr[:, 1:]
                    return log_var_map, proba_map


class MixtureDensityEstimatorFromUncertaintiesAndFlow(nn.Module):
    def __init__(self, in_channels, batch_norm, output_channels=3, estimate_small_variance=False,
                 output_all_channels_together=False):
        super(MixtureDensityEstimatorFromUncertaintiesAndFlow, self).__init__()
        # 9
        self.output_channels = output_channels
        self.output_all_channels_together = output_all_channels_together
        self.estimate_small_variance = estimate_small_variance
        self.conv_0 = conv(in_channels, 32, kernel_size=3, stride=1, padding=1, batch_norm=batch_norm)
        self.conv_1 = conv(32, 16, kernel_size=3, stride=1, padding=1, batch_norm=batch_norm)
        if self.estimate_small_variance:
            self.predict_uncertainty_final = nn.Conv2d(16, output_channels+1, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            self.predict_uncertainty_final = nn.Conv2d(16, output_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.conv_1(self.conv_0(x))
        uncertainty = self.predict_uncertainty_final(x)
        if self.output_all_channels_together:
            return uncertainty
        else:
            if self.estimate_small_variance:
                # shape is b*h*w, 4, 1, 1
                large_log_var = uncertainty[:, 0].unsqueeze(1)
                small_var = uncertainty[:, 1].unsqueeze(1)
                small_log_var = F.logsigmoid(small_var)
                # constraining the small var to 0 and 1 and putting sigmoid to it
                if self.output_channels == 1:
                    proba_map = torch.ones_like(large_log_var)
                    # in case one only predicts the log variance (unimodel distribution)
                else:
                    proba_map = uncertainty[:, 2:]
                return large_log_var, small_log_var, proba_map
            else:
                # shape is b, 3, h, w
                log_var_map = uncertainty[:, 0].unsqueeze(1)  # always one that is not fixed
                if self.output_channels == 1:
                    proba_map = torch.ones_like(log_var_map)
                    # in case one only predicts the log variance (unimodel distribution)
                else:
                    proba_map = uncertainty[:, 1:]  # all the others are probability, sum should be 1 there
                return log_var_map, proba_map
