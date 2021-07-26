'''
extracted from https://github.com/XiSHEN0220/RANSAC-Flow/blob/master/model/ssimLoss.py
'''
import torch
import torch.nn.functional as F
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]).type(
        torch.cuda.FloatTensor)
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, mask, compute_average=False, sum_normalized=False):
    b, _, h, w = img2.shape
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    loss = 1 - ssim_map
    if compute_average:
        loss = torch.sum(loss * mask.float()) / (mask.sum().float() + 1e-7)
        return loss
    else:
        loss = loss * mask.float()
        L = 0
        for bb in range(0, b):
            norm_const = float(h * w) / (mask[bb, ...].sum().float() + 1e-6)
            L = L + loss[bb][mask[bb, ...].repeat(3, 1, 1) != 0].sum() * norm_const

        if sum_normalized:
            return L / b / 3
        else:
            return L / 3


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = 3
        self.window = create_window(window_size, self.channel)
        self.windowMask = torch.ones(1, 1, self.window_size,
                                     self.window_size).cuda() / self.window_size / self.window_size

    def forward(self, img1, img2, match):
        (_, channel, _, _) = img1.size()
        if len(match.shape) == 3:
            match = match.unsqueeze(1)
        mask = F.conv2d(match.float(), self.windowMask, padding=self.window_size // 2) + 1e-7
        mask = ((mask > 0.5).type(torch.cuda.FloatTensor) + 1e-7)

        ## maximize ssim
        return _ssim(img1, img2, self.window, self.window_size, self.channel, mask)