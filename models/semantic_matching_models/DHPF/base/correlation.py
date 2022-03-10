r"""Provides functions that creates/manipulates correlation matrices"""
import torch.nn.functional as F
import torch


class Correlation:
    @classmethod
    def bmm_interp(cls, src_feat, trg_feat, interp_size):
        r"""Performs batch-wise matrix-multiplication after interpolation"""
        src_feat = F.interpolate(src_feat, interp_size, mode='bilinear', align_corners=True)
        trg_feat = F.interpolate(trg_feat, interp_size, mode='bilinear', align_corners=True)

        src_feat = src_feat.view(src_feat.size(0), src_feat.size(1), -1).transpose(1, 2)
        trg_feat = trg_feat.view(trg_feat.size(0), trg_feat.size(1), -1)

        return torch.bmm(src_feat, trg_feat)  # shape is b, hsws, htwt

    @classmethod
    def mutual_nn_filter(cls, correlation_matrix):
        r"""Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18)"""
        corr_src_max = torch.max(correlation_matrix, dim=2, keepdim=True)[0]
        corr_trg_max = torch.max(correlation_matrix, dim=1, keepdim=True)[0]
        corr_src_max[corr_src_max == 0] += 1e-30
        corr_trg_max[corr_trg_max == 0] += 1e-30

        corr_src = correlation_matrix / corr_src_max
        corr_trg = correlation_matrix / corr_trg_max

        return correlation_matrix * (corr_src * corr_trg)
