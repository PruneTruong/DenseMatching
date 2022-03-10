"""Implementation of regularized Hough matching algorithm (RHM)"""
import math

import torch.nn.functional as F
import torch

from .base.geometry import Geometry


class HoughMatching:
    r"""Regularized Hough matching algorithm"""
    def __init__(self, rf, img_side, ncells=8192):
        r"""Constructor of HoughMatching"""
        super(HoughMatching, self).__init__()

        device = rf.device
        self.nbins_x, self.nbins_y, hs_cellsize = self.build_hspace(img_side, img_side, ncells)
        self.bin_ids = self.compute_bin_id(img_side, rf, rf, hs_cellsize, self.nbins_x)
        self.hspace = rf.new_zeros((len(rf), self.nbins_y * self.nbins_x))
        self.hbin_ids = self.bin_ids.add(torch.arange(0, len(rf)).to(device).
                                         mul(self.hspace.size(1)).unsqueeze(1).expand_as(self.bin_ids))
        self.hsfilter = Geometry.gaussian2d(7).to(device)

    def run(self, votes):
        r"""Regularized Hough matching"""
        hspace = self.hspace.view(-1).index_add(0, self.hbin_ids.view(-1), votes.view(-1)).view_as(self.hspace)
        hspace = torch.sum(hspace, dim=0)
        hspace = F.conv2d(hspace.view(1, 1, self.nbins_y, self.nbins_x),
                          self.hsfilter.unsqueeze(0).unsqueeze(0), padding=3).view(-1)

        return torch.index_select(hspace, dim=0, index=self.bin_ids.view(-1)).view_as(votes)

    def compute_bin_id(self, src_imsize, src_box, trg_box, hs_cellsize, nbins_x):
        r"""Computes Hough space bin ids for voting"""
        src_ptref = src_imsize.float()
        src_trans = Geometry.center(src_box)
        trg_trans = Geometry.center(trg_box)
        xy_vote = (src_ptref.unsqueeze(0).expand_as(src_trans) - src_trans).unsqueeze(2).\
                      repeat(1, 1, len(trg_box)) + trg_trans.t().unsqueeze(0).repeat(len(src_box), 1, 1)

        bin_ids = (xy_vote / hs_cellsize).long()

        return bin_ids[:, 0, :] + bin_ids[:, 1, :] * nbins_x

    def build_hspace(self, src_imsize, trg_imsize, ncells):
        r"""Build Hough space"""
        hs_width = src_imsize[0] + trg_imsize[0]
        hs_height = src_imsize[1] + trg_imsize[1]
        hs_cellsize = math.sqrt((hs_width * hs_height) / ncells)
        nbins_x = int(hs_width / hs_cellsize) + 1
        nbins_y = int(hs_height / hs_cellsize) + 1

        return nbins_x, nbins_y, hs_cellsize
