"""Provides functions that manipulate boxes and points"""
import torch

from .correlation import Correlation


class Geometry:
    @classmethod
    def initialize(cls, feat_size, device):
        cls.max_pts = 400
        cls.eps = 1e-30
        cls.rfs = cls.receptive_fields(11, 4, feat_size).to(device)
        cls.rf_center = Geometry.center(cls.rfs)

    @classmethod
    def center(cls, box):
        r"""Computes centers, (x, y), of box (N, 4)"""
        x_center = box[:, 0] + (box[:, 2] - box[:, 0]) // 2
        y_center = box[:, 1] + (box[:, 3] - box[:, 1]) // 2
        return torch.stack((x_center, y_center)).t().to(box.device)

    @classmethod
    def receptive_fields(cls, rfsz, jsz, feat_size):
        r"""Returns a set of receptive fields (N, 4)"""
        width = feat_size[1]
        height = feat_size[0]

        feat_ids = torch.tensor(list(range(width))).repeat(1, height).t().repeat(1, 2)
        feat_ids[:, 0] = torch.tensor(list(range(height))).unsqueeze(1).repeat(1, width).view(-1)

        box = torch.zeros(feat_ids.size()[0], 4)
        box[:, 0] = feat_ids[:, 1] * jsz - rfsz // 2
        box[:, 1] = feat_ids[:, 0] * jsz - rfsz // 2
        box[:, 2] = feat_ids[:, 1] * jsz + rfsz // 2
        box[:, 3] = feat_ids[:, 0] * jsz + rfsz // 2

        return box

    @classmethod
    def gaussian2d(cls, side=7):
        r"""Returns 2-dimensional gaussian filter"""
        dim = [side, side]

        siz = torch.LongTensor(dim)
        sig_sq = (siz.float()/2/2.354).pow(2)
        siz2 = (siz-1)/2

        x_axis = torch.arange(-siz2[0], siz2[0] + 1).unsqueeze(0).expand(dim).float()
        y_axis = torch.arange(-siz2[1], siz2[1] + 1).unsqueeze(1).expand(dim).float()

        gaussian = torch.exp(-(x_axis.pow(2)/2/sig_sq[0] + y_axis.pow(2)/2/sig_sq[1]))
        gaussian = gaussian / gaussian.sum()

        return gaussian

    @classmethod
    def neighbours(cls, box, kps):
        r"""Returns boxes in one-hot format that covers given keypoints"""
        box_duplicate = box.unsqueeze(2).repeat(1, 1, len(kps.t())).transpose(0, 1)
        kps_duplicate = kps.unsqueeze(1).repeat(1, len(box), 1)

        xmin = kps_duplicate[0].ge(box_duplicate[0])
        ymin = kps_duplicate[1].ge(box_duplicate[1])
        xmax = kps_duplicate[0].le(box_duplicate[2])
        ymax = kps_duplicate[1].le(box_duplicate[3])

        nbr_onehot = torch.mul(torch.mul(xmin, ymin), torch.mul(xmax, ymax)).t()
        n_neighbours = nbr_onehot.sum(dim=1)

        return nbr_onehot, n_neighbours

    @classmethod
    def transfer_kps(cls, correlation_matrix, kps, n_pts, transpose_kp=False):
        r"""Transfer keypoints by nearest-neighbour assignment"""
        if len(correlation_matrix.shape) != 3:
            b, c, h, w = correlation_matrix.shape
            correlation_matrix = correlation_matrix.view(b, -1, h*w)
        correlation_matrix = Correlation.mutual_nn_filter(correlation_matrix)

        prd_kps = []
        for ct, kpss, np in zip(correlation_matrix, kps, n_pts):

            if transpose_kp:
                kpss = torch.t(kpss)
            # 1. Prepare geometries & argmax target indices
            kp = kpss.narrow_copy(1, 0, np)
            _, trg_argmax_idx = torch.max(ct, dim=1)
            geomet = cls.rfs[:, :2].unsqueeze(0).repeat(len(kp.t()), 1, 1)

            # 2. Retrieve neighbouring source boxes that cover source key-points
            src_nbr_onehot, n_neighbours = cls.neighbours(cls.rfs, kp)

            # 3. Get displacements from source neighbouring box centers to each key-point
            src_displacements = kp.t().unsqueeze(1).repeat(1, len(cls.rfs), 1) - geomet
            src_displacements = src_displacements * src_nbr_onehot.unsqueeze(2).repeat(1, 1, 2).float()

            # 4. Transfer the neighbours based on given correlation matrix
            vector_summator = torch.zeros_like(geomet)
            src_idx = src_nbr_onehot.nonzero()

            trg_idx = trg_argmax_idx.index_select(dim=0, index=src_idx[:, 1])
            vector_summator[src_idx[:, 0], src_idx[:, 1]] = geomet[src_idx[:, 0], trg_idx]
            vector_summator += src_displacements
            prd = (vector_summator.sum(dim=1) / n_neighbours.unsqueeze(1).repeat(1, 2).float()).t()

            # 5. Concatenate pad-points
            pads = (torch.zeros((2, cls.max_pts - np)).to(prd.device) - 1)
            prd = torch.cat([prd, pads], dim=1)
            prd_kps.append(prd)

        return torch.stack(prd_kps)
