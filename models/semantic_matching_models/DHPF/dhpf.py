"""Implementation of Dynamic Hyperpixel Flow"
Extracted and modified from DHPF"""

from functools import reduce
from operator import add

import torch.nn as nn
import torch

from .base.correlation import Correlation
from .base.geometry import Geometry
from .base.norm import Norm
from .base import resnet
from . import gating
from . import rhm
import numpy as np
import math
from utils_flow.correlation_to_matches_utils import correlation_to_flow_w_argmax, correlation_to_flow_w_soft_argmax
import torch.nn.functional as F


class DynamicHPF(nn.Module):
    r"""Dynamic Hyperpixel Flow (DHPF)"""
    def __init__(self, backbone='resnet101', img_side=240):
        r"""Constructor for DHPF"""
        super(DynamicHPF, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        # 1. Backbone network initialization
        if backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True).to(device)
            self.in_channels = [64, 256, 256, 256, 512, 512, 512, 512, 1024,
                                1024, 1024, 1024, 1024, 1024, 2048, 2048, 2048]
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True).to(device)
            self.in_channels = [64, 256, 256, 256, 512, 512, 512, 512,
                                1024, 1024, 1024, 1024, 1024, 1024, 1024,
                                1024, 1024, 1024, 1024, 1024, 1024, 1024,
                                1024, 1024, 1024, 1024, 1024, 1024, 1024,
                                1024, 1024, 2048, 2048, 2048]
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.backbone.eval()

        # 2. Dynamic layer gatings initialization
        self.learner = gating.GumbelFeatureSelection(self.in_channels).to(device)

        # 3. Miscellaneous
        self.relu = nn.ReLU()
        self.upsample_size = [int(img_side / 4)] * 2
        Geometry.initialize(self.upsample_size, device)
        self.rhm = rhm.HoughMatching(Geometry.rfs, torch.tensor([img_side, img_side]).to(device))

    def set_epoch(self, epoch):
        self.epoch = epoch

    # Forward pass
    def forward(self, im_source, im_target, **kwargs):
        # 1. Compute correlations between hyperimages
        src_img = im_source
        trg_img = im_target
        correlation_matrix, layer_sel = self.hyperimage_correlation(src_img, trg_img)

        # 2. Compute geometric matching scores to re-weight appearance matching scores (RHM)
        with torch.no_grad():  # no back-prop thru rhm due to memory issue
            geometric_scores = torch.stack([self.rhm.run(c.clone().detach()) for c in correlation_matrix], dim=0)
        correlation_matrix *= geometric_scores

        return correlation_matrix, layer_sel

    def hyperimage_correlation(self, src_img, trg_img):
        r"""Dynamically construct hyperimages and compute their correlations"""
        layer_sel = []
        correlation, src_norm, trg_norm = 0, 0, 0

        # Concatenate source & target images (B,6,H,W)
        # Perform group convolution (group=2) for faster inference time
        pair_img = torch.cat([src_img, trg_img], dim=1)

        # Layer 0
        with torch.no_grad():
            feat = self.backbone.conv1.forward(pair_img)
            feat = self.backbone.bn1.forward(feat)
            feat = self.backbone.relu.forward(feat)
            feat = self.backbone.maxpool.forward(feat)

            src_feat = feat.narrow(1, 0, feat.size(1) // 2).clone()
            trg_feat = feat.narrow(1, feat.size(1) // 2, feat.size(1) // 2).clone()

        # Save base maps
        base_src_feat = self.learner.reduction_ffns[0](src_feat)
        base_trg_feat = self.learner.reduction_ffns[0](trg_feat)
        base_correlation = Correlation.bmm_interp(base_src_feat, base_trg_feat, self.upsample_size)
        base_src_norm = Norm.feat_normalize(base_src_feat, self.upsample_size)
        base_trg_norm = Norm.feat_normalize(base_trg_feat, self.upsample_size)

        src_feat, trg_feat, lsel = self.learner(0, src_feat, trg_feat)
        if src_feat is not None and trg_feat is not None:
            correlation += Correlation.bmm_interp(src_feat, trg_feat, self.upsample_size)
            src_norm += Norm.feat_normalize(src_feat, self.upsample_size)
            trg_norm += Norm.feat_normalize(trg_feat, self.upsample_size)
        layer_sel.append(lsel)

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(self.bottleneck_ids, self.layer_ids)):
            with torch.no_grad():
                res = feat
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)
                if bid == 0:
                    res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)
                feat += res

                src_feat = feat.narrow(1, 0, feat.size(1) // 2).clone()
                trg_feat = feat.narrow(1, feat.size(1) // 2, feat.size(1) // 2).clone()

            src_feat, trg_feat, lsel = self.learner(hid + 1, src_feat, trg_feat)
            if src_feat is not None and trg_feat is not None:
                correlation += Correlation.bmm_interp(src_feat, trg_feat, self.upsample_size)
                src_norm += Norm.feat_normalize(src_feat, self.upsample_size)
                trg_norm += Norm.feat_normalize(trg_feat, self.upsample_size)
            layer_sel.append(lsel)

            with torch.no_grad():
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        layer_sel = torch.stack(layer_sel).t()

        # If no layers are selected, select the base map
        if (layer_sel.sum(dim=1) == 0).sum() > 0:
            empty_sel = (layer_sel.sum(dim=1) == 0).nonzero().view(-1).long()
            if src_img.size(0) == 1:
                correlation = base_correlation
                src_norm = base_src_norm
                trg_norm = base_trg_norm
            else:
                correlation[empty_sel] += base_correlation[empty_sel]
                src_norm[empty_sel] += base_src_norm[empty_sel]
                trg_norm[empty_sel] += base_trg_norm[empty_sel]

        if self.learner.training:
            src_norm[src_norm == 0.0] += 0.0001
            trg_norm[trg_norm == 0.0] += 0.0001
        src_norm = src_norm.pow(0.5).unsqueeze(2)
        trg_norm = trg_norm.pow(0.5).unsqueeze(1)

        # Appearance matching confidence (p(m_a)): cosine similarity between hyperpimages
        correlation_ts = self.relu(correlation / (torch.bmm(src_norm, trg_norm) + 0.001)).pow(2)

        return correlation_ts, layer_sel

    def children(self):
        return self.learner.children()

    def modules(self):
        return self.learner.modules()

    def named_buffers(self, prefix='', recurse=True):
        return self.learner.named_buffers(prefix, recurse)

    def named_children(self):
        return self.learner.named_children()

    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        return self.learner.named_modules(memo, prefix)

    def named_parameters(self, prefix='', recurse=True):
        return self.learner.named_parameters(prefix, recurse)

    def parameters(self, recurse=True):
        return self.learner.parameters(recurse)

    def buffers(self, recurse=True):
        return self.learner.buffers(recurse)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.learner.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        self.learner.load_state_dict(state_dict, strict)

    def eval(self):
        self.learner.eval()

    def train(self, bool_=True):
        self.learner.train(bool_)
        self.backbone.eval()

    def to(self, device):
        self.learner.to(device)

    def pre_process_data(self, source_img, target_img):
        # img has shape bx3xhxw
        # computes everything at 240 x 240
        device = self.device
        b, _, h_scale, w_scale = target_img.shape

        h_preprocessed = 240
        w_preprocessed = 240

        source_img = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                     size=(h_preprocessed, w_preprocessed),
                                                     mode='area')
        target_img = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                     size=(h_preprocessed, w_preprocessed),
                                                     mode='area')
        mean_vector = np.array([0.485, 0.456, 0.406])
        std_vector = np.array([0.229, 0.224, 0.225])
        source_img = source_img.float().div(255.0)
        target_img = target_img.float().div(255.0)
        mean = torch.as_tensor(mean_vector, dtype=source_img.dtype, device=source_img.device)
        std = torch.as_tensor(std_vector, dtype=source_img.dtype, device=source_img.device)
        source_img.sub_(mean[:, None, None]).div_(std[:, None, None])
        target_img.sub_(mean[:, None, None]).div_(std[:, None, None])

        ratio_x = float(w_scale)/float(w_preprocessed)
        ratio_y = float(h_scale)/float(h_preprocessed)
        return source_img.to(self.device), target_img.to(self.device), ratio_x, ratio_y

    def corr_to_flow_strategy(self,  correlation_matrix_s_to_t, ratio_x, ratio_y, h_scale, w_scale, output_shape=None,
                              batch=None, inference_strategy='original', return_corr=False):

        if inference_strategy is None or inference_strategy == 'None':
            inference_strategy = 'original'

        b = correlation_matrix_s_to_t.shape[0]
        h = int(math.sqrt(correlation_matrix_s_to_t.shape[-1]))

        if inference_strategy == 'original':
            # but need to be like this for transfer keypoint that they use

            # the scaling to 4x is done is there too.
            # 2. Transfer key-points (nearest neighbor assignment)
            if 'target_kps' in batch.keys():
                target_coor_resized = batch['target_kps'].clone()[0, :batch['n_pts'][0]]
                len_coor = target_coor_resized.shape[0]

                target_coor_resized[:, 0] /= ratio_x
                target_coor_resized[:, 1] /= ratio_y
                prd_kps = Geometry.transfer_kps(correlation_matrix_s_to_t,
                                                [torch.t(target_coor_resized).to(self.device)], [len_coor])
                # always multiplied by 4

                flow_est = torch.zeros(h_scale, w_scale, 2).float().to(self.device)
                # computes the flow
                point_target_coords = batch['target_kps'][0, :batch['n_pts'][0]].clone().to(self.device)
                predicted_source_coords = torch.t(prd_kps[0])[:len_coor]

                predicted_source_coords[:, 0] *= ratio_x
                predicted_source_coords[:, 1] *= ratio_y

                point_target_coords[:, 0] = torch.clamp(point_target_coords[:, 0], 0, w_scale - 1)
                point_target_coords[:, 1] = torch.clamp(point_target_coords[:, 1], 0, h_scale - 1)

                flow_est[torch.round(point_target_coords[:, 1]).long(), torch.round(point_target_coords[:, 0]).long()] = \
                    predicted_source_coords - point_target_coords
                flow_est = flow_est.unsqueeze(0).permute(0, 3, 1, 2)
            else:
                h_tgt, w_tgt = 240, 240
                X, Y = np.meshgrid(np.linspace(0, w_tgt - 1, w_tgt),
                                   np.linspace(0, h_tgt - 1, h_tgt))

                grid_X_vec = torch.from_numpy(X).view(-1, 1).float()
                grid_Y_vec = torch.from_numpy(Y).view(-1, 1).float()

                target_coor_resized = torch.cat((grid_X_vec, grid_Y_vec), 1).to(self.device)  # N, 2

                prd_kps = Geometry.transfer_kps(correlation_matrix_s_to_t,
                                                [torch.t(target_coor_resized).to(self.device)], [len(target_coor_resized)])

                predicted_source_coords = torch.t(prd_kps[0]).view(h_tgt, w_tgt, -1)
                flow_est = predicted_source_coords - target_coor_resized.view(h_tgt, w_tgt, -1)
                flow_est = flow_est.unsqueeze(0).permute(0, 3, 1, 2)

                flow_est = F.interpolate(flow_est, (h_scale, w_scale), mode='bilinear', align_corners=False)

                flow_est[:, 0] *= ratio_x
                flow_est[:, 1] *= ratio_y

        elif inference_strategy == 'argmax':
            # shape of this is b, h_t*w_t, h_s*w_s before. after permute it is b, h_s*w_s, h_t*w_t
            correlation_matrix_t_to_s = Correlation.mutual_nn_filter(correlation_matrix_s_to_t.permute(0, 2, 1))
            # correct shape now, b, h_s*w_s, h_t*w_t
            flow_est = correlation_to_flow_w_argmax(correlation_matrix_t_to_s.view(b, -1, h, h),
                                                    output_shape=output_shape,
                                                    return_mapping=False)
        else:
            # softargmax
            # shape of this is b, h_t*w_t, h_s*w_s before. after permute it is b, h_s*w_s, h_t*w_t
            correlation_matrix_t_to_s = Correlation.mutual_nn_filter(correlation_matrix_s_to_t.permute(0, 2, 1))
            # correct shape now, b, h_s*w_s, h_t*w_t
            flow_est = correlation_to_flow_w_soft_argmax(correlation_matrix_t_to_s.view(b, -1, h, h),
                                                         output_shape=output_shape, apply_softmax=True)

        if return_corr:
            correlation_matrix_t_to_s = Correlation.mutual_nn_filter(correlation_matrix_s_to_t
                                                                     .permute(0, 2, 1)).view(b, -1, h, h)
            # correlation_matrix_t_to_s = correlation_matrix_s_to_t.permute(0, 2, 1).view(b, -1,  h, h)
            return flow_est, correlation_matrix_t_to_s
        else:
            return flow_est

    def estimate_flow(self, source_img, target_img, output_shape=None, scaling=1.0, mode='channel_first', batch=None,
                      inference_strategy='original', return_corr=False):
        """
        Estimates the flow field relating the target to the source image. Returned flow has output_shape if provided,
        otherwise the same dimension than the target image. If scaling is provided, the output shape is the
        target image dimension multiplied by this scaling factor.
        Args:
            source_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            target_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            output_shape: int or list of int, or None, output shape of the returned flow field
            scaling: float, scaling factor applied to target image shape, to obtain the outputted flow field dimensions
                     if output_shape is None
            mode: if channel_first, flow has shape b, 2, H, W. Else shape is b, H, W, 2

        Returns:
            flow_est: estimated flow field relating the target to the reference image,resized and scaled to output_shape
                      (can be defined by scaling parameter)
        """
        b, _, h_scale, w_scale = target_img.shape
        # define output_shape
        output_shape = (h_scale, w_scale)
        if output_shape is None and scaling != 1.0:
            output_shape = (int(h_scale * scaling), int(w_scale * scaling))

        source_img, target_img, ratio_x, ratio_y = self.pre_process_data(source_img, target_img)

        if output_shape is not None:
            ratio_x *= float(output_shape[1]) / float(w_scale)
            ratio_y *= float(output_shape[0]) / float(h_scale)
        else:
            output_shape = (h_scale, w_scale)

        # 1. DHPF forward pass
        correlation_matrix_s_to_t, layer_sel = self.forward(target_img, source_img)
        # shape of this is b, h_t*w_t, h_s*w_s

        return self.corr_to_flow_strategy(correlation_matrix_s_to_t, ratio_x, ratio_y, h_scale, w_scale,
                                          output_shape=output_shape, batch=batch, inference_strategy=inference_strategy,
                                          return_corr=return_corr)
