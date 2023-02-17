# extracted and modified from CATs
from operator import add
from functools import reduce, partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, trunc_normal_


from models.semantic_matching_models import resnet
from admin.model_constructor import model_constructor
from utils_flow.correlation_to_matches_utils import correlation_to_flow_w_argmax
from utils_flow.pixel_wise_mapping import warp
r'''
Modified timm library Vision Transformer implementation
https://github.com/rwightman/pytorch-image-models
'''


def unnormalise_and_convert_mapping_to_flow(map):
    # here map is normalised to -1;1
    # we put it back to 0,W-1, then convert it to flow
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    # mesh grid
    mapping[:,0,:,:] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0 # unormalise
    mapping[:,1,:,:] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0 # unormalise

    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if mapping.is_cuda:
        grid = grid.cuda()
    flow = mapping - grid
    return flow


class FeatureL2Norm(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature, dim=1):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), dim) + epsilon, 0.5).unsqueeze(dim).expand_as(feature)
        return torch.div(feature, norm)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiscaleBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_multiscale = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        '''
        Multi-level aggregation
        '''
        B, N, H, W = x.shape
        if N == 1:
            x = x.flatten(0, 1)
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x.view(B, N, H, W)
        x = x.flatten(0, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp2(self.norm4(x)))
        x = x.view(B, N, H, W).transpose(1, 2).flatten(0, 1) 
        x = x + self.drop_path(self.attn_multiscale(self.norm3(x)))
        x = x.view(B, H, N, W).transpose(1, 2).flatten(0, 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.view(B, N, H, W)
        return x


class TransformerAggregator(nn.Module):
    def __init__(self, num_hyperpixel, img_size=224, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.pos_embed_x = nn.Parameter(torch.zeros(1, num_hyperpixel, 1, img_size, embed_dim // 2))
        self.pos_embed_y = nn.Parameter(torch.zeros(1, num_hyperpixel, img_size, 1, embed_dim // 2))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            MultiscaleBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.proj = nn.Linear(embed_dim, img_size ** 2)
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed_x, std=.02)
        trunc_normal_(self.pos_embed_y, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, corr, source, target):
        B = corr.shape[0]
        x = corr.clone()
        
        pos_embed = torch.cat((self.pos_embed_x.repeat(1, 1, self.img_size, 1, 1), self.pos_embed_y.repeat(1, 1, 1, self.img_size, 1)), dim=4)
        pos_embed = pos_embed.flatten(2, 3)

        x = torch.cat((x.transpose(-1, -2), target), dim=3) + pos_embed
        x = self.proj(self.blocks(x)).transpose(-1, -2) + corr  # swapping the axis for swapping self-attention.

        x = torch.cat((x, source), dim=3) + pos_embed
        x = self.proj(self.blocks(x)) + corr 

        return x.mean(1)


class FeatureExtractionHyperPixel(nn.Module):
    def __init__(self, hyperpixel_ids, feature_size, freeze=True):
        super().__init__()
        self.backbone = resnet.resnet101(pretrained=True)
        self.feature_size = feature_size
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        nbottlenecks = [3, 4, 23, 3]
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.hyperpixel_ids = hyperpixel_ids

    def forward(self, img):
        r"""Extract desired a list of intermediate features"""

        feats = []

        # Layer 0
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)
        if 0 in self.hyperpixel_ids:
            feats.append(feat.clone())

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(self.bottleneck_ids, self.layer_ids)):
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

            if hid + 1 in self.hyperpixel_ids:
                feats.append(feat.clone())
                #if hid + 1 == max(self.hyperpixel_ids):
                #    break
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        # Up-sample & concatenate features to construct a hyperimage
        for idx, feat in enumerate(feats):
            feats[idx] = F.interpolate(feat, self.feature_size, None, 'bilinear', True)

        return feats


class CATs(nn.Module):
    def __init__(self,
                 feature_size=16,
                 feature_proj_dim=128,
                 depth=1,
                 num_heads=6,
                 mlp_ratio=4,
                 hyperpixel_ids=[2,17,21,22,25,26,28],  # [0,8,20,21,26,28,29,30] for spair
                 freeze=True, forward_pass_strategy=None, inference_strategy='softargmax'):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.inference_strategy = inference_strategy
        self.forward_pass_strategy = forward_pass_strategy
        self.feature_size = feature_size
        self.feature_proj_dim = feature_proj_dim
        self.decoder_embed_dim = self.feature_size ** 2 + self.feature_proj_dim
        
        channels = [64] + [256] * 3 + [512] * 4 + [1024] * 23 + [2048] * 3

        self.feature_extraction = FeatureExtractionHyperPixel(hyperpixel_ids, feature_size, freeze)
        self.proj = nn.ModuleList([
            nn.Linear(channels[i], self.feature_proj_dim) for i in hyperpixel_ids
        ])

        self.decoder = TransformerAggregator(
            img_size=self.feature_size, embed_dim=self.decoder_embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_hyperpixel=len(hyperpixel_ids))
            
        self.l2norm = FeatureL2Norm()

        # 1-d indices for generating Gaussian kernels
        self.x = np.linspace(0, self.feature_size - 1, self.feature_size)
        self.x = torch.tensor(self.x, dtype=torch.float, requires_grad=False).to(self.device)
        self.y = np.linspace(0, self.feature_size - 1, self.feature_size)
        self.y = torch.tensor(self.y, dtype=torch.float, requires_grad=False).to(self.device)

        self.x_normal = np.linspace(-1,1,self.feature_size)
        self.x_normal = nn.Parameter(torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False))
        self.y_normal = np.linspace(-1,1,self.feature_size)
        self.y_normal = nn.Parameter(torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def softmax_with_temperature(self, x, beta, d = 1):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M # subtract maximum value for stability
        exp_x = torch.exp(x/beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    def soft_argmax(self, corr, beta=0.02, apply_softmax=True):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        b,_,h,w = corr.size()

        if apply_softmax:
            corr = self.softmax_with_temperature(corr, beta=beta, d=1)
        else:
            corr = corr / (corr.sum(dim=1, keepdim=True) + 1e-8)
        corr = corr.view(-1, h, w, h, w) # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
        x_normal = self.x_normal.expand(b,w)
        x_normal = x_normal.view(b,w,1,1)
        grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        
        grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
        y_normal = self.y_normal.expand(b,h)
        y_normal = y_normal.view(b,h,1,1)
        grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        return grid_x, grid_y

    def apply_gaussian_kernel(self, corr, sigma=5):
        h, w = corr.shape[-2:]
        b = corr.shape[0]
        corr = corr.view(b, -1, h, w)
        hw = corr.shape[1]

        idx = corr.max(dim=1)[1]  # b x h x w    get maximum value along channel
        idx_y = (idx // w).view(b, 1, 1, h, w).float()
        idx_x = (idx % w).view(b, 1, 1, h, w).float()

        x = self.x.view(1, 1, w, 1, 1).expand(b, 1, w, h, w)
        y = self.y.view(1, h, 1, 1, 1).expand(b, h, 1, h, w)

        gauss_kernel = torch.exp(-((x - idx_x) ** 2 + (y - idx_y) ** 2) / (2 * sigma ** 2))
        gauss_kernel = gauss_kernel.view(b, hw, h, w)

        return gauss_kernel * corr

    def mutual_nn_filter(self, correlation_matrix):
        r"""Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18)"""
        corr_src_max = torch.max(correlation_matrix, dim=3, keepdim=True)[0]
        corr_trg_max = torch.max(correlation_matrix, dim=2, keepdim=True)[0]
        corr_src_max[corr_src_max == 0] += 1e-30
        corr_trg_max[corr_trg_max == 0] += 1e-30

        corr_src = correlation_matrix / corr_src_max
        corr_trg = correlation_matrix / corr_trg_max

        return correlation_matrix * (corr_src * corr_trg)
    
    def corr(self, src, trg):
        return src.flatten(2).transpose(-1, -2) @ trg.flatten(2)

    def forward(self, im_source, im_target, mode='flow_prediction', *args, **kwargs):
        B, _, H, W = im_target.size()

        src_feats = self.feature_extraction(im_source)
        tgt_feats = self.feature_extraction(im_target)

        corrs = []
        src_feats_proj = []
        tgt_feats_proj = []
        for i, (src, tgt) in enumerate(zip(src_feats, tgt_feats)):
            corr = self.corr(self.l2norm(src), self.l2norm(tgt))
            corrs.append(corr)
            src_feats_proj.append(self.proj[i](src.flatten(2).transpose(-1, -2)))
            tgt_feats_proj.append(self.proj[i](tgt.flatten(2).transpose(-1, -2)))

        src_feats = torch.stack(src_feats_proj, dim=1)
        tgt_feats = torch.stack(tgt_feats_proj, dim=1)
        corr = torch.stack(corrs, dim=1)
        
        corr = self.mutual_nn_filter(corr)

        refined_corr = self.decoder(corr, src_feats, tgt_feats)

        if mode == 'flow_prediction' or self.forward_pass_strategy == 'flow_prediction':
            output_shape = im_target.shape[-2:]
            grid_x_t_to_s, grid_y_t_to_s = self.soft_argmax(refined_corr.view(B, -1, self.feature_size, self.feature_size))

            flow_t_to_s = torch.cat((grid_x_t_to_s, grid_y_t_to_s), dim=1)
            flow_t_to_s = unnormalise_and_convert_mapping_to_flow(flow_t_to_s)
            h, w = flow_t_to_s.shape[-2:]
            flow_est_t_to_s = F.interpolate(flow_t_to_s, size=output_shape, mode='bilinear', align_corners=False)
            flow_est_t_to_s[:, 0] *= float(output_shape[1]) / float(w)
            flow_est_t_to_s[:, 1] *= float(output_shape[0]) / float(h)

            grid_x_s_to_t, grid_y_s_to_t = self.soft_argmax(refined_corr.view(B, -1, self.feature_size, self.feature_size)
                        .permute(0, 2, 3, 1).reshape(B, -1, self.feature_size, self.feature_size))

            flow_s_to_t = torch.cat((grid_x_s_to_t, grid_y_s_to_t), dim=1)
            flow_s_to_t = unnormalise_and_convert_mapping_to_flow(flow_s_to_t)
            h, w = flow_s_to_t.shape[-2:]
            flow_est_s_to_t = F.interpolate(flow_s_to_t, size=output_shape, mode='bilinear', align_corners=False)
            flow_est_s_to_t[:, 0] *= float(output_shape[1]) / float(w)
            flow_est_s_to_t[:, 1] *= float(output_shape[0]) / float(h)
            return {'flow_estimates': [flow_est_t_to_s],
                    'flow_from_t_to_s': flow_est_t_to_s,
                    'flow_from_s_to_t': flow_est_s_to_t,
                    'correlation_from_t_to_s': refined_corr.view(B, -1, self.feature_size, self.feature_size),
                    'target_feat': tgt_feats
                    }

        elif mode == 'train_corr':
            return {'correlation_from_t_to_s': refined_corr.view(B, -1, self.feature_size, self.feature_size),
                    'target_feat': tgt_feats, 'target_shape': im_target.shape[-2:]}

        grid_x_t_to_s, grid_y_t_to_s = self.soft_argmax(refined_corr.view(B, -1, self.feature_size, self.feature_size))
        flow_t_to_s = torch.cat((grid_x_t_to_s, grid_y_t_to_s), dim=1)
        flow_t_to_s = unnormalise_and_convert_mapping_to_flow(flow_t_to_s)
        return flow_t_to_s

    def pre_process_data(self, source_img, target_img):
        # img has shape bx3xhxw
        # computes everything at 240 x 240
        device = self.device
        b, _, h_scale, w_scale = target_img.shape

        h_preprocessed = 256
        w_preprocessed = 256

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

        ratio_x = float(w_scale) / float(w_preprocessed)
        ratio_y = float(h_scale) / float(h_preprocessed)
        return source_img.to(self.device), target_img.to(self.device), ratio_x, ratio_y

    def estimate_flow(self, source_img, target_img, output_shape=None, scaling=1.0, mode='channel_first',
                      *args, **kwargs):
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

        if output_shape is None and scaling != 1.0:
            output_shape = (int(h_scale * scaling), int(w_scale * scaling))
        elif output_shape is None:
            output_shape = (h_scale, w_scale)

        source_img, target_img, ratio_x, ratio_y = self.pre_process_data(source_img, target_img)

        output = self.forward(im_source=source_img, im_target=target_img, mode='train_corr')
        correlation_from_t_to_s = output['correlation_from_t_to_s']
        h, w = correlation_from_t_to_s.shape[-2:]
        correlation_from_t_to_s = correlation_from_t_to_s.view(b, -1, h, w)

        if self.inference_strategy == 'argmax':
            flow_est = correlation_to_flow_w_argmax(correlation_from_t_to_s, output_shape)
        else:
            # softargmax
            grid_x, grid_y = self.soft_argmax(correlation_from_t_to_s, apply_softmax=True)
            flow_est = torch.cat((grid_x, grid_y), dim=1)
            flow_est = unnormalise_and_convert_mapping_to_flow(flow_est)
            flow_est = F.interpolate(flow_est, size=output_shape, mode='bilinear', align_corners=False)
            flow_est[:, 0] *= output_shape[1] / w
            flow_est[:, 1] *= output_shape[0] / h

        if mode != 'channel_first':
            return flow_est.permute(0, 2, 3, 1)
        return flow_est

    def estimate_flow_and_confidence_map(self, source_img, target_img, output_shape=None,
                                         scaling=1.0, mode='channel_first'):
        """
        Returns the flow field and corresponding confidence map/uncertainty map relating the target to the source image.
        Here, the confidence map corresponds to the inverse of the forward-backward cycle consistency error map.
        Returned flow has output_shape if provided, otherwise the same dimension than the target image.
        If scaling is provided, the output shape is the target image dimension multiplied by this scaling factor.
        Args:
            source_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            target_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            output_shape: int or list of int, or None, output shape of the returned flow field
            scaling: float, scaling factor applied to target image shape, to obtain the outputted flow field dimensions
                     if output_shape is None
            mode: if channel_first, flow has shape b, 2, H, W. Else shape is b, H, W, 2

        Returns:
            flow_est: estimated flow field relating the target to the reference image, resized and scaled to
                      output_shape (can be defined by scaling parameter)
            uncertainty_est: dict with keys 'cyclic_consistency_error'
        """
        flow_est, correlation_from_t_to_s = self.estimate_flow(source_img, target_img, output_shape=output_shape,
                                                               scaling=scaling, mode=mode, return_corr=True)

        b, c, h, w = correlation_from_t_to_s.shape
        # correlation_from_t_to_s  shape b, h_sw_s, h_t, w_t
        max_score, idx_B_Avec = torch.max(correlation_from_t_to_s, dim=1)
        max_score = max_score.view(b, h, w)
        max_score = F.interpolate(max_score.unsqueeze(1), flow_est.shape[-2:], mode='bilinear', align_corners=False)

        uncertain_score = 1.0 / (max_score + 1e-8)

        flow_est_backward, correlation_from_s_to_t = self.estimate_flow(target_img, source_img, output_shape=output_shape,
                                                                        scaling=scaling, mode=mode, return_corr=True)
        cyclic_consistency_error = torch.norm(flow_est + warp(flow_est_backward, flow_est), dim=1, p=2,
                                              keepdim=True)
        uncertainty_est = {'cyclic_consistency_error': cyclic_consistency_error,
                           'corr_score_uncertainty': uncertain_score}
        if mode == 'channel_first':
            return flow_est, uncertainty_est
        else:
            return flow_est.permute(0, 2, 3, 1), uncertainty_est


@model_constructor
def cats(feature_size=16, feature_proj_dim=128, depth=1, num_heads=6, mlp_ratio=4,
         hyperpixel_ids=[2,17,21,22,25,26,28],  # [0,8,20,21,26,28,29,30] for spair
         freeze=True, forward_pass_strategy=None):
    return CATs(feature_size, feature_proj_dim, depth, num_heads, mlp_ratio, hyperpixel_ids, freeze,
                forward_pass_strategy=forward_pass_strategy)

