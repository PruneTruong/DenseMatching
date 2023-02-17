# extracted and modified from SF-Net (https://arxiv.org/abs/1904.01810)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import math

from admin.model_constructor import model_constructor
from models.non_matching_corr import LearntBinParam
from utils_flow.pixel_wise_mapping import warp
from utils_flow.correlation_to_matches_utils import correlation_to_flow_w_argmax, cost_volume_to_probabilistic_mapping
from utils_flow.flow_and_mapping_operations import unnormalize, convert_mapping_to_flow


class FeatureExtraction(nn.Module):
    def __init__(self, train_features=False):
        super(FeatureExtraction, self).__init__()
        model = models.resnet101(pretrained=True)
        resnet_feature_layers = ['conv1',
                                 'bn1',
                                 'relu',
                                 'maxpool',
                                 'layer1',
                                 'layer2',
                                 'layer3',
                                 'layer4',]
        layer1 = 'layer1'
        layer2 = 'layer2'
        layer3 = 'layer3'
        layer4 = 'layer4'
        layer1_idx = resnet_feature_layers.index(layer1)
        layer2_idx = resnet_feature_layers.index(layer2)
        layer3_idx = resnet_feature_layers.index(layer3)
        layer4_idx = resnet_feature_layers.index(layer4)
        resnet_module_list = [model.conv1,
                              model.bn1,
                              model.relu,
                              model.maxpool,
                              model.layer1,
                              model.layer2,
                              model.layer3,
                              model.layer4]
        self.layer1 = nn.Sequential(*resnet_module_list[:layer1_idx + 1])
        self.layer2 = nn.Sequential(*resnet_module_list[layer1_idx + 1:layer2_idx + 1])
        self.layer3 = nn.Sequential(*resnet_module_list[layer2_idx + 1:layer3_idx + 1])
        self.layer4 = nn.Sequential(*resnet_module_list[layer3_idx + 1:layer4_idx + 1])
        for param in self.layer1.parameters():
            param.requires_grad = train_features
        for param in self.layer2.parameters():
            param.requires_grad = train_features
        for param in self.layer3.parameters():
            param.requires_grad = train_features
        for param in self.layer4.parameters():
            param.requires_grad = train_features

    def forward(self, image_batch):
        layer1_feat = self.layer1(image_batch)
        layer2_feat = self.layer2(layer1_feat)
        layer3_feat = self.layer3(layer2_feat)
        layer4_feat = self.layer4(layer3_feat)
        return layer1_feat, layer2_feat, layer3_feat, layer4_feat


class adap_layer_feat3(nn.Module):
    def __init__(self):
        super(adap_layer_feat3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        
    def forward(self, feature):
        feature = feature + self.conv1(feature) 
        feature = feature + self.conv2(feature)
        return feature

    
class adap_layer_feat4(nn.Module):
    def __init__(self):
        super(adap_layer_feat4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )
            
    def forward(self, feature):
        feature = feature + self.conv1(feature)
        feature = feature + self.conv2(feature)
        return feature


class matching_layer(nn.Module):
    def __init__(self):
        super(matching_layer, self).__init__()
        self.relu = nn.ReLU()
        
    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, feature1, feature2):
        feature1 = self.L2normalize(feature1)
        feature2 = self.L2normalize(feature2)
        b, c, h1, w1 = feature1.size()
        b, c, h2, w2 = feature2.size()
        feature1 = feature1.view(b, c, h1 * w1)
        feature2 = feature2.view(b, c, h2 * w2)
        corr = torch.bmm(feature2.transpose(1, 2), feature1)
        corr = corr.view(b, h2 * w2, h1, w1) # Channel : target // Spatial grid : source
        corr = self.relu(corr)
        return corr


class find_correspondence(nn.Module):
    def __init__(self, feature_H, feature_W, beta, kernel_sigma):
        super(find_correspondence, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.beta = beta
        self.kernel_sigma = kernel_sigma
        
        # regular grid / [-1,1] normalized
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1,1,feature_W), np.linspace(-1,1,feature_H)) # grid_X & grid_Y : feature_H x feature_W
        self.grid_X = torch.tensor(self.grid_X, dtype=torch.float, requires_grad=False).to(device)
        self.grid_Y = torch.tensor(self.grid_Y, dtype=torch.float, requires_grad=False).to(device)
        
        # kernels for computing gradients
        self.dx_kernel = torch.tensor([-1,0,1], dtype=torch.float, requires_grad=False).view(1,1,1,3).expand(1,2,1,3).to(device)
        self.dy_kernel = torch.tensor([-1,0,1], dtype=torch.float, requires_grad=False).view(1,1,3,1).expand(1,2,3,1).to(device)
        
        # 1-d indices for generating Gaussian kernels
        self.x = np.linspace(0,feature_W-1,feature_W)
        self.x = torch.tensor(self.x, dtype=torch.float, requires_grad=False).to(device)
        self.y = np.linspace(0,feature_H-1,feature_H)
        self.y = torch.tensor(self.y, dtype=torch.float, requires_grad=False).to(device)
        
        # 1-d indices for kernel-soft-argmax / [-1,1] normalized
        self.x_normal = np.linspace(-1, 1,feature_W)
        self.x_normal = torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False).to(device)
        self.y_normal = np.linspace(-1, 1,feature_H)
        self.y_normal = torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False).to(device)

    def apply_gaussian_kernel(self, corr, sigma=5):
        b, hw, h, w = corr.size()

        idx = corr.max(dim=1)[1] # b x h x w    get maximum value along channel
        idx_y = (idx // w).view(b, 1, 1, h, w).float()
        idx_x = (idx % w).view(b, 1, 1, h, w).float()
        
        x = self.x.view(1,1,w,1,1).expand(b, 1, w, h, w)
        y = self.y.view(1,h,1,1,1).expand(b, h, 1, h, w)

        gauss_kernel = torch.exp(-((x-idx_x)**2 + (y-idx_y)**2) / (2 * sigma**2))
        gauss_kernel = gauss_kernel.view(b, hw, h, w)

        return gauss_kernel * corr
    
    def softmax_with_temperature(self, x, beta, d = 1):
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M  # subtract maximum value for stability
        return F.softmax(x * beta, dim=1)
    
    def kernel_soft_argmax(self, corr, apply_kernel=True, apply_softmax=True):
        b,_,h,w = corr.size()

        if apply_kernel:
            corr = self.apply_gaussian_kernel(corr, sigma=self.kernel_sigma)

        if apply_softmax:
            corr = self.softmax_with_temperature(corr, beta = self.beta, d = 1)
        else:
            # here this is supposed to be the results of softmax. sum(dim=1) equal to 1!
            # but not the case for bin. divide by sum so sum equal to 1
            corr = corr / (corr.sum(dim=1, keepdim=True) + 1e-8)

        corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
        x_normal = self.x_normal.expand(b,w)
        x_normal = x_normal.view(b,w,1,1)
        grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        
        grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
        y_normal = self.y_normal.expand(b,h)
        y_normal = y_normal.view(b,h,1,1)
        grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        return grid_x, grid_y, corr
    
    def get_flow_smoothness(self, flow, GT_mask):
        flow_dx = F.conv2d(F.pad(flow,(1,1,0,0)),self.dx_kernel)/2 # (padLeft, padRight, padTop, padBottom)
        flow_dy = F.conv2d(F.pad(flow,(0,0,1,1)),self.dy_kernel)/2 # (padLeft, padRight, padTop, padBottom)

        flow_dx = torch.abs(flow_dx) * GT_mask # consider foreground regions only
        flow_dy = torch.abs(flow_dy) * GT_mask
        
        smoothness = torch.cat((flow_dx, flow_dy), 1)
        return smoothness
    
    def forward(self, corr, GT_mask=None, apply_kernel=True, apply_softmax=True, return_corr=False):
        b,_,h,w = corr.size()
        grid_X = self.grid_X.expand(b, h, w) # x coordinates of a regular grid
        grid_X = grid_X.unsqueeze(1) # b x 1 x h x w
        grid_Y = self.grid_Y.expand(b, h, w) # y coordinates of a regular grid
        grid_Y = grid_Y.unsqueeze(1)
                
        if self.beta is not None:
            grid_x, grid_y, corr_softmaxed = self.kernel_soft_argmax(corr, apply_kernel=apply_kernel,
                                                                     apply_softmax=apply_softmax)
        else: # discrete argmax
            _,idx = torch.max(corr,dim=1)
            grid_x = idx % w
            grid_x = (grid_x.float() / (w-1) - 0.5) * 2
            grid_y = idx // w
            grid_y = (grid_y.float() / (h-1) - 0.5) * 2
            grid_x = grid_x.unsqueeze(1)
            grid_y = grid_y.unsqueeze(1)
            corr_softmaxed = corr
        
        grid = torch.cat((grid_x.permute(0,2,3,1), grid_y.permute(0,2,3,1)),3)
        # 2-channels@3rd-dim, first channel for x / second channel for y
        flow = torch.cat((grid_x - grid_X, grid_y - grid_Y),1)
        # 2-channels@1st-dim, first channel for x / second channel for y
        
        if GT_mask is None: # test
            if return_corr:
                return grid, flow, corr_softmaxed
            else:
                return grid, flow
        else: # train
            smoothness = self.get_flow_smoothness(flow, GT_mask)
            return grid, flow, smoothness


class SFNet(nn.Module):
    def __init__(self, feature_H=20, feature_W=20, beta=50.0, kernel_sigma=5.0,
                 inference_strategy='softargmax_padding', train_features=False, forward_pass_strategy=None):
        super(SFNet, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.feature_extraction = FeatureExtraction(train_features=train_features)
        self.adap_layer_feat3 = adap_layer_feat3()
        self.adap_layer_feat4 = adap_layer_feat4()
        self.matching_layer = matching_layer()
        self.find_correspondence = find_correspondence(feature_H, feature_W, beta=beta, kernel_sigma=kernel_sigma)
        self.feature_w = feature_W
        self.feature_h = feature_H
        self.inference_strategy = inference_strategy
        self.forward_pass_strategy = forward_pass_strategy

    @staticmethod
    def L2normalize(x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)    

    def load_state_dict(self, best_weights, strict: bool = True):
        if 'state_dict' in best_weights.keys():
            super(SFNet, self).load_state_dict(best_weights['state_dict'], strict=strict)
        elif 'state_dict1' in best_weights.keys():
            adap3_dict = best_weights['state_dict1']
            adap4_dict = best_weights['state_dict2']
            self.adap_layer_feat3.load_state_dict(adap3_dict, strict=False)
            self.adap_layer_feat4.load_state_dict(adap4_dict, strict=False)
        else:
            super(SFNet, self).load_state_dict(best_weights, strict=strict)

        if 'epoch' in best_weights.keys():
            self.epoch = best_weights['epoch']

    def parameters(self, recurse=True):
        return list(self.adap_layer_feat3.parameters(recurse)) + list(self.adap_layer_feat4.parameters(recurse))

    def forward(self, im_source, im_target, GT_src_mask=None, GT_tgt_mask=None, mode=None,
                *args, **kwargs):
        # Feature extraction
        src_feat1, src_feat2, src_feat3, src_feat4 = self.feature_extraction(im_source)
        # 256,80,80 // 512,40,40 // 1024,20,20 // 2048, 10, 10
        tgt_feat1, tgt_feat2, tgt_feat3, tgt_feat4 = self.feature_extraction(im_target)
        # Adaptation layers
        src_feat3 = self.adap_layer_feat3(src_feat3)
        tgt_feat3 = self.adap_layer_feat3(tgt_feat3)

        src_feat4 = self.adap_layer_feat4(src_feat4)
        src_feat4 = F.interpolate(src_feat4, scale_factor=2, mode='bilinear', align_corners=True)
        tgt_feat4 = self.adap_layer_feat4(tgt_feat4)
        tgt_feat4 = F.interpolate(tgt_feat4, scale_factor=2, mode='bilinear', align_corners=True)

        # Correlation S2T
        corr_feat3 = self.matching_layer(src_feat3, tgt_feat3)  # channel : target / spatial grid : source
        corr_feat4 = self.matching_layer(src_feat4, tgt_feat4)
        corr_S2T = corr_feat3 * corr_feat4
        corr_S2T = self.L2normalize(corr_S2T)
        # Correlation T2S
        b,_,h,w = corr_feat3.size()
        corr_feat3 = corr_feat3.view(b,h*w,h*w).transpose(1,2).view(b,h*w,h,w) # channel : source / spatial grid : target
        corr_feat4 = corr_feat4.view(b,h*w,h*w).transpose(1,2).view(b,h*w,h,w)
        corr_T2S = corr_feat3 * corr_feat4
        corr_T2S = self.L2normalize(corr_T2S)

        if self.forward_pass_strategy == 'corr_prediction_no_kernel' or mode == 'inference':
            return {'correlation_from_s_to_t': corr_S2T,
                    'correlation_from_t_to_s': corr_T2S, 'target_feat': tgt_feat4}
        else:
            # Establish correspondences
            grid_S2T, flow_S2T, smoothness_S2T = self.find_correspondence(corr_S2T, GT_src_mask)
            grid_T2S, flow_T2S, smoothness_T2S = self.find_correspondence(corr_T2S, GT_tgt_mask)
            
            # Estimate warped masks
            warped_src_mask = F.grid_sample(GT_tgt_mask, grid_S2T, mode='bilinear')
            warped_tgt_mask = F.grid_sample(GT_src_mask, grid_T2S, mode='bilinear')
            
            # Estimate warped flows
            warped_flow_S2T = -F.grid_sample(flow_T2S, grid_S2T, mode='bilinear') * GT_src_mask
            warped_flow_T2S = -F.grid_sample(flow_S2T, grid_T2S, mode='bilinear') * GT_tgt_mask
            flow_S2T = flow_S2T * GT_src_mask
            flow_T2S = flow_T2S * GT_tgt_mask

            return {'est_src_mask': warped_src_mask, 'smoothness_S2T':smoothness_S2T, 'grid_S2T':grid_S2T,
                    'est_tgt_mask': warped_tgt_mask, 'smoothness_T2S':smoothness_T2S, 'grid_T2S':grid_T2S,
                    'flow_S2T':flow_S2T, 'flow_T2S':flow_T2S,
                    'warped_flow_S2T':warped_flow_S2T, 'warped_flow_T2S':warped_flow_T2S,
                    'correlation_from_s_to_t': corr_S2T,
                    'correlation_from_t_to_s': corr_T2S,
                    }

    def pre_process_data(self, source_img, target_img, padding=True):
        # img has shape bx3xhxw
        # computes everything at 240 x 240
        device = self.device
        b, _, h_scale, w_scale = target_img.shape

        if padding:
            h_preprocessed = (self.feature_h - 2)*16
            w_preprocessed = (self.feature_w - 2)*16
        else:
            h_preprocessed = self.feature_h * 16
            w_preprocessed = self.feature_h * 16

        source_img = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                     size=(h_preprocessed, w_preprocessed),
                                                     mode='area')
        target_img = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                     size=(h_preprocessed, w_preprocessed),
                                                     mode='area')

        if padding:
            # padd with 16
            p2d = [16, 16, 16, 16]  # pad last dim by (1, 1) and 2nd to last by (2, 2)
            source_img = F.pad(source_img, p2d, "constant", 0)
            target_img = F.pad(target_img, p2d, "constant", 0)

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
                      return_corr=False, *args, **kwargs):
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
            return_corr: return correlation from target to source?

        Returns:
            flow_est: estimated flow field relating the target to the reference image,resized and scaled to output_shape
                      (can be defined by scaling parameter)
        """
        b, _, h_scale, w_scale = target_img.shape
        # define output_shape

        if output_shape is None and scaling != 1.0:
            output_shape = (int(h_scale * scaling), int(w_scale * scaling))
        elif output_shape is None and scaling == 1.0:
            output_shape = (h_scale, w_scale)

        padding = False
        if 'padding' in self.inference_strategy:
            # only for softargmax though! as in original work
            padding = True
        source_img, target_img, ratio_x, ratio_y = self.pre_process_data(source_img, target_img, padding=padding)

        output = self.forward(im_source=source_img, im_target=target_img, mode='inference')

        if self.inference_strategy == 'argmax':
            # this is for without padding!
            flow_est = correlation_to_flow_w_argmax(output['correlation_from_t_to_s'], output_shape, do_softmax=False)
        elif self.inference_strategy == 'argmax_w_kernel':
            corr_T2S = self.find_correspondence.apply_gaussian_kernel(output['correlation_from_t_to_s'],
                                                                      sigma=self.find_correspondence.kernel_sigma)
            flow_est = correlation_to_flow_w_argmax(corr_T2S, output_shape, do_softmax=False)
        else:
            # softargmax, corresponds to strategy used in original work (with padding)
            grid_T2S, flow_T2S, corr_T2S_softmaxed = \
                self.find_correspondence(output['correlation_from_t_to_s'], return_corr=True)
            small_grid = grid_T2S[:, 1:-1, 1:-1, :]
            small_grid[:, :, :, 0] = small_grid[:, :, :, 0] * (self.feature_w//2)/(self.feature_w//2 - 1)
            small_grid[:, :, :, 1] = small_grid[:, :, :, 1] * (self.feature_h//2)/(self.feature_h//2 - 1)
            small_grid = small_grid.permute(0, 3, 1, 2)
            grid = F.interpolate(small_grid, size=output_shape, mode='bilinear', align_corners=True)
            mapping = unnormalize(grid)

            flow_est = convert_mapping_to_flow(mapping)

        if mode != 'channel_first':
            return flow_est.permute(0, 2, 3, 1)
        if return_corr:
            h, w = output['correlation_from_t_to_s'].shape[-2:]
            corr = torch.nn.functional.softmax(output['correlation_from_t_to_s'].view(b, -1, h, w), dim=1)
            return flow_est, corr
        return flow_est

    def estimate_flow_and_confidence_map(self, source_img, target_img, output_shape=None, scaling=1.0,
                                         mode='channel_first'):
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


class SFNetWithBin(nn.Module):
    def __init__(self, feature_H=20, feature_W=20, beta=50.0, kernel_sigma=5.0, activation='stable_softmax',
                 temperature=1./50., initial_bin_value=1.0, forward_pass_strategy=None, inference_strategy='argmax',
                 train_features=False):
        super().__init__()
        self.matching_model = SFNet(feature_H, feature_W, beta, kernel_sigma,
                                    forward_pass_strategy=forward_pass_strategy, train_features=train_features)

        self.bin_model = LearntBinParam(initial_value=initial_bin_value)
        self.activation = activation
        self.temperature = temperature
        self.inference_strategy = inference_strategy

    @staticmethod
    def get_shape(correlation):
        if len(correlation.shape) == 3:
            b, c, hw = correlation.shape
            h = w = int(math.sqrt(hw))
        else:
            b = correlation.shape[0]
            h, w = correlation.shape[-2:]
        return b, h, w

    def cost_volume_to_probabilistic_mapping(self, A):
        """ Affinity -> Stochastic Matrix
        A is dimension B x C x H x W, matching points are in C
        """
        return cost_volume_to_probabilistic_mapping(A, self.activation, self.temperature)

    def forward(self, *args, **kwargs):

        correlation = self.matching_model(*args, **kwargs)
        # {'correlation_from_s_to_t': corr_S2T, 'correlation_from_t_to_s': corr_T2S}

        target_feat = correlation['target_feat']
        correlation = correlation['correlation_from_t_to_s']

        b, h, w = self.get_shape(correlation)
        correlation = self.bin_model(correlation=correlation, ref_feature=target_feat)
        correlation = self.cost_volume_to_probabilistic_mapping(correlation.view(b, -1, h, w))
        return correlation

    def load_state_dict(self, state_dict, strict=True):
        matching_model_dict = OrderedDict(
                [(k.replace('matching_model.', ''), v) for k, v in state_dict.items() if 'matching_model.' in k])

        bin_model_dict = OrderedDict(
                [(k.replace('bin_model.', ''), v) for k, v in state_dict.items() if 'bin_model.' in k])
        self.matching_model.load_state_dict(matching_model_dict, strict)
        self.bin_model.load_state_dict(bin_model_dict, strict)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def pre_process_data(self, source_img, target_img, padding=True):
        return self.matching_model.pre_process_data(source_img, target_img, padding)

    def estimate_flow(self, source_img, target_img, output_shape=None, scaling=1.0, mode='channel_first',
                      return_corr=False, *args, **kwargs):
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

        if self.inference_strategy == 'original_with_no_bin':
            return self.matching_model.estimate_flow(source_img, target_img, output_shape=output_shape, scaling=scaling,
                                                     mode=mode, return_corr=return_corr)
        else:
            padding = False
            if 'padding' in self.inference_strategy and 'argmax' not in self.inference_strategy:
                padding = True

            b, _, h_scale, w_scale = target_img.shape
            # define output_shape
            output_shape = (h_scale, w_scale)
            if output_shape is None and scaling != 1.0:
                output_shape = (int(h_scale * scaling), int(w_scale * scaling))

            source_img, target_img, ratio_x, ratio_y = self.matching_model.pre_process_data(source_img, target_img,
                                                                                            padding=padding)

            if output_shape is not None:
                ratio_x *= float(output_shape[1]) / float(w_scale)
                ratio_y *= float(output_shape[0]) / float(h_scale)
            else:
                output_shape = (h_scale, w_scale)

            correlation_from_t_to_s = self.forward(im_source=source_img, im_target=target_img, mode='inference')
            h, w = correlation_from_t_to_s.shape[-2:]
            assert correlation_from_t_to_s.shape[1] == h*w + 1

            # already applied kernel and softmax!
            # shape of this is b, h_s*w_s + 1, h_t*w_t
            correlation_from_t_to_s_ = correlation_from_t_to_s.view(b, -1, h, w)
            correlation_from_t_to_s = correlation_from_t_to_s.view(b, -1, h, w)[:, :h*w]  # remove the bin

            if self.inference_strategy == 'argmax':
                # with no padding!
                flow_est = correlation_to_flow_w_argmax(correlation_from_t_to_s, output_shape)
            else:
                # with padding or not,  softargmax
                grid_T2S, flow_T2S = self.matching_model.find_correspondence(correlation_from_t_to_s,
                                                                             apply_kernel=False, apply_softmax=False)
                # kernel already applied!
                small_grid = grid_T2S[:, 1:-1, 1:-1, :]
                small_grid[:, :, :, 0] = small_grid[:, :, :, 0] * (self.matching_model.feature_w // 2) / \
                                         (self.matching_model.feature_w // 2 - 1)
                small_grid[:, :, :, 1] = small_grid[:, :, :, 1] * (self.matching_model.feature_h // 2) / \
                                         (self.matching_model.feature_h // 2 - 1)
                small_grid = small_grid.permute(0, 3, 1, 2)
                grid = F.interpolate(small_grid, size=output_shape, mode='bilinear', align_corners=True)
                mapping = unnormalize(grid)

                flow_est = convert_mapping_to_flow(mapping)

            if mode != 'channel_first':
                return flow_est.permute(0, 2, 3, 1)

            if return_corr:
                return flow_est, correlation_from_t_to_s_
            return flow_est

    def estimate_flow_and_confidence_map(self, source_img, target_img, output_shape=None, scaling=1.0,
                                         mode='channel_first'):
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

        max_score, idx_B_Avec = torch.max(correlation_from_t_to_s[:, :h*w], dim=1)
        max_score = max_score.view(b, h, w)
        max_score = F.interpolate(max_score.unsqueeze(1), flow_est.shape[-2:], mode='bilinear', align_corners=False)
        uncertain_score = 1.0 / (max_score + 1e-8)

        non_occlusion_thresh = (1.0 - correlation_from_t_to_s[:, -1]).view(b, h, w)
        non_occlusion_thresh = F.interpolate(non_occlusion_thresh.unsqueeze(1), flow_est.shape[-2:],
                                             mode='bilinear', align_corners=False)
        max_score_and_non_occlusion = max_score * non_occlusion_thresh
        uncertain_max_score_and_non_occlusion = 1.0 / (max_score_and_non_occlusion + 1e-8)

        flow_est_backward, correlation_from_s_to_t = self.estimate_flow(target_img, source_img, output_shape=output_shape,
                                                                        scaling=scaling, mode=mode, return_corr=True)
        cyclic_consistency_error = torch.norm(flow_est + warp(flow_est_backward, flow_est), dim=1, p=2,
                                              keepdim=True)
        uncertainty_est = {'cyclic_consistency_error': cyclic_consistency_error,
                           'corr_score_uncertainty': uncertain_score,
                           'corr_score_and_occ_uncertainty': uncertain_max_score_and_non_occlusion}
        if mode == 'channel_first':
            return flow_est, uncertainty_est
        else:
            return flow_est.permute(0, 2, 3, 1), uncertainty_est


@model_constructor
def sfnet_with_bin(feature_H=20, feature_W=20, beta=50.0, kernel_sigma=5.0, activation='stable_softmax',
                   temperature=1./50., initial_bin_value=1.0, train_features=False, forward_pass_strategy=None):

    net = SFNetWithBin(feature_H, feature_W, beta, kernel_sigma, activation, temperature,
                       initial_bin_value=initial_bin_value, forward_pass_strategy=forward_pass_strategy,
                       train_features=train_features)
    return net
