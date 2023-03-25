# extracted and modified from NC-Net
from __future__ import print_function, division
import torch
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _quadruple
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import math
from packaging import version

from admin.model_constructor import model_constructor
from models.semantic_matching_models.eval_util_dynamic import bilinearInterpPointTnf, PointsToPixelCoords, PointsToUnitCoords
from models.non_matching_corr import LearntBinParam
from utils_flow.correlation_to_matches_utils import correlation_to_flow_w_argmax, corr_to_matches, \
    correlation_to_flow_w_soft_argmax, cost_volume_to_probabilistic_mapping
from utils_flow.pixel_wise_mapping import warp


def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)


def conv4d(data, filters, bias=None, permute_filters=True, use_half=False):
    b, c, h, w, d, t = data.size()

    # permute to avoid making contiguous inside loop
    data = data.permute(2, 0, 1, 3, 4, 5).contiguous()

    # Same permutation is done with filters, unless already provided with permutation
    if permute_filters:
        # permute to avoid making contiguous inside loop
        filters = filters.permute(2, 0, 1, 3, 4, 5).contiguous()

    c_out = filters.size(1)
    if use_half:
        output = Variable(
            torch.HalfTensor(h, b, c_out, w, d, t), requires_grad=data.requires_grad
        )
    else:
        output = Variable(
            torch.zeros(h, b, c_out, w, d, t), requires_grad=data.requires_grad
        )

    padding = filters.size(0) // 2
    if use_half:
        Z = Variable(torch.zeros(padding, b, c, w, d, t).half())
    else:
        Z = Variable(torch.zeros(padding, b, c, w, d, t))

    if data.is_cuda:
        Z = Z.cuda(data.get_device())
        output = output.cuda(data.get_device())

    data_padded = torch.cat((Z, data, Z), 0)

    for i in range(output.size(0)):  # loop on first feature dimension
        # convolve with center channel of filter (at position=padding)
        output[i, :, :, :, :, :] = F.conv3d(
            data_padded[i + padding, :, :, :, :, :],
            filters[padding, :, :, :, :, :],
            bias=bias,
            stride=1,
            padding=padding,
        )
        # convolve with upper/lower channels of filter (at postions [:padding] [padding+1:])
        for p in range(1, padding + 1):
            output[i, :, :, :, :, :] = output[i, :, :, :, :, :] + F.conv3d(
                data_padded[i + padding - p, :, :, :, :, :],
                filters[padding - p, :, :, :, :, :],
                bias=None,
                stride=1,
                padding=padding,
            )
            output[i, :, :, :, :, :] = output[i, :, :, :, :, :] + F.conv3d(
                data_padded[i + padding + p, :, :, :, :, :],
                filters[padding + p, :, :, :, :, :],
                bias=None,
                stride=1,
                padding=padding,
            )

    output = output.permute(1, 2, 0, 3, 4, 5).contiguous()
    return output


class Conv4d(_ConvNd):
    """
    Applies a 4D convolution over an input signal composed of several input planes.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        pre_permuted_filters=True, init=False
    ):
        # stride, dilation and groups !=1 functionality not tested
        stride = 1
        dilation = 1
        groups = 1
        # zero padding is added automatically in conv4d function to preserve tensor size
        padding = 0
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)
        if version.parse(torch.__version__) >= version.parse("1.3"):
            super(Conv4d, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                transposed=False, output_padding=_quadruple(0), groups=groups, bias=bias,
                padding_mode='zeros')
        else:
            super(Conv4d, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _quadruple(0), groups, bias)

        # weights will be sliced along one dimension during convolution loop
        # make the looping dimension to be the first one in the tensor,
        # so that we don't need to call contiguous() inside the loop
        self.pre_permuted_filters = pre_permuted_filters
        if self.pre_permuted_filters:
            self.weight.data = self.weight.data.permute(2, 0, 1, 3, 4, 5).contiguous()
        self.use_half = False

        if init:
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_in')
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        # filters pre-permuted in constructor
        return conv4d(
            input,
            self.weight,
            bias=self.bias,
            permute_filters=not self.pre_permuted_filters,
            use_half=self.use_half,
        )


class FeatureExtraction(torch.nn.Module):
    def __init__(self, train_fe=False, feature_extraction_cnn='resnet101', feature_extraction_model_file='',
                 normalization=True, last_layer='', use_cuda=True):
        super(FeatureExtraction, self).__init__()
        self.normalization = normalization
        self.feature_extraction_cnn = feature_extraction_cnn
        if feature_extraction_cnn == 'vgg':
            self.model = models.vgg16(pretrained=True)
            # keep feature extraction network up to indicated layer
            vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
                                  'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
                                  'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
                                  'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
                                  'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5']
            if last_layer == '':
                last_layer = 'pool4'
            last_layer_idx = vgg_feature_layers.index(last_layer)
            self.model = nn.Sequential(*list(self.model.features.children())[:last_layer_idx + 1])
        # for resnet below
        resnet_feature_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
        if feature_extraction_cnn == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            if last_layer == '':
                last_layer = 'layer3'
            resnet_module_list = [getattr(self.model, l) for l in resnet_feature_layers]
            last_layer_idx = resnet_feature_layers.index(last_layer)
            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx + 1])
        elif feature_extraction_cnn == 'densenet201':
            self.model = models.densenet201(pretrained=True)
            # keep feature extraction network up to denseblock3
            # self.model = nn.Sequential(*list(self.model.features.children())[:-3])
            # keep feature extraction network up to transitionlayer2
            self.model = nn.Sequential(*list(self.model.features.children())[:-4])
        else:
            raise ValueError

        if train_fe == False:
            # freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False
        # move to GPU
        if use_cuda:
            self.model = self.model.cuda()

    def forward(self, image_batch):
        features = self.model(image_batch)
        if self.normalization and not self.feature_extraction_cnn == 'resnet101fpn':
            features = featureL2Norm(features)
        return features


class FeatureCorrelation(torch.nn.Module):
    def __init__(self, shape='3D', normalization=True):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.shape = shape
        self.ReLU = nn.ReLU()

    def forward(self, feature_A, feature_B):
        if self.shape == '3D':
            b, c, h, w = feature_A.size()
            # reshape features for matrix multiplication
            feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
            feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
            # perform matrix mult.
            feature_mul = torch.bmm(feature_B, feature_A)
            # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        elif self.shape == '4D':
            b, c, hA, wA = feature_A.size()
            b, c, hB, wB = feature_B.size()
            # reshape features for matrix multiplication
            feature_A = feature_A.view(b, c, hA * wA).transpose(1, 2)  # size [b,c,h*w]
            feature_B = feature_B.view(b, c, hB * wB)  # size [b,c,h*w]
            # perform matrix mult.
            feature_mul = torch.bmm(feature_A, feature_B)
            # indexed [batch,row_A,col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b, hA, wA, hB, wB).unsqueeze(1)

        if self.normalization:
            correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))

        return correlation_tensor


class NeighConsensus(torch.nn.Module):
    def __init__(self, use_cuda=True, kernel_sizes=[3, 3, 3], channels=[10, 10, 1], symmetric_mode=True, leaky_relu=False, init=False):
        super(NeighConsensus, self).__init__()
        self.symmetric_mode = symmetric_mode
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            if i == 0:
                ch_in = 1
            else:
                ch_in = channels[i - 1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(Conv4d(in_channels=ch_in, out_channels=ch_out, kernel_size=k_size, bias=True, init=init))
            if leaky_relu:
                nn_modules.append(nn.LeakyReLU(0.1))
            else:
                nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)
        if use_cuda:
            self.conv.cuda()

    def forward(self, x):
        if self.symmetric_mode:
            # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
            # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
            x = self.conv(x) + self.conv(x.permute(0, 1, 4, 5, 2, 3)).permute(0, 1, 4, 5, 2, 3)
            # because of the ReLU layers in between linear layers,
            # this operation is different than convolving a single time with the filters+filters^T
            # and therefore it makes sense to do this.
        else:
            x = self.conv(x)
        return x


def MutualMatching(corr4d):
    # mutual matching
    batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()

    corr4d_B = corr4d.view(batch_size, fs1 * fs2, fs3, fs4)  # [batch_idx,k_A,i_B,j_B]
    corr4d_A = corr4d.view(batch_size, fs1, fs2, fs3 * fs4)

    # get max
    corr4d_B_max, _ = torch.max(corr4d_B, dim=1, keepdim=True)
    corr4d_A_max, _ = torch.max(corr4d_A, dim=3, keepdim=True)

    eps = 1e-5
    corr4d_B = corr4d_B / (corr4d_B_max + eps)
    corr4d_A = corr4d_A / (corr4d_A_max + eps)

    corr4d_B = corr4d_B.view(batch_size, 1, fs1, fs2, fs3, fs4)
    corr4d_A = corr4d_A.view(batch_size, 1, fs1, fs2, fs3, fs4)

    corr4d = corr4d * (corr4d_A * corr4d_B)  # parenthesis are important for symmetric output

    return corr4d


def maxpool4d(corr4d_hres, k_size=4):
    slices = []
    for i in range(k_size):
        for j in range(k_size):
            for k in range(k_size):
                for l in range(k_size):
                    slices.append(corr4d_hres[:, 0, i::k_size, j::k_size, k::k_size, l::k_size].unsqueeze(0))
    slices = torch.cat(tuple(slices), dim=1)
    corr4d, max_idx = torch.max(slices, dim=1, keepdim=True)
    max_l = torch.fmod(max_idx, k_size)
    max_k = torch.fmod(max_idx.sub(max_l).div(k_size), k_size)
    max_j = torch.fmod(max_idx.sub(max_l).div(k_size).sub(max_k).div(k_size), k_size)
    max_i = max_idx.sub(max_l).div(k_size).sub(max_k).div(k_size).sub(max_j).div(k_size)
    # i,j,k,l represent the *relative* coords of the max point in the box of size k_size*k_size*k_size*k_size
    return (corr4d, max_i, max_j, max_k, max_l)


class ImMatchNet(nn.Module):
    def __init__(self,
                 feature_extraction_cnn='resnet101',
                 feature_extraction_last_layer='',
                 feature_extraction_model_file=None,
                 return_correlation=False,
                 ncons_kernel_sizes=[5, 5, 5],
                 ncons_channels=[16, 16, 1],
                 normalize_features=True,
                 train_fe=False,
                 use_cuda=True,
                 relocalization_k_size=0,
                 half_precision=False,
                 checkpoint=None,
                 leaky_relu=False, init=False, inference_strategy='argmax'
                 ):

        super(ImMatchNet, self).__init__()
        # Load checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # either gpu or cpu
        if checkpoint is not None and checkpoint != '':
            print('Loading checkpoint...')
            checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            checkpoint['state_dict'] = OrderedDict(
                [(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
            # override relevant parameters
            if 'epoch' in checkpoint.keys():
                print('epoch {}'.format(checkpoint['epoch']))
                self.epoch = checkpoint['epoch']

            if 'args' in checkpoint.keys():
                print(checkpoint['args'])
                print('Using checkpoint parameters: ')
                ncons_channels = checkpoint['args'].ncons_channels
                print('  ncons_channels: ' + str(ncons_channels))
                ncons_kernel_sizes = checkpoint['args'].ncons_kernel_sizes
                print('  ncons_kernel_sizes: ' + str(ncons_kernel_sizes))

        self.use_cuda = use_cuda
        self.normalize_features = normalize_features
        self.return_correlation = return_correlation
        self.relocalization_k_size = relocalization_k_size
        self.half_precision = half_precision

        self.FeatureExtraction = FeatureExtraction(train_fe=train_fe,
                                                   feature_extraction_cnn=feature_extraction_cnn,
                                                   feature_extraction_model_file=feature_extraction_model_file,
                                                   last_layer=feature_extraction_last_layer,
                                                   normalization=normalize_features,
                                                   use_cuda=self.use_cuda)

        self.FeatureCorrelation = FeatureCorrelation(shape='4D', normalization=False)

        self.NeighConsensus = NeighConsensus(use_cuda=self.use_cuda,
                                             kernel_sizes=ncons_kernel_sizes,
                                             channels=ncons_channels,
                                             leaky_relu=leaky_relu, init=init)

        # Load weights
        if checkpoint is not None and checkpoint != '':
            print('Copying weights...')
            for name, param in self.FeatureExtraction.state_dict().items():
                if 'num_batches_tracked' not in name:
                    self.FeatureExtraction.state_dict()[name].copy_(
                        checkpoint['state_dict']['FeatureExtraction.' + name])
            for name, param in self.NeighConsensus.state_dict().items():
                self.NeighConsensus.state_dict()[name].copy_(checkpoint['state_dict']['NeighConsensus.' + name])
            print('Done!')

        self.FeatureExtraction.eval()
        self.inference_strategy = inference_strategy

        if self.half_precision:
            for p in self.NeighConsensus.parameters():
                p.data = p.data.half()
            for l in self.NeighConsensus.conv:
                if isinstance(l, Conv4d):
                    l.use_half = True

    def load_state_dict(self, state_dict, strict=True):
        for name, param in self.FeatureExtraction.state_dict().items():
            if 'num_batches_tracked' not in name:
                self.FeatureExtraction.state_dict()[name].copy_(
                    state_dict['FeatureExtraction.' + name])
        for name, param in self.NeighConsensus.state_dict().items():
            self.NeighConsensus.state_dict()[name].copy_(state_dict['NeighConsensus.' + name])
        print('Loaded weights in NC-Net')

    # used only for foward pass at eval and for training with strong supervision
    def forward(self, im_source, im_target, *args, **kwargs):
        # feature extraction
        feature_A = self.FeatureExtraction(im_source)
        feature_B = self.FeatureExtraction(im_target)
        if self.half_precision:
            feature_A = feature_A.half()
            feature_B = feature_B.half()
        # feature correlation
        corr4d = self.FeatureCorrelation(feature_A, feature_B)
        # do 4d maxpooling for relocalization
        if self.relocalization_k_size > 1:
            corr4d, max_i, max_j, max_k, max_l = maxpool4d(corr4d, k_size=self.relocalization_k_size)
        # run match processing model
        corr4d = MutualMatching(corr4d)
        corr4d = self.NeighConsensus(corr4d)
        corr4d = MutualMatching(corr4d)

        if self.relocalization_k_size > 1:
            delta4d = (max_i, max_j, max_k, max_l)
            return (corr4d, delta4d)
        else:
            return corr4d

    def train(self, bool_=True):
        self.NeighConsensus.train(bool_)
        self.FeatureExtraction.eval()

    def pre_process_data(self, source_img, target_img):
        # img has shape bx3xhxw
        # computes everything at 400 x 400
        # this is what NC-Net does in the original code
        device = self.device
        b, _, h_scale, w_scale = target_img.shape

        h_preprocessed = 400
        w_preprocessed = 400

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

        b, _, h_scale, w_scale = target_img.shape
        # define output_shape
        if output_shape is None and scaling != 1.0:
            output_shape = (int(h_scale * scaling), int(w_scale * scaling))
        elif output_shape is None:
            output_shape = (h_scale, w_scale)

        source_img, target_img, ratio_x, ratio_y = self.pre_process_data(source_img, target_img)

        correlation_from_t_to_s = self.forward(im_source=source_img, im_target=target_img)
        h_, w_ = correlation_from_t_to_s.shape[-2:]
        correlation_from_t_to_s = correlation_from_t_to_s.view(b, -1, h_, w_)

        if self.inference_strategy == 'argmax':
            # like in original work
            flow_est = correlation_to_flow_w_argmax(correlation_from_t_to_s, output_shape=output_shape,
                                                    do_softmax=True)
            # flow_est = get_dense_flow(correlation_from_t_to_s, output_shape, self.device)\
            #    .unsqueeze(0).to(self.device).float()

        else:
            # softargmax
            flow_est = correlation_to_flow_w_soft_argmax(correlation_from_t_to_s, output_shape=output_shape,
                                                         temperature=1.0, apply_softmax=True, stable_softmax=False)

        if mode != 'channel_first':
            flow_est = flow_est.permute(0, 2, 3, 1)

        if return_corr:
            correlation_from_t_to_s = torch.nn.functional.softmax(correlation_from_t_to_s.view(b, -1, h_, w_), dim=1)
            return flow_est, correlation_from_t_to_s
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


def get_dense_flow(correlation_from_t_to_s, output_shape, device):

    # softmax done before
    xA, yA, xB, yB, sB = corr_to_matches(correlation_from_t_to_s, do_softmax=False, scale='centered')
    matches = (xA, yA, xB, yB)
    h_tgt, w_tgt = output_shape[:2]
    X, Y = np.meshgrid(np.linspace(0, w_tgt - 1, w_tgt),
                       np.linspace(0, h_tgt - 1, h_tgt))

    grid_X_vec = torch.from_numpy(X).view(-1, 1).float()
    grid_Y_vec = torch.from_numpy(Y).view(-1, 1).float()

    grid_XY_vec = torch.cat((grid_X_vec, grid_Y_vec), 1).permute(1, 0).to(device)

    target_points_ = grid_XY_vec.to(device)
    target_points_norm = PointsToUnitCoords(target_points_.unsqueeze(0),
                                            torch.Tensor(output_shape).unsqueeze(0).to(device))
    # 1, 2, N

    # compute points stage 1 only
    warped_points_norm = bilinearInterpPointTnf(matches, target_points_norm)
    warped_points = PointsToPixelCoords(warped_points_norm, torch.Tensor(output_shape).unsqueeze(0).to(device))
    warped_points = torch.t(warped_points.squeeze())
    # N, 2
    warped_points = warped_points.reshape(output_shape[0], output_shape[1], 2)
    return warped_points.permute(2, 0, 1) - grid_XY_vec.reshape(2, output_shape[0], output_shape[1])  # 2, h, w


def get_sparse_flow(correlation_from_t_to_s, output_shape, device, batch):

    # softmax done before
    xA, yA, xB, yB, sB = corr_to_matches(correlation_from_t_to_s, do_softmax=False, scale='centered')
    matches = (xA, yA, xB, yB)
    target_points = batch['target_kps'].clone()[0, :batch['n_pts'][0]].to(device)
    target_points_ = torch.t(target_points)
    target_points_norm = PointsToUnitCoords(target_points_.unsqueeze(0),
                                            torch.Tensor(output_shape).unsqueeze(0).to(device))
    # 1, 2, N

    # compute points stage 1 only
    warped_points_norm = bilinearInterpPointTnf(matches, target_points_norm)
    warped_points = PointsToPixelCoords(warped_points_norm, torch.Tensor(output_shape).unsqueeze(0).to(device))
    warped_points = torch.t(warped_points.squeeze())

    valid_target = torch.round(target_points[:, 0]).le(output_shape[1] - 1) & \
                   torch.round(target_points[:, 1]).le(output_shape[0] - 1) & \
                   torch.round(target_points[:, 0]).ge(0) & torch.round(target_points[:, 1]).ge(0)
    # valid = valid_source * valid_target
    valid = valid_target
    target_points = target_points[valid]

    predicted_source_coords = warped_points[valid]

    flow_est = torch.zeros(output_shape[0], output_shape[0], 2).to(device)
    flow_est[torch.round(target_points[:, 1]).long(), torch.round(target_points[:, 0]).long()] = \
        predicted_source_coords - target_points
    flow_est = flow_est.unsqueeze(0).permute(0, 3, 1, 2)
    return flow_est


class NCNetWithBin(nn.Module):
    def __init__(self, feature_extraction_cnn='resnet101',
                 feature_extraction_last_layer='',
                 feature_extraction_model_file=None,
                 return_correlation=False,
                 ncons_kernel_sizes=[5, 5, 5],
                 ncons_channels=[16, 16, 1],
                 normalize_features=True,
                 train_fe=False,
                 relocalization_k_size=0,
                 half_precision=False,
                 initial_bin_value=1.0,
                 activation='softmax',
                 temperature=1.0, inference_strategy='argmax',
                 leaky_relu=False, init=False):
        super().__init__()
        use_cuda = True
        self.matching_model = ImMatchNet(feature_extraction_cnn, feature_extraction_last_layer,
                                         feature_extraction_model_file, return_correlation, ncons_kernel_sizes,
                                         ncons_channels, normalize_features, train_fe, use_cuda,
                                         relocalization_k_size, half_precision, leaky_relu=leaky_relu, init=init,
                                         inference_strategy=inference_strategy)
        self.bin_model = LearntBinParam(initial_value=initial_bin_value)
        self.activation = activation
        self.temperature = temperature
        self.inference_strategy = inference_strategy

    def train(self, bool_=True):
        self.matching_model.train(bool_)
        self.bin_model.train(bool_)

    @staticmethod
    def get_shape(correlation):
        if len(correlation.shape) == 3:
            b, c, hw = correlation.shape
            h = w = int(math.sqrt(hw))
        else:
            b = correlation.shape[0]
            h, w = correlation.shape[-2:]
        return b, h, w

    def stoch_mat(self, A):
        """ Affinity -> Stochastic Matrix
        A is dimension B x C x H x W, matching points are in C
        """
        return cost_volume_to_probabilistic_mapping(A, self.activation, self.temperature)

    def forward(self, *args, **kwargs):

        output = self.matching_model(*args, **kwargs)
        if isinstance(output, dict):
            output = output['correlation_from_t_to_s']
        b, h, w = self.get_shape(output)

        output = self.bin_model(output)
        output = self.stoch_mat(output.view(b, -1, h, w))
        return output

    def load_state_dict(self, state_dict, strict=True):
        matching_model_dict = OrderedDict(
                [(k.replace('matching_model.', ''), v) for k, v in state_dict.items() if 'matching_model.' in k])

        bin_model_dict = OrderedDict(
                [(k.replace('bin_model.', ''), v) for k, v in state_dict.items() if 'bin_model.' in k])
        self.matching_model.load_state_dict(matching_model_dict, strict)
        self.bin_model.load_state_dict(bin_model_dict, strict)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def estimate_flow(self, source_img, target_img, output_shape=None, scaling=1.0, mode='channel_first',
                      return_corr=False,  *args, **kwargs):
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

        source_img, target_img, ratio_x, ratio_y = self.matching_model.pre_process_data(source_img, target_img)

        correlation_from_t_to_s = self.forward(im_source=source_img, im_target=target_img)
        # shape of this is b, h_s*w_s + 1, h_t*w_t
        h, w = correlation_from_t_to_s.shape[-2:]
        correlation_from_t_to_s = correlation_from_t_to_s.view(b, -1, h, w)
        assert correlation_from_t_to_s.shape[1] == h*w + 1

        # remove bin channel
        correlation_from_t_to_s_ = correlation_from_t_to_s
        correlation_from_t_to_s = correlation_from_t_to_s[:, :h*w]  # remove the bin
        if self.inference_strategy == 'argmax':
            # flow_est = convert_correlation_to_flow(correlation_from_t_to_s, output_shape)
            flow_est = get_dense_flow(correlation_from_t_to_s.view(b, h, w, h, w).unsqueeze(1), output_shape,
                                      self.matching_model.device)\
               .unsqueeze(0).to(self.matching_model.device).float()
        else:
            # softargmax
            flow_est = correlation_to_flow_w_soft_argmax(correlation_from_t_to_s, output_shape=output_shape,
                                                         temperature=1.0, apply_softmax=False, stable_softmax=False)

        if mode != 'channel_first':
            return flow_est.permute(0, 2, 3, 1)

        if return_corr:
            return flow_est, correlation_from_t_to_s_
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

        max_score, idx_B_Avec = torch.max(correlation_from_t_to_s[:, :h*w], dim=1)
        max_score = max_score.view(b, h, w)
        max_score = F.interpolate(max_score.unsqueeze(1), flow_est.shape[-2:], mode='bilinear', align_corners=False)
        uncertain_score = 1.0 / (max_score + 1e-8)

        non_occlusion_thresh = (1.0 - correlation_from_t_to_s[:, -1]).view(b, h, w)
        non_occlusion_thresh = F.interpolate(non_occlusion_thresh.unsqueeze(1), flow_est.shape[-2:],
                                             mode='bilinear', align_corners=False)
        max_score_and_non_occlusion = max_score * non_occlusion_thresh
        uncertain_max_score_and_non_occlusion = 1.0 / (max_score_and_non_occlusion + 1e-8)

        flow_est_backward, correlation_from_s_to_t = self.estimate_flow(target_img, source_img,
                                                                        output_shape=output_shape,
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
def ncnet_with_bin(feature_extraction_cnn='resnet101', feature_extraction_last_layer='',
                   feature_extraction_model_file=None, return_correlation=False, ncons_kernel_sizes=[5, 5, 5],
                   ncons_channels=[16, 16, 1], normalize_features=True, train_fe=False, relocalization_k_size=0,
                   half_precision=False, initial_bin_value=1.0, activation='softmax', temperature=1.0, leaky_relu=False,
                   init=False):

    net = NCNetWithBin(feature_extraction_cnn, feature_extraction_last_layer, feature_extraction_model_file,
                       return_correlation, ncons_kernel_sizes, ncons_channels, normalize_features,
                       train_fe, relocalization_k_size, half_precision,  initial_bin_value, activation, temperature,
                       leaky_relu=leaky_relu, init=init)
    return net
