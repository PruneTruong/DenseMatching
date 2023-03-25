import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
import torch.nn.functional as F
from packaging import version


from models.modules.mod import CMDTop, OpticalFlowEstimator, deconv, conv, predict_flow, \
    unnormalise_and_convert_mapping_to_flow
from models.modules.feature_correlation_layer import FeatureL2Norm, GlobalFeatureCorrelationLayer
from models.modules.consensus_network_modules import MutualMatching, NeighConsensus, FeatureCorrelation
from models.modules.local_correlation import correlation  # the custom cost volume layer
from models.base_matching_net import BaseGLUMultiScaleMatchingNet, set_glunet_parameters
from models.inference_utils import matches_from_flow, estimate_mask
from models.modules.bilinear_deconv import BilinearConvTranspose2d
from utils_flow.flow_and_mapping_operations import convert_flow_to_mapping


class VGGPyramid(nn.Module):
    def __init__(self, train=False, pretrained=True):
        super().__init__()
        self.n_levels = 5
        source_model = models.vgg16(pretrained=pretrained)

        modules = OrderedDict()
        tmp = []
        n_block = 0
        first_relu = False

        for c in source_model.features.children():
            if (isinstance(c, nn.ReLU) and not first_relu) or (isinstance(c, nn.MaxPool2d)):
                first_relu = True

                if isinstance(c, nn.MaxPool2d):
                    c = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # replace with ceil mode True
                tmp.append(c)
                modules['level_' + str(n_block)] = nn.Sequential(*tmp)
                for param in modules['level_' + str(n_block)].parameters():
                    param.requires_grad = train

                tmp = []
                n_block += 1
            else:
                tmp.append(c)

            if n_block == self.n_levels:
                break

        self.__dict__['_modules'] = modules

    def remove_specific_layers(self, remove_layers=[]):
        # example ['layer4']
        if len(remove_layers) > 0:
            for layer in remove_layers:
                nb_layer = int(layer.split('layer')[1])
                self.n_levels = nb_layer
                print(self.n_levels)

    def modify_stride_and_padding(self):
        # remove last max pooling
        removed = list(self.__dict__['_modules']['level_4'].children())[:-1]
        self.__dict__['_modules']['level_4'] = torch.nn.Sequential(*removed)

    def get_final_output(self, x):
        # last layer is usually 1/16 of original size (without modifications)
        for layer_n in range(0, self.n_levels):
            x = self.__dict__['_modules']['level_' + str(layer_n)](x)
        return x

    def forward(self, x, quarter_resolution_only=False, eigth_resolution=False, additional_coarse_level=False,
                return_only_final_output=False, *args, **kwargs):

        if return_only_final_output:
            return self.get_final_output(x)

        outputs = []
        if quarter_resolution_only:
            x_full = self.__dict__['_modules']['level_' + str(0)](x)
            x_half = self.__dict__['_modules']['level_' + str(1)](x_full)
            x_quarter = self.__dict__['_modules']['level_' + str(2)](x_half)
            outputs.append(x_quarter)
        elif eigth_resolution:
            x_full = self.__dict__['_modules']['level_' + str(0)](x)
            outputs.append(x_full)
            x_half = self.__dict__['_modules']['level_' + str(1)](x_full)
            x_quarter = self.__dict__['_modules']['level_' + str(2)](x_half)
            outputs.append(x_quarter)
            x_eight = self.__dict__['_modules']['level_' + str(3)](x_quarter)
            outputs.append(x_eight)
        else:
            for layer_n in range(0, self.n_levels):
                x = self.__dict__['_modules']['level_' + str(layer_n)](x)
                outputs.append(x)

            if version.parse(torch.__version__) >= version.parse("1.6"):
                x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False,
                                                    recompute_scale_factor=True)
            else:
                x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
            outputs.append(x)
        return outputs


class SemanticGLUNetModel(BaseGLUMultiScaleMatchingNet):
    """
    Semantic-GLU-Net.
    The flows (flow2, flow1) are predicted such that they are scaled to the input image resolution. To obtain the flow
    from target to source at original resolution, one just needs to bilinearly upsample (without further scaling).
    """
    def __init__(self, batch_norm=True, pyramid_type='VGG', md=4, init_deconv_w_bilinear=True,
                 cyclic_consistency=False, consensus_network=True, iterative_refinement=False):

        # not used here, just to make it easier and use new framework
        params = set_glunet_parameters()
        super().__init__(params)
        self.div = 1.0
        self.pyramid_type = pyramid_type
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.iterative_refinement = iterative_refinement
        self.cyclic_consistency = cyclic_consistency
        self.consensus_network = consensus_network
        if self.cyclic_consistency:
            self.corr = FeatureCorrelation(shape='4D', normalization=False)
        elif consensus_network:
            ncons_kernel_sizes = [3, 3, 3]
            ncons_channels = [10, 10, 1]
            self.corr = FeatureCorrelation(shape='4D', normalization=False)
            # normalisation is applied in code here

            self.NeighConsensus = NeighConsensus(use_cuda=True,
                                                 kernel_sizes=ncons_kernel_sizes,
                                                 channels=ncons_channels)
        else:
            self.corr = GlobalFeatureCorrelationLayer()
        # L2 feature normalisation
        self.l2norm = FeatureL2Norm()

        dd = np.cumsum([128,128,96,64,32])
        # weights for decoder at different levels
        nd = 16*16  # global correlation
        od = nd + 2
        self.decoder4 = CMDTop(in_channels=od, batch_norm=batch_norm)

        if init_deconv_w_bilinear:
            self.deconv4 = BilinearConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
        else:
            self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        nd = (2*md+1)**2  # constrained correlation, 4 pixels on each side
        od = nd + 2
        self.decoder3 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)

        # weights for refinement module
        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1, batch_norm=batch_norm)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2, batch_norm=batch_norm)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4, batch_norm=batch_norm)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8, batch_norm=batch_norm)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1, batch_norm=batch_norm)
        self.dc_conv7 = predict_flow(32)

        # 1/8 of original resolution
        nd = (2*md+1)**2  # constrained correlation, 4 pixels on each side
        od = nd + 2  # only gets the upsampled flow
        self.decoder2 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)

        if init_deconv_w_bilinear:
            self.deconv2 = BilinearConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
        else:
            self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        self.upfeat2 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)

        # 1/4 of original resolution
        nd = (2*md+1)**2  # constrained correlation, 4 pixels on each side
        od = nd + 4
        self.decoder1 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)

        self.l_dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1, batch_norm=batch_norm)
        self.l_dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2, batch_norm=batch_norm)
        self.l_dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4, batch_norm=batch_norm)
        self.l_dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8, batch_norm=batch_norm)
        self.l_dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
        self.l_dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1, batch_norm=batch_norm)
        self.l_dc_conv7 = predict_flow(32)

        # initialize modules
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # In earlier versions batch norm parameters was initialized with default initialization,
                # which changed in pytorch 1.2. In 1.1 and earlier the weight was set to U(0,1).
                # So we use the same initialization here.
                # m.weight.data.fill_(1)
                m.weight.data.uniform_()
                m.bias.data.zero_()

        if pyramid_type == 'VGG':
            self.pyramid = VGGPyramid()
        else:
            ValueError('The pyramid that you chose does not exist: {}'.format(pyramid_type))

    def coarsest_resolution_flow(self, c14, c24, h_256, w_256):
        ratio_x = 16.0 / float(w_256)
        ratio_y = 16.0 / float(h_256)
        b = c14.shape[0]
        if self.cyclic_consistency:
            corr4d = self.corr(self.l2norm(c24), self.l2norm(c14))  # first source, then target
            # run match processing model
            corr4d = MutualMatching(corr4d)
            corr4 = corr4d.squeeze(1).view(b, c24.shape[2] * c24.shape[3], c14.shape[2], c14.shape[3])
        elif self.consensus_network:
            corr4d = self.corr(self.l2norm(c24), self.l2norm(c14))  # first source, then target
            # run match processing model
            corr4d = MutualMatching(corr4d)
            corr4d = self.NeighConsensus(corr4d)
            corr4d = MutualMatching(corr4d)  # size is [b, 1, hsource, wsource, htarget, wtarget]
            corr4 = corr4d.squeeze(1).view(c24.shape[0], c24.shape[2] * c24.shape[3], c14.shape[2], c14.shape[3])
        else:
            corr4 = self.corr(self.l2norm(c24), self.l2norm(c14))
        corr4 = self.l2norm(F.relu(corr4))
        b, c, h, w = corr4.size()
        if torch.cuda.is_available():
            init_map = torch.FloatTensor(b, 2, h, w).zero_().cuda()
        else:
            init_map = torch.FloatTensor(b, 2, h, w).zero_()
        est_map4 = self.decoder4(x1=corr4, x3=init_map)
        # conversion to flow and from there constrained correlation
        flow4 = unnormalise_and_convert_mapping_to_flow(est_map4) / self.div
        flow4[:, 0, :, :] /= ratio_x
        flow4[:, 1, :, :] /= ratio_y
        return corr4, flow4

    def forward(self, im_target, im_source, im_target_256, im_source_256, im_target_pyr=None, im_source_pyr=None,
                im_target_pyr_256=None, im_source_pyr_256=None):
        """
        Args:
            im_target: torch Tensor Bx3xHxW, normalized with imagenet weights
            im_source: torch Tensor Bx3xHxW, normalized with imagenet weights
            im_target_256: torch Tensor Bx3x256x256, normalized with imagenet weights
            im_source_256: torch Tensor Bx3x256x256, normalized with imagenet weights
            im_target_pyr: in case the pyramid features are already computed.
            im_source_pyr: in case the pyramid features are already computed.
            im_target_pyr_256: in case the pyramid features are already computed.
            im_source_pyr_256: in case the pyramid features are already computed.

        Returns:
            output_256: dict with keys 'flow_estimates'. It contains the flow field of the two deepest levels
                        corresponding to the L-Net (flow4 and flow3), they are scaled for input resolution of 256x256.
            output: dict with keys 'flow_estimates'. It contains the flow field of the two shallowest levels
                    corresponding to the H-Net (flow2 and flow1), they are scaled for original (high resolution)
                    input resolution.
        """
        # all indices 1 refer to target images
        # all indices 2 refer to source images

        b, _, h_original, w_original = im_target.size()
        b, _, h_256, w_256 = im_target_256.size()
        div = self.div

        # pyramid, original reso
        if im_target_pyr is None:
            im_target_pyr = self.pyramid(im_target, eigth_resolution=True)
        if im_source_pyr is None:
            im_source_pyr = self.pyramid(im_source, eigth_resolution=True)
        c11 = im_target_pyr[-2]  # size original_res/4xoriginal_res/4
        c21 = im_source_pyr[-2]
        c12 = im_target_pyr[-1]  # size original_res/8xoriginal_res/8
        c22 = im_source_pyr[-1]

        # pyramid, 256 reso
        if im_target_pyr_256 is None:
            im_target_pyr_256 = self.pyramid.forward(im_target_256, additional_coarse_level=True)
        if im_source_pyr_256 is None:
            im_source_pyr_256 = self.pyramid.forward(im_source_256, additional_coarse_level=True)
        c13 = im_target_pyr_256[-3]
        c23 = im_source_pyr_256[-3]
        c14 = im_target_pyr_256[-2]
        c24 = im_source_pyr_256[-2]
        c15 = im_target_pyr_256[-1]
        c25 = im_source_pyr_256[-1]

        # RESOLUTION 256x256
        # level 16x16
        c24_concat = torch.cat((c24, F.interpolate(input=c25, size=(16, 16), mode='bilinear', align_corners=False)), 1)
        c14_concat = torch.cat((c14, F.interpolate(input=c15, size=(16, 16), mode='bilinear', align_corners=False)), 1)
        corr4, flow4 = self.coarsest_resolution_flow(c14_concat, c24_concat, h_256, w_256)
        up_flow4 = self.deconv4(flow4)

        # level 32x32
        ratio_x = 32.0 / float(w_256)
        ratio_y = 32.0 / float(h_256)
        up_flow_4_warping = up_flow4 * div
        up_flow_4_warping[:, 0, :, :] *= ratio_x
        up_flow_4_warping[:, 1, :, :] *= ratio_y
        c23_concat = torch.cat((c23, F.interpolate(input=c24, size=(32, 32), mode='bilinear', align_corners=False),
                                F.interpolate(input=c25, size=(32, 32), mode='bilinear', align_corners=False)), 1)
        c13_concat = torch.cat((c13, F.interpolate(input=c14, size=(32, 32), mode='bilinear', align_corners=False),
                                F.interpolate(input=c15, size=(32, 32), mode='bilinear', align_corners=False)), 1)
        warp3 = self.warp(c23_concat, up_flow_4_warping)
        # constrained correlation now
        corr3 = correlation.FunctionCorrelation(reference_features=c13_concat, query_features=warp3)
        corr3 = self.leakyRELU(corr3)
        corr3 = torch.cat((corr3, up_flow4), 1)
        x, res_flow3 = self.decoder3(corr3)
        flow3 = res_flow3 + up_flow4
        # flow 3 refined (at 32x32 resolution)
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow3 = flow3 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        if self.iterative_refinement:
            # from 32x32 resolution, if upsampling to 1/8*original resolution is too big,
            # do iterative upsampling so that gap is always smaller than 2.
            R_w = float(w_original)/8.0/32.0
            R_h = float(h_original)/8.0/32.0
            if R_w > R_h:
                R = R_w
            else:
                R = R_h

            minimum_ratio = 3.0
            nbr_extra_layers = max(0, int(round(np.log(R/minimum_ratio)/np.log(2))))

            if nbr_extra_layers == 0:
                flow3[:, 0, :, :] *= float(w_original) / float(256)
                flow3[:, 1, :, :] *= float(h_original) / float(256)
                # ==> put the upflow in the range [Horiginal x Woriginal]
            else:
                # adding extra layers
                flow3[:, 0, :, :] *= float(w_original) / float(256)
                flow3[:, 1, :, :] *= float(h_original) / float(256)
                for n in range(nbr_extra_layers):
                    ratio = 1.0 / (8.0 * 2 ** (nbr_extra_layers - n ))
                    up_flow3 = F.interpolate(input=flow3, size=(int(h_original * ratio), int(w_original * ratio)),
                                             mode='bilinear',
                                             align_corners=False)
                    c23_bis = torch.nn.functional.interpolate(c22, size=(int(h_original * ratio), int(w_original * ratio)), mode='area')
                    c13_bis = torch.nn.functional.interpolate(c12, size=(int(h_original * ratio), int(w_original * ratio)), mode='area')
                    warp3 = self.warp(c23_bis, up_flow3 * div * ratio)
                    corr3 = correlation.FunctionCorrelation(reference_features=c13_bis, query_features=warp3)
                    corr3 = self.leakyRELU(corr3)
                    corr3 = torch.cat((corr3, up_flow3), 1)
                    x, res_flow3 = self.decoder2(corr3)
                    flow3 = res_flow3 + up_flow3

            # ORIGINAL RESOLUTION
            up_flow3 = F.interpolate(input=flow3, size=(int(h_original / 8.0), int(w_original / 8.0)), mode='bilinear',
                                     align_corners=False)
        else:
            # ORIGINAL RESOLUTION
            up_flow3 = F.interpolate(input=flow3, size=(int(h_original / 8.0), int(w_original / 8.0)), mode='bilinear',
                                     align_corners=False)
            up_flow3[:, 0, :, :] *= float(w_original) / float(256)
            up_flow3[:, 1, :, :] *= float(h_original) / float(256)
            # ==> put the upflow in the range [Horiginal x Woriginal]

        # ORIGINAL RESOLUTION
        # level 1/8 of original resolution
        ratio = 1.0/8.0
        warp2 = self.warp(c22, up_flow3*div*ratio)
        corr2 = correlation.FunctionCorrelation(reference_features=c12, query_features=warp2)
        corr2 = self.leakyRELU(corr2)
        corr2 = torch.cat((corr2, up_flow3), 1)
        x, res_flow2 = self.decoder2(corr2)
        flow2 = res_flow2 + up_flow3
        up_flow2 = self.deconv2(flow2)
        up_feat2 = self.upfeat2(x)

        # level 1/4 of original resolution
        ratio = 1.0 / 4.0
        warp1 = self.warp(c21, up_flow2*div*ratio)
        corr1 = correlation.FunctionCorrelation(reference_features=c11, query_features=warp1)
        corr1 = self.leakyRELU(corr1)
        corr1 = torch.cat((corr1, up_flow2, up_feat2), 1)
        x, res_flow1 = self.decoder1(corr1)
        flow1 = res_flow1 + up_flow2
        x = self.l_dc_conv4(self.l_dc_conv3(self.l_dc_conv2(self.l_dc_conv1(x))))
        flow1 = flow1 + self.l_dc_conv7(self.l_dc_conv6(self.l_dc_conv5(x)))

        output = {'flow_estimates': [flow2, flow1]}
        output_256 = {'flow_estimates': [flow4, flow3]}
        return output_256, output

    def flipping_condition(self, im_source_base, im_target_base, device):

        condition = 'min_avg_flow'

        # should only happen during evaluation
        list_average_flow = []
        false_true = [False, True]
        for apply_flipping in false_true:
            im_source, im_target, im_source_256, im_target_256, ratio_x, ratio_y = \
                self.pre_process_data(im_source_base, im_target_base, apply_flip=apply_flipping)
            b, _, h_256, w_256 = im_target_256.size()

            with torch.no_grad():
                # pyramid, 256 reso
                im1_pyr_256 = self.pyramid(im_target_256, additional_coarse_level=True)
                im2_pyr_256 = self.pyramid(im_source_256, additional_coarse_level=True)
                c14 = im1_pyr_256[-2]
                c24 = im2_pyr_256[-2]
                c15 = im1_pyr_256[-1]
                c25 = im2_pyr_256[-1]
                c24_concat = torch.cat(
                    (c24, F.interpolate(input=c25, size=(16, 16), mode='bilinear', align_corners=False)), 1)
                c14_concat = torch.cat(
                    (c14, F.interpolate(input=c15, size=(16, 16), mode='bilinear', align_corners=False)), 1)

            corr4, flow4 = self.coarsest_resolution_flow(c14_concat, c24_concat, h_256, w_256)
            if condition == 'min_avg_flow':
                average_flow = torch.mean(torch.abs(flow4[:, 0, :, :]), dim=(1, 2))
            else:
                value, indices = torch.max(corr4[0].view(16 * 16, 16 * 16), dim=(1))
                average_flow = value.sum()
            list_average_flow.append(average_flow.item())
        if condition == 'min_avg_flow':
            target_image_is_flipped = false_true[np.argmin(np.float32(list_average_flow))]
        else:
            target_image_is_flipped = false_true[np.argmax(np.float32(list_average_flow))]
        if target_image_is_flipped:
            list_average_flow = []
            # if previous way found that target is flipped with respect to the source ==> check that the
            # other way finds the same thing
            # ==> the source becomes the target and the target becomes source
            for apply_flipping in false_true:
                im_source, im_target, im_source_256, im_target_256, ratio_x, ratio_y = \
                    self.pre_process_data(im_target_base, im_source_base, apply_flip=apply_flipping)
                b, _, h_256, w_256 = im_target_256.size()

                with torch.no_grad():
                    # pyramid, 256 reso
                    im1_pyr_256 = self.pyramid(im_target_256, additional_coarse_level=True)
                    im2_pyr_256 = self.pyramid(im_source_256, additional_coarse_level=True)
                    c14 = im1_pyr_256[-2]
                    c24 = im2_pyr_256[-2]
                    c15 = im1_pyr_256[-1]
                    c25 = im2_pyr_256[-1]
                    c24_concat = torch.cat(
                        (c24, F.interpolate(input=c25, size=(16, 16), mode='bilinear', align_corners=False)), 1)
                    c14_concat = torch.cat(
                        (c14, F.interpolate(input=c15, size=(16, 16), mode='bilinear', align_corners=False)), 1)

                corr4, flow4 = self.coarsest_resolution_flow(c14_concat, c24_concat, h_256, w_256)
                if condition == 'min_avg_flow':
                    average_flow = torch.mean(torch.abs(flow4[:, 0, :, :]), dim=(1, 2))
                else:
                    value, indices = torch.max(corr4[0].view(16 * 16, 16 * 16), dim=(1))
                    average_flow = value.sum()
                list_average_flow.append(average_flow.item())
            if condition == 'min_avg_flow':
                target_image_is_flipped = false_true[np.argmin(np.float32(list_average_flow))]
            else:
                target_image_is_flipped = false_true[np.argmax(np.float32(list_average_flow))]

        self.target_image_is_flipped = target_image_is_flipped
        im_source, im_target, im_source_256, im_target_256, ratio_x, ratio_y = \
            self.pre_process_data(im_source_base, im_target_base, apply_flip=target_image_is_flipped)

        return im_source.to(device).contiguous(), im_target.to(device).contiguous(), \
               im_source_256.to(device).contiguous(), im_target_256.to(device).contiguous(), \
               ratio_x, ratio_y

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
        w_scale = target_img.shape[3]
        h_scale = target_img.shape[2]
        # define output_shape
        if output_shape is None and scaling != 1.0:
            output_shape = (int(h_scale*scaling), int(w_scale*scaling))
        source_img, target_img, source_img_256, target_img_256, ratio_x, ratio_y \
            = self.pre_process_data(source_img, target_img)

        output_256, output = self.forward(target_img, source_img, target_img_256, source_img_256)

        flow_est_list = output['flow_estimates']
        flow_est = flow_est_list[-1]

        if output_shape is not None:
            ratio_x *= float(output_shape[1]) / float(w_scale)
            ratio_y *= float(output_shape[0]) / float(h_scale)
        else:
            output_shape = (h_scale, w_scale)
        flow_est = torch.nn.functional.interpolate(input=flow_est, size=output_shape, mode='bilinear',
                                                   align_corners=False)

        flow_est[:, 0, :, :] *= ratio_x
        flow_est[:, 1, :, :] *= ratio_y

        # compute flow in opposite direction
        output_256_backward, output_backward = self.forward(source_img, target_img, source_img_256, target_img_256)
        flow_est_backward = output_backward['flow_estimates'][-1]

        flow_est_backward = torch.nn.functional.interpolate(input=flow_est_backward, size=output_shape, mode='bilinear',
                                                            align_corners=False)
        flow_est_backward[:, 0, :, :] *= ratio_x
        flow_est_backward[:, 1, :, :] *= ratio_y

        cyclic_consistency_error = torch.norm(flow_est + self.warp(flow_est_backward, flow_est), dim=1, p=2,
                                              keepdim=True)
        uncertainty_est = {'cyclic_consistency_error': cyclic_consistency_error,
                           'inv_cyclic_consistency_error': 1.0 / (1.0 + cyclic_consistency_error)}

        if mode == 'channel_first':
            return flow_est, uncertainty_est
        else:
            return flow_est.permute(0, 2, 3, 1), uncertainty_est

    def estimate_flow_and_confidence_map_with_flipping_condition(self, source_img, target_img, output_shape=None,
                                                                 scaling=1.0, mode='channel_first'):
        """
        Returns the flow field and corresponding confidence map relating the target to the source image.
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
        flow_est = self.estimate_flow_with_flipping_condition(source_img, target_img, output_shape=output_shape,
                                                              scaling=scaling)
        flow_est_backward = self.estimate_flow_with_flipping_condition(target_img, source_img,
                                                                       output_shape=output_shape, scaling=scaling)

        cyclic_consistency_error = torch.norm(flow_est + self.warp(flow_est_backward, flow_est), dim=1, p=2,
                                              keepdim=True)
        uncertainty_est = {'cyclic_consistency_error': cyclic_consistency_error,
                           'inv_cyclic_consistency_error': 1.0 / (1.0 + cyclic_consistency_error)}

        if mode == 'channel_first':
            return flow_est, uncertainty_est
        else:
            return flow_est.permute(0, 2, 3, 1), uncertainty_est

    def get_matches_and_confidence(self, source_img, target_img, scaling=1.0/4.0,
                                   confident_mask_type='cyclic_consistency_error_below_3', min_number_of_pts=200):
        """
        Computes matches and corresponding confidence value.
        Confidence value is obtained with forward-backward cyclic consistency.
        Args:
            source_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            target_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            scaling: float, scaling factor applied to target image shape, to obtain the outputted flow field dimensions,
                     where the matches are extracted
            confident_mask_type: default is 'proba_interval_1_above_10' for PDCNet.
                                 See inference_utils/estimate_mask for more details
            min_number_of_pts: below that number, we discard the retrieved matches (little blobs in cyclic
                               consistency mask)

        Returns:
            dict with keys 'kp_source', 'kp_target', 'confidence_value', 'flow' and 'mask'
            flow and mask are torch tensors

        """
        flow_estimated, uncertainty_est = self.estimate_flow_and_confidence_map(source_img, target_img, scaling=scaling)

        mask = estimate_mask(confident_mask_type, uncertainty_est, list_item=-1)
        mapping_estimated = convert_flow_to_mapping(flow_estimated)
        # remove point that lead to outside the source image
        mask = mask & mapping_estimated[:, 0].ge(0) & mapping_estimated[:, 1].ge(0) & \
            mapping_estimated[:, 0].le(source_img.shape[-1] // scaling - 1) & \
            mapping_estimated[:, 1].le(source_img.shape[-2] // scaling - 1)

        # get corresponding keypoints
        scaling_kp = np.float32(target_img.shape[-2:]) / np.float32(flow_estimated.shape[-2:])  # h, w

        mkpts_s, mkpts_t = matches_from_flow(flow_estimated, mask, scaling=scaling_kp[::-1])

        # between 0 and 1
        confidence_values = uncertainty_est['inv_cyclic_consistency_error'].squeeze()[mask.squeeze()].cpu().numpy()
        sort_index = np.argsort(np.array(confidence_values)).tolist()[::-1]  # from highest to smallest
        confidence_values = np.array(confidence_values)[sort_index]

        mkpts_s = np.array(mkpts_s)[sort_index]
        mkpts_t = np.array(mkpts_t)[sort_index]

        if len(mkpts_s) < min_number_of_pts:
            mkpts_s = np.empty([0, 2], dtype=np.float32)
            mkpts_t = np.empty([0, 2], dtype=np.float32)
            confidence_values = np.empty([0], dtype=np.float32)

        pred = {'kp_source': mkpts_s, 'kp_target': mkpts_t, 'confidence_value': confidence_values,
                'flow': self.resize_and_rescale_flow(flow_estimated, target_img.shape[-2:]),
                'mask': F.interpolate(input=mask.unsqueeze(1).float(), size=target_img.shape[-2:], mode='bilinear',
                                      align_corners=False).squeeze(1)}
        return pred
