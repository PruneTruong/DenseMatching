"""
Adaptation of the implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018

Jinwei Gu and Zhile Ren

"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from packaging import version

from third_party.GOCor.GOCor import local_gocor
from third_party.GOCor.GOCor.optimizer_selection_functions import define_optimizer_local_corr
from models.modules.local_correlation import correlation
from models.modules.bilinear_deconv import BilinearConvTranspose2d
from models.base_matching_net import BaseMultiScaleMatchingNet
from models.inference_utils import matches_from_flow, estimate_mask
from utils_flow.flow_and_mapping_operations import convert_flow_to_mapping


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False):
    if batch_norm:
        return nn.Sequential(
                            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, bias=True),
                            nn.BatchNorm2d(out_planes),
                            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(
                            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, bias=True),
                            nn.LeakyReLU(0.1))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    deconv_ = nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

    nn.init.kaiming_normal_(deconv_.weight.data, mode='fan_in')
    if deconv_.bias is not None:
        deconv_.bias.data.zero_()
    return deconv_


class PWCNetModel(BaseMultiScaleMatchingNet):
    """
    PWC-Net model
    """
    def __init__(self, div=20.0, refinement=True, batch_norm=False, md=4, init_deconv_w_bilinear=False,
                 local_corr_type='local_corr', local_gocor_arguments=None, same_local_corr_at_all_levels=True):
        # the original PWCNet model has the deconv initialized to random.
        super().__init__()
        nbr_features = [196, 128, 96, 64, 32, 16, 3]
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.div = div
        self.refinement = refinement

        nd = (2*md+1)**2
        dd = np.cumsum([128, 128, 96, 64, 32])
        od = nd
        self.conv6_0 = conv(od, 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv6_1 = conv(od + dd[0], 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv6_2 = conv(od + dd[1], 96, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv6_3 = conv(od + dd[2], 64, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv6_4 = conv(od + dd[3], 32, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.predict_flow6 = predict_flow(od + dd[4])
        if init_deconv_w_bilinear:
            # initialize the deconv to bilinear weights speeds up the training significantly
            self.deconv6 = BilinearConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
        else:
            self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat6 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + nbr_features[1] + 4
        self.conv5_0 = conv(od, 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv5_1 = conv(od + dd[0], 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv5_2 = conv(od + dd[1], 96, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv5_3 = conv(od + dd[2], 64, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv5_4 = conv(od + dd[3], 32, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.predict_flow5 = predict_flow(od + dd[4])
        if init_deconv_w_bilinear:
            # initialize the deconv to bilinear weights speeds up the training significantly
            self.deconv5 = BilinearConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
        else:
            self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat5 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        
        od = nd + nbr_features[2] + 4
        self.conv4_0 = conv(od,      128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1, batch_norm=batch_norm)
        self.predict_flow4 = predict_flow(od+dd[4])
        if init_deconv_w_bilinear:
            # initialize the deconv to bilinear weights speeds up the training significantly
            self.deconv4 = BilinearConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
        else:
            self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd + nbr_features[3] + 4
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv3_1 = conv(od+dd[0], 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv3_2 = conv(od+dd[1], 96,  kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv3_3 = conv(od+dd[2], 64,  kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv3_4 = conv(od+dd[3], 32,  kernel_size=3, stride=1, batch_norm=batch_norm)
        self.predict_flow3 = predict_flow(od+dd[4])
        if init_deconv_w_bilinear:
            # initialize the deconv to bilinear weights speeds up the training significantly
            self.deconv3 = BilinearConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
        else:
            self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd + nbr_features[4] + 4
        self.conv2_0 = conv(od,      128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv2_1 = conv(od+dd[0], 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv2_2 = conv(od+dd[1], 96,  kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv2_3 = conv(od+dd[2], 64,  kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv2_4 = conv(od+dd[3], 32,  kernel_size=3, stride=1, batch_norm=batch_norm)
        self.predict_flow2 = predict_flow(od+dd[4])
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        # weights for refinement module
        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1, batch_norm=batch_norm)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2, batch_norm=batch_norm)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4, batch_norm=batch_norm)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8, batch_norm=batch_norm)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1, batch_norm=batch_norm)
        self.dc_conv7 = predict_flow(32)

        # pyramid level weights
        self.conv1a = conv(3, 16, kernel_size=3, stride=2, batch_norm=batch_norm)
        self.conv1aa = conv(16, 16, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv1b = conv(16, 16, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv2a = conv(16, 32, kernel_size=3, stride=2, batch_norm=batch_norm)
        self.conv2aa = conv(32, 32, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv2b = conv(32, 32, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv3a = conv(32, 64, kernel_size=3, stride=2, batch_norm=batch_norm)
        self.conv3aa = conv(64, 64, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv3b = conv(64, 64, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv4a = conv(64, 96, kernel_size=3, stride=2, batch_norm=batch_norm)
        self.conv4aa = conv(96, 96, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv4b = conv(96, 96, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv5a = conv(96, 128, kernel_size=3, stride=2, batch_norm=batch_norm)
        self.conv5aa = conv(128, 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv5b = conv(128, 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv6aa = conv(128, 196, kernel_size=3, stride=2, batch_norm=batch_norm)
        self.conv6a = conv(196, 196, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv6b = conv(196, 196, kernel_size=3, stride=1, batch_norm=batch_norm)

        # initialise the network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

        # local correlation, GOCor or default feature correlation layer
        self.local_corr_type = local_corr_type
        if self.local_corr_type == 'LocalGOCor':
            self.same_local_corr_at_all_levels = same_local_corr_at_all_levels
            if self.same_local_corr_at_all_levels:
                initializer = local_gocor.LocalCorrSimpleInitializer()
                optimizer = define_optimizer_local_corr(local_gocor_arguments)
                self.local_corr = local_gocor.LocalGOCor(filter_initializer=initializer, filter_optimizer=optimizer)
            else:
                initializer_6 = local_gocor.LocalCorrSimpleInitializer()
                optimizer_6 = define_optimizer_local_corr(local_gocor_arguments)
                self.local_corr_6 = local_gocor.LocalGOCor(filter_initializer=initializer_6, filter_optimizer=optimizer_6)

                initializer_5 = local_gocor.LocalCorrSimpleInitializer()
                optimizer_5 = define_optimizer_local_corr(local_gocor_arguments)
                self.local_corr_5 = local_gocor.LocalGOCor(filter_initializer=initializer_5, filter_optimizer=optimizer_5)

                initializer_4 = local_gocor.LocalCorrSimpleInitializer()
                optimizer_4 = define_optimizer_local_corr(local_gocor_arguments)
                self.local_corr_4 = local_gocor.LocalGOCor(filter_initializer=initializer_4, filter_optimizer=optimizer_4)

                initializer_3 = local_gocor.LocalCorrSimpleInitializer()
                optimizer_3 = define_optimizer_local_corr(local_gocor_arguments)
                self.local_corr_3 = local_gocor.LocalGOCor(filter_initializer=initializer_3, filter_optimizer=optimizer_3)

                initializer_2 = local_gocor.LocalCorrSimpleInitializer()
                optimizer_2 = define_optimizer_local_corr(local_gocor_arguments)
                self.local_corr_2 = local_gocor.LocalGOCor(filter_initializer=initializer_2, filter_optimizer=optimizer_2)

    def forward(self, im_reference, im_query):
        # im1 is reference image/target image and im2 is query image/source image

        div = self.div
        # get the different feature pyramid levels
        c11 = self.conv1b(self.conv1aa(self.conv1a(im_reference)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im_query)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))

        # for original image size of 192, here size is (3x3)
        if 'GOCor' in self.local_corr_type:
            if self.same_local_corr_at_all_levels:
                corr6 = self.local_corr(c16, c26)
            else:
                corr6 = self.local_corr_6(c16, c26)
        else:
            corr6 = correlation.FunctionCorrelation(reference_features=c16, query_features=c26)
        corr6 = self.leakyRELU(corr6)

        # decoder 6
        x = torch.cat((self.conv6_0(corr6), corr6),1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        ratio = 1.0/32.0
        warp5 = self.warp(c25, up_flow6*ratio*div)
        if 'GOCor' in self.local_corr_type:
            if self.same_local_corr_at_all_levels:
                corr5 = self.local_corr(c15, warp5)
            else:
                corr5 = self.local_corr_5(c15, warp5)
        else:
            corr5 = correlation.FunctionCorrelation(reference_features=c15, query_features=warp5)
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x), 1)
        x = torch.cat((self.conv5_1(x), x), 1)
        x = torch.cat((self.conv5_2(x), x), 1)
        x = torch.cat((self.conv5_3(x), x), 1)
        x = torch.cat((self.conv5_4(x), x), 1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

        ratio = 1.0/16.0
        warp4 = self.warp(c24, up_flow5*ratio*div)
        if 'GOCor' in self.local_corr_type:
            if self.same_local_corr_at_all_levels:
                corr4 = self.local_corr(c14, warp4)
            else:
                corr4 = self.local_corr_4(c14, warp4)
        else:
            corr4 = correlation.FunctionCorrelation(reference_features=c14, query_features=warp4)
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x), 1)
        x = torch.cat((self.conv4_1(x), x), 1)
        x = torch.cat((self.conv4_2(x), x), 1)
        x = torch.cat((self.conv4_3(x), x), 1)
        x = torch.cat((self.conv4_4(x), x), 1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        ratio = 1.0/8.0
        warp3 = self.warp(c23, up_flow4*ratio*div)
        if 'GOCor' in self.local_corr_type:
            if self.same_local_corr_at_all_levels:
                corr3 = self.local_corr(c13, warp3)
            else:
                corr3 = self.local_corr_3(c13, warp3)
        else:
            corr3 = correlation.FunctionCorrelation(reference_features=c13, query_features=warp3)
        corr3 = self.leakyRELU(corr3)

        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x), 1)
        x = torch.cat((self.conv3_1(x), x), 1)
        x = torch.cat((self.conv3_2(x), x), 1)
        x = torch.cat((self.conv3_3(x), x), 1)
        x = torch.cat((self.conv3_4(x), x), 1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        ratio = 1.0/4.0
        warp2 = self.warp(c22, up_flow3*div*ratio)
        if 'GOCor' in self.local_corr_type:
            if self.same_local_corr_at_all_levels:
                corr2 = self.local_corr(c12, warp2)
            else:
                corr2 = self.local_corr_2(c12, warp2)
        else:
            corr2 = correlation.FunctionCorrelation(reference_features=c12, query_features=warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)
        flow2 = self.predict_flow2(x)

        if self.refinement:
            x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
            flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        return {'flow_estimates': [flow6, flow5, flow4, flow3, flow2]}

    @staticmethod
    def warp(x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo
        # makes a mapping out of the flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W-1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H-1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)

        if version.parse(torch.__version__) >= version.parse("1.3"):
            output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        else:
            output = nn.functional.grid_sample(x, vgrid)

        # the mask makes a difference here
        mask = torch.ones(x.size()).cuda()
        if version.parse(torch.__version__) >= version.parse("1.3"):
            mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)
        else:
            mask = nn.functional.grid_sample(mask, vgrid)
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
        return output * mask

    def pre_process_data(self, source_img, target_img):
        """
        Resizes images so that their size is dividable by 64, then scale values to [0, 1].
        Args:
            source_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            target_img:  torch tensor, bx3xHxW in range [0, 255], not normalized yet

        Returns:
            source_img, target_img, normalized to [0, 1] and put to BGR (according to original PWCNet)
            ratio_x, ratio_y: ratio from original sizes to size dividable by 64.
        """
        b, _, h_scale, w_scale = target_img.shape
        int_preprocessed_width = int(math.floor(math.ceil(w_scale / 64.0) * 64.0))
        int_preprocessed_height = int(math.floor(math.ceil(h_scale / 64.0) * 64.0))

        '''
        source_img = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                     size=(int_preprocessed_height, int_preprocessed_width),
                                                     mode='area').byte()
        target_img = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                     size=(int_preprocessed_height, int_preprocessed_width),
                                                     mode='area').byte()
        source_img = source_img.float().div(255.0)
        target_img = target_img.float().div(255.0)
        '''
        # this gives slightly better values
        source_img_copy = torch.zeros((b, 3, int_preprocessed_height, int_preprocessed_width))
        target_img_copy = torch.zeros((b, 3, int_preprocessed_height, int_preprocessed_width))
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((int_preprocessed_height, int_preprocessed_width),
                                                          interpolation=2),
                                        transforms.ToTensor()])
        # only /255 the tensor
        for i in range(source_img.shape[0]):
            source_img_copy[i] = transform(source_img[i].byte())
            target_img_copy[i] = transform(target_img[i].byte())

        source_img = source_img_copy
        target_img = target_img_copy

        ratio_x = float(w_scale) / float(int_preprocessed_width)
        ratio_y = float(h_scale) / float(int_preprocessed_height)

        # convert to BGR
        return source_img[:, [2, 1, 0]].to(self.device), target_img[:, [2, 1, 0]].to(self.device), ratio_x, ratio_y

    def estimate_flow(self, source_img, target_img, output_shape=None, scaling=1.0, mode='channel_first'):
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
        w_scale = target_img.shape[3]
        h_scale = target_img.shape[2]
        # define output_shape
        if output_shape is None and scaling != 1.0:
            output_shape = (int(h_scale*scaling), int(w_scale*scaling))

        source_img, target_img, ratio_x, ratio_y = self.pre_process_data(source_img, target_img)
        output = self.forward(im_reference=target_img, im_query=source_img)

        flow_est_list = output['flow_estimates']
        flow_est = self.div * flow_est_list[-1]  # original PWC-Net predicted at /20

        if output_shape is not None:
            flow_est = torch.nn.functional.interpolate(input=flow_est, size=(h_scale, w_scale), mode='bilinear',
                                                       align_corners=False)
            ratio_x *= float(output_shape[1]) / w_scale
            ratio_y *= float(output_shape[0]) / h_scale
        else:
            flow_est = torch.nn.functional.interpolate(input=flow_est, size=(h_scale, w_scale), mode='bilinear',
                                                       align_corners=False)

        flow_est[:, 0, :, :] *= ratio_x
        flow_est[:, 1, :, :] *= ratio_y
        if mode == 'channel_first':
            return flow_est
        else:
            return flow_est.permute(0, 2, 3, 1)

    def estimate_flow_and_confidence_map(self, source_img, target_img, output_shape=None,
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
        w_scale = target_img.shape[3]
        h_scale = target_img.shape[2]
        # define output_shape
        if output_shape is None and scaling != 1.0:
            output_shape = (int(h_scale * scaling), int(w_scale * scaling))

        source_img, target_img, ratio_x, ratio_y = self.pre_process_data(source_img, target_img)
        output = self.forward(im_reference=target_img, im_query=source_img)

        flow_est_list = output['flow_estimates']
        flow_est = self.div * flow_est_list[-1]  # original PWC-Net predicted at /20

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
        output_backward = self.forward(source_img, target_img)
        flow_est_backward = self.div * output_backward['flow_estimates'][-1]  # PWC-Net predicted at /20

        flow_est_backward = torch.nn.functional.interpolate(input=flow_est_backward, size=output_shape, mode='bilinear',
                                                            align_corners=False)
        flow_est_backward[:, 0, :, :] *= ratio_x
        flow_est_backward[:, 1, :, :] *= ratio_y

        cyclic_consistency_error = torch.norm(flow_est + self.warp(flow_est_backward, flow_est), dim=1)
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



