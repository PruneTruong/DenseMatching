"""
Adaptation of the implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018

Jinwei Gu and Zhile Ren

"""

import torch
import torch.nn as nn
import numpy as np
from ..modules.local_correlation import correlation
from .base_net import BasePWCNet, conv, deconv, predict_flow
from third_party.GOCor.GOCor import local_gocor
from third_party.GOCor.GOCor.optimizer_selection_functions import define_optimizer_local_corr



class PWCNet_model(BasePWCNet):
    """
    PWC-Net model
    """
    def __init__(self, div=20.0, refinement=True, batch_norm=False, md=4,
                 local_corr_type='local_corr', local_gocor_arguments=None, same_local_corr_at_all_levels=True):
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
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat6 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + nbr_features[1] + 4
        self.conv5_0 = conv(od, 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv5_1 = conv(od + dd[0], 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv5_2 = conv(od + dd[1], 96, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv5_3 = conv(od + dd[2], 64, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv5_4 = conv(od + dd[3], 32, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.predict_flow5 = predict_flow(od + dd[4])
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat5 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        
        od = nd + nbr_features[2] + 4
        self.conv4_0 = conv(od,      128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1, batch_norm=batch_norm)
        self.predict_flow4 = predict_flow(od+dd[4]) 
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat4 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd + nbr_features[3] + 4
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv3_1 = conv(od+dd[0], 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv3_2 = conv(od+dd[1], 96,  kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv3_3 = conv(od+dd[2], 64,  kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv3_4 = conv(od+dd[3], 32,  kernel_size=3, stride=1, batch_norm=batch_norm)
        self.predict_flow3 = predict_flow(od+dd[4]) 
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
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
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
        # im1 is reference image and im2 is query image

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



