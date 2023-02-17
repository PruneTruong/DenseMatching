import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from admin.model_constructor import model_constructor
from models.modules.mod import deconv, unnormalise_and_convert_mapping_to_flow
from models.base_matching_net import BaseGLUMultiScaleMatchingNet, set_glunet_parameters
from models.feature_backbones.VGG_features import VGGPyramid
from models.modules.local_correlation import correlation
from models.inference_utils import matches_from_flow, estimate_mask
from models.modules.bilinear_deconv import BilinearConvTranspose2d
from utils_flow.flow_and_mapping_operations import convert_flow_to_mapping


class GLUNetModel(BaseGLUMultiScaleMatchingNet):
    """
    GLU-Net model.
    The flows (flow2, flow1) are predicted such that they are scaled to the input image resolution. To obtain the flow
    from target to source at original resolution, one just needs to bilinearly upsample (without further scaling).
    """
    def __init__(self, iterative_refinement=False, scale_low_resolution=False,
                 use_interp_instead_of_deconv=False, init_deconv_w_bilinear=True,
                 global_corr_type='feature_corr_layer', global_gocor_arguments=None, normalize='relu_l2norm',
                 normalize_features=True, cyclic_consistency=False,
                 local_corr_type='feature_corr_layer', local_gocor_arguments=None, same_local_corr_at_all_levels=True,
                 local_decoder_type='OpticalFlowEstimator', global_decoder_type='CMDTop',
                 decoder_inputs='corr_flow_feat', refinement_at_adaptive_reso=True, refinement_at_all_levels=False,
                 refinement_at_finest_level=True, apply_refinement_finest_resolution=True,
                 give_flow_to_refinement_module=False, pyramid_type='VGG', md=4, upfeat_channels=2,
                 train_features=False, make_two_feature_copies=False):

        params = set_glunet_parameters(global_corr_type=global_corr_type, gocor_global_arguments=global_gocor_arguments,
                                       normalize=normalize, normalize_features=normalize_features,
                                       cyclic_consistency=cyclic_consistency, md=md,
                                       local_corr_type=local_corr_type, gocor_local_arguments=local_gocor_arguments,
                                       same_local_corr_at_all_levels=same_local_corr_at_all_levels,
                                       local_decoder_type=local_decoder_type, global_decoder_type=global_decoder_type,
                                       decoder_inputs=decoder_inputs,
                                       give_flow_to_refinement_module=give_flow_to_refinement_module,
                                       refinement_at_adaptive_reso=refinement_at_adaptive_reso,
                                       refinement_at_all_levels=refinement_at_all_levels,
                                       refinement_at_finest_level=refinement_at_finest_level,
                                       apply_refinement_finest_resolution=apply_refinement_finest_resolution,
                                       nbr_upfeat_channels=upfeat_channels,
                                       make_two_feature_copies=make_two_feature_copies)
        super().__init__(params)
        self.iterative_refinement = iterative_refinement

        # if you want all levels to be scaled for the high resolution images
        self.scale_low_resolution = scale_low_resolution

        # interpolation is actually better than using deconv.
        self.use_interp_instead_of_deconv = use_interp_instead_of_deconv

        # level 4, 16x16
        nd = 16*16  # global correlation
        od = nd + 2
        # here, I follow DGC-Net global correlation module + decoder
        decoder4, num_channels_last_conv = self.initialize_mapping_decoder(self.params.global_decoder_type, in_channels=od,
                                                                           batch_norm=self.params.batch_norm)
        self.decoder4 = decoder4
        if not self.use_interp_instead_of_deconv:
            if init_deconv_w_bilinear:
                # initialize the deconv to bilinear weights speeds up the training significantly
                self.deconv4 = BilinearConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
            else:
                self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        # level 3, 32x32
        nd = (2*self.params.md+1)**2  # constrained correlation, 4 pixels on each side
        # for local correlations, I follow PWC-Net (not normalized features, local correlation, leaky relu and decoder)
        decoder3, num_channels_last_conv = self.initialize_flow_decoder(decoder_type=self.params.local_decoder_type,
                                                                        decoder_inputs=self.params.decoder_inputs,
                                                                        nbr_upfeat_channels=0,
                                                                        in_channels_corr=nd)
        self.decoder3 = decoder3
        input_to_refinement_3 = num_channels_last_conv

        if self.params.give_flow_to_refinement_module:
            input_to_refinement_3 += 2
        # weights for refinement module
        if self.params.refinement_at_adaptive_reso:
            self.initialize_adaptive_reso_refinement_module(input_to_refinement_3, self.params.batch_norm)

        # level 2, 1/8 of original resolution
        nd = (2*self.params.md+1)**2  # constrained correlation, 4 pixels on each side
        decoder2, num_channels_last_conv = self.initialize_flow_decoder(decoder_type=self.params.local_decoder_type,
                                                                        decoder_inputs=self.params.decoder_inputs,
                                                                        nbr_upfeat_channels=0,
                                                                        in_channels_corr=nd)
        self.decoder2 = decoder2
        input_to_refinement_2 = num_channels_last_conv

        if 'feat' in self.params.decoder_inputs:
            # same upfeat than in PWCNet
            self.upfeat2 = deconv(input_to_refinement_2, self.params.nbr_upfeat_channels, kernel_size=4,
                                  stride=2, padding=1)

        if not self.use_interp_instead_of_deconv:
            if init_deconv_w_bilinear:
                # initialize the deconv to bilinear weights speeds up the training significantly
                self.deconv2 = BilinearConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
            else:
                self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        # level 1, 1/4 of original resolution
        nd = (2*self.params.md+1)**2  # constrained correlation, 4 pixels on each side
        decoder1, num_channels_last_conv = self.initialize_flow_decoder(decoder_type=self.params.local_decoder_type,
                                                                        decoder_inputs=self.params.decoder_inputs,
                                                                        in_channels_corr=nd,
                                                                        nbr_upfeat_channels=self.params.nbr_upfeat_channels)
        self.decoder1 = decoder1
        input_to_refinement_1 = num_channels_last_conv
        if self.params.give_flow_to_refinement_module:
            input_to_refinement_1 += 2
        # weights of the final refinement (context network)
        if self.params.refinement_at_finest_level:
            self.initialize_last_level_refinement_module(input_to_refinement_1, self.params.batch_norm)

        # initialize modules
        for m in self.modules():
            # deconv initialized in its own module
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

        # initialize the global and local correlation modules
        # comprises GOCor or feature correlation layer
        self.initialize_global_corr()
        self.initialize_local_corr()

        # features back-bone extractor
        if pyramid_type == 'VGG':
            if make_two_feature_copies:
                self.pyramid_256 = VGGPyramid(train=train_features)
            feature_extractor = VGGPyramid(train=train_features)
        else:
            raise NotImplementedError('The feature extractor that you selected in not implemented: {}'
                                      .format(pyramid_type))
        self.pyramid = feature_extractor

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
        # im1 is target image, im2 is source image
        b, _, h_original, w_original = im_target.size()
        b, _, h_256, w_256 = im_target_256.size()  # fixed size of 256x256
        div = 1.0

        c14, c24, c13, c23, c12, c22, c11, c21 = self.extract_features(im_target, im_source, im_target_256,
                                                                       im_source_256, im_target_pyr,
                                                                       im_source_pyr, im_target_pyr_256,
                                                                       im_source_pyr_256)
        # RESOLUTION 256x256
        # level 4: 16x16
        # here same procedure then DGC-Net global mapping decoder.
        ratio_x = 16.0 / float(w_256)
        ratio_y = 16.0 / float(h_256)

        corr4 = self.get_global_correlation(c14, c24)

        b, c, h, w = corr4.size()
        if torch.cuda.is_available():
            init_map = torch.FloatTensor(b, 2, h, w).zero_().cuda()
        else:
            init_map = torch.FloatTensor(b, 2, h, w).zero_()
        # init_map is fed to the decoder to be consistent with decoder of DGC-Net (but not particularly needed)
        est_map4 = self.decoder4(x1=corr4, x3=init_map)
        # conversion to flow and from there constrained correlation
        flow4 = unnormalise_and_convert_mapping_to_flow(est_map4)
        # we want flow4 to be scaled for h_256xw_256, so we multiply by these ratios.
        flow4[:, 0, :, :] /= ratio_x
        flow4[:, 1, :, :] /= ratio_y

        if self.use_interp_instead_of_deconv:
            up_flow4 = F.interpolate(input=flow4, size=(32, 32), mode='bilinear', align_corners=False)
        else:
            up_flow4 = self.deconv4(flow4)

        # level 3: 32x32
        ratio_x = 32.0 / float(w_256)
        ratio_y = 32.0 / float(h_256)
        # the flow needs to be scaled for the current resolution in order to warp.
        # otherwise flow4 and up_flow4 are scaled for h_256xw_256.
        up_flow_4_warping = up_flow4 * div
        up_flow_4_warping[:, 0, :, :] *= ratio_x
        up_flow_4_warping[:, 1, :, :] *= ratio_y
        warp3 = self.warp(c23, up_flow_4_warping)

        # constrained correlation now
        # we follow the same procedure than PWCNet (features not normalized, lcoal corr and then leaky relu)
        if 'GOCor' in self.params.local_corr_type:
            if self.params.same_local_corr_at_all_levels:
                corr3 = self.local_corr(c13, warp3)
            else:

                corr3 = self.local_corr_3(c13, warp3)
        else:
            # feature correlation layer
            corr3 = correlation.FunctionCorrelation(c13, warp3)
        corr3 = self.leakyRELU(corr3)

        if self.params.decoder_inputs == 'corr_flow_feat':
            corr3 = torch.cat((corr3, up_flow4), 1)
        elif self.params.decoder_inputs == 'corr':
            corr3 = corr3
        elif self.params.decoder_inputs == 'corr_flow':
            corr3 = torch.cat((corr3, up_flow4), 1)
        else:
            raise ValueError('Wrong input decoder, you chose {}'.format(self.params.decoder_inputs))
        x3, res_flow3 = self.decoder3(corr3)

        # PWC-Net refinement context module
        if self.params.refinement_at_adaptive_reso:
            if self.params.give_flow_to_refinement_module:
                input_refinement = res_flow3 + up_flow4
                x3 = torch.cat((x3, input_refinement), 1)
            x_, res_flow3_ = self.PWCNetRefinementAdaptiveReso(x3)
            res_flow3 = res_flow3 + res_flow3_

        flow3 = res_flow3 + up_flow4

        if self.iterative_refinement:
            # from 32x32 resolution, if upsampling to 1/8*original resolution is too big,
            # do iterative upsampling so that gap is always smaller than 2.
            R_w = float(w_original)/8.0/32.0
            R_h = float(w_original)/8.0/32.0
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
                    c23_bis = torch.nn.functional.interpolate(c22, size=(int(h_original * ratio),
                                                                         int(w_original * ratio)), mode='area')
                    c13_bis = torch.nn.functional.interpolate(c12, size=(int(h_original * ratio),
                                                                         int(w_original * ratio)), mode='area')
                    warp3 = self.warp(c23_bis, up_flow3 * div * ratio)

                    if 'GOCor' in self.params.local_corr_type:
                        if self.params.same_local_corr_at_all_levels:
                            corr3 = self.local_corr(c13_bis, warp3)
                        else:

                            corr3 = self.local_corr_3(c13_bis, warp3)
                    else:
                        # feature correlation layer
                        corr3 = correlation.FunctionCorrelation(c13_bis, warp3)
                    corr3 = self.leakyRELU(corr3)

                    if self.params.decoder_inputs == 'corr_flow_feat':
                        corr3 = torch.cat((corr3, up_flow3), 1)
                    elif self.params.decoder_inputs == 'corr':
                        corr3 = corr3
                    elif self.params.decoder_inputs == 'corr_flow':
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
            # before the flow was scaled to h_256xw_256. Now, since we go to the high resolution images, we need
            # to scale the flow to h_original x w_original
            up_flow3[:, 0, :, :] *= float(w_original) / 256.0
            up_flow3[:, 1, :, :] *= float(h_original) / 256.0
            # ==> put the upflow in the range [Horiginal x Woriginal]

        # level 2 : 1/8 of original resolution
        ratio = 1.0 / 8.0
        warp2 = self.warp(c22, up_flow3*div*ratio)
        if 'GOCor' in self.params.local_corr_type:
            if self.params.same_local_corr_at_all_levels:
                corr2 = self.local_corr(c12, warp2)
            else:

                corr2 = self.local_corr_2(c12, warp2)
        else:
            # feature correlation layer
            corr2 = correlation.FunctionCorrelation(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        if self.params.decoder_inputs == 'corr_flow_feat':
            corr2 = torch.cat((corr2, up_flow3), 1)
        elif self.params.decoder_inputs == 'corr':
            corr2 = corr2
        elif self.params.decoder_inputs == 'corr_flow':
            corr2 = torch.cat((corr2, up_flow3), 1)
        else:
            raise ValueError('Wrong input decoder, you chose {}'.format(self.params.decoder_inputs))

        x2, res_flow2 = self.decoder2(corr2)
        flow2 = res_flow2 + up_flow3
        if self.use_interp_instead_of_deconv:
            up_flow2 = F.interpolate(input=flow2, size=(int(h_original / 4.0), int(w_original / 4.0)), mode='bilinear',
                                     align_corners=False)
        else:
            up_flow2 = self.deconv2(flow2)

        if self.params.decoder_inputs == 'corr_flow_feat':
            up_feat2 = self.upfeat2(x2)

        # level 1: 1/4 of original resolution
        ratio = 1.0 / 4.0
        warp1 = self.warp(c21, up_flow2*div*ratio)
        if 'GOCor' in self.params.local_corr_type:
            if self.params.same_local_corr_at_all_levels:
                corr1 = self.local_corr(c11, warp1)
            else:

                corr1 = self.local_corr_1(c11, warp1)
        else:
            # feature correlation layer
            corr1 = correlation.FunctionCorrelation(c11, warp1)
        corr1 = self.leakyRELU(corr1)

        if self.params.decoder_inputs == 'corr_flow_feat':
            corr1 = torch.cat((corr1, up_flow2, up_feat2), 1)
        elif self.params.decoder_inputs == 'corr':
            corr1 = corr1
        elif self.params.decoder_inputs == 'corr_flow':
            corr1 = torch.cat((corr1, up_flow2), 1)
        else:
            raise ValueError('Wrong input decoder, you chose {}'.format(self.params.decoder_inputs))

        x, res_flow1 = self.decoder1(corr1)

        if self.params.refinement_at_finest_level and self.params.apply_refinement_finest_resolution:
            if self.params.give_flow_to_refinement_module:
                input_refinement = res_flow1 + up_flow2
                x = torch.cat((x, input_refinement), 1)
            x_, res_flow1_ = self.PWCNetRefinementFinal(x)
            res_flow1 = res_flow1 + res_flow1_

        flow1 = res_flow1 + up_flow2

        if self.scale_low_resolution:
            # Here, we also want to scale the low resolution flows (flow4 and flow3) to the high resolution
            # original image sizes h_original x w_original
            # prepare output dict
            output_256 = {'flow_estimates': [flow4, flow3], 'correlation': corr4}
            flow4 = flow4.clone()
            flow4[:, 0] *= float(w_original) / float(w_256)
            flow4[:, 1] *= float(h_original) / float(h_256)

            flow3 = flow3.clone()
            flow3[:, 0] *= float(w_original) / float(w_256)
            flow3[:, 1] *= float(h_original) / float(h_256)

            output = {'flow_estimates': [flow4, flow3, flow2, flow1]}
        else:
            # prepare output dict
            output = {'flow_estimates': [flow2, flow1]}
            output_256 = {'flow_estimates': [flow4, flow3], 'correlation': corr4}
        return output_256, output

    # FOR FLIPPING CONDITION
    def coarsest_resolution_flow(self, c14, c24, h_256, w_256):
        ratio_x = 16.0 / float(w_256)
        ratio_y = 16.0 / float(h_256)

        corr4 = self.get_global_correlation(c14, c24)

        b, c, h, w = corr4.size()
        if torch.cuda.is_available():
            init_map = torch.FloatTensor(b, 2, h, w).zero_().cuda()
        else:
            init_map = torch.FloatTensor(b, 2, h, w).zero_()
        est_map4 = self.decoder4(x1=corr4, x3=init_map)
        # conversion to flow and from there constrained correlation
        flow4 = unnormalise_and_convert_mapping_to_flow(est_map4)
        flow4[:, 0, :, :] /= ratio_x
        flow4[:, 1, :, :] /= ratio_y
        return corr4, flow4

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


@model_constructor
def glunet_vgg16(global_corr_type='global_corr', global_gocor_arguments=None, normalize='relu_l2norm',
                 normalize_features=True, cyclic_consistency=False, init_deconv_w_bilinear=True,
                 local_corr_type='local_corr', local_gocor_arguments=None, same_local_corr_at_all_levels=True,
                 local_decoder_type='OpticalFlowEstimator', global_decoder_type='CMDTop',
                 decoder_inputs='corr_flow_feat', refinement_at_adaptive_reso=True, refinement_at_all_levels=False,
                 refinement_at_finest_level=True, apply_refinement_finest_resolution=True,
                 give_flow_to_refinement_module=False, nbr_upfeat_channels=2, train_features=False,
                 iterative_refinement=False, scale_low_resolution=False, use_interp_instead_of_deconv=False):

    net = GLUNetModel(iterative_refinement=iterative_refinement,
                      global_gocor_arguments=global_gocor_arguments, global_corr_type=global_corr_type,
                      normalize=normalize, normalize_features=normalize_features,
                      cyclic_consistency=cyclic_consistency, local_corr_type=local_corr_type,
                      local_gocor_arguments=local_gocor_arguments,
                      same_local_corr_at_all_levels=same_local_corr_at_all_levels,
                      local_decoder_type=local_decoder_type, global_decoder_type=global_decoder_type,
                      decoder_inputs=decoder_inputs,
                      refinement_at_adaptive_reso=refinement_at_adaptive_reso,
                      refinement_at_all_levels=refinement_at_all_levels,
                      refinement_at_finest_level=refinement_at_finest_level,
                      apply_refinement_finest_resolution=apply_refinement_finest_resolution,
                      give_flow_to_refinement_module=give_flow_to_refinement_module, pyramid_type='VGG', md=4,
                      upfeat_channels=nbr_upfeat_channels,
                      train_features=train_features, scale_low_resolution=scale_low_resolution,
                      use_interp_instead_of_deconv=use_interp_instead_of_deconv, init_deconv_w_bilinear=init_deconv_w_bilinear)
    return net