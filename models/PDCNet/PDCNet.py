import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from admin.model_constructor import model_constructor
from models.modules.mod import deconv, unnormalise_and_convert_mapping_to_flow
from models.modules.local_correlation import correlation
from models.modules.bilinear_deconv import BilinearConvTranspose2d
from models.modules.feature_correlation_layer import GlobalFeatureCorrelationLayer
from models.feature_backbones.VGG_features import VGGPyramid
from models.PDCNet.mod_uncertainty import MixtureDensityEstimatorFromCorr, MixtureDensityEstimatorFromUncertaintiesAndFlow
from models.base_matching_net import set_glunet_parameters
from models.PDCNet.base_pdcnet import ProbabilisticGLU


class PDCNetModel(ProbabilisticGLU):
    """PDCNet model.
    The flows (flow2, flow1) are predicted such that they are scaled to the input image resolution. To obtain the flow
    from target to source at original resolution, one just needs to bilinearly upsample (without further scaling).
    """
    def __init__(self, global_gocor_arguments=None, global_corr_type='feature_corr_layer', normalize='relu_l2norm',
                 normalize_features=True, cyclic_consistency=False,
                 local_corr_type='feature_corr_layer', local_gocor_arguments=None, same_local_corr_at_all_levels=True,
                 local_decoder_type='OpticalFlowEstimator', global_decoder_type='CMDTop',
                 decoder_inputs='corr_flow_feat', pyramid_type='VGG', md=4, upfeat_channels=2,
                 train_features=False, batch_norm=True, use_interp_instead_of_deconv=False, init_deconv_w_bilinear=True,
                 refinement_at_adaptive_reso=True, refinement_at_all_levels=False,
                 refinement_at_finest_level=True, apply_refinement_finest_resolution=True,
                 corr_for_corr_uncertainty_decoder='corr', scale_low_resolution=False,
                 var_1_minus_plus=1.0, var_2_minus=2.0, var_2_plus=0.0, var_2_plus_256=0.0, var_3_minus_plus=256 ** 2,
                 var_3_minus_plus_256=256 ** 2, estimate_three_modes=False, estimate_one_mode=False, laplace_distr=True,
                 give_layer_before_flow_to_uncertainty_decoder=True, make_two_feature_copies=False):

        params = set_glunet_parameters(global_corr_type=global_corr_type, gocor_global_arguments=global_gocor_arguments,
                                       normalize=normalize, normalize_features=normalize_features,
                                       cyclic_consistency=cyclic_consistency, md=md,
                                       local_corr_type=local_corr_type, gocor_local_arguments=local_gocor_arguments,
                                       same_local_corr_at_all_levels=same_local_corr_at_all_levels,
                                       local_decoder_type=local_decoder_type, global_decoder_type=global_decoder_type,
                                       decoder_inputs=decoder_inputs,
                                       refinement_at_adaptive_reso=refinement_at_adaptive_reso,
                                       refinement_at_all_levels=refinement_at_all_levels,
                                       refinement_at_finest_level=refinement_at_finest_level,
                                       apply_refinement_finest_resolution=apply_refinement_finest_resolution,
                                       nbr_upfeat_channels=upfeat_channels,
                                       make_two_feature_copies=make_two_feature_copies)

        super().__init__(params)
        self.div = 1.0
        self.leakyRELU = nn.LeakyReLU(0.1)

        # if you want all levels to be scaled for the high resolution images
        self.scale_low_resolution = scale_low_resolution

        # interpolation is actually better than using deconv.
        self.use_interp_instead_of_deconv = use_interp_instead_of_deconv

        # variances for the different mixture modes
        self.estimate_one_mode = estimate_one_mode
        self.estimate_three_modes = estimate_three_modes
        assert not (estimate_one_mode and estimate_three_modes), 'ambiguous mode arguments'
        self.laplace_distr = laplace_distr  # will be useful to compute the final confidence p_r
        self.var_1_minus_plus = torch.as_tensor(var_1_minus_plus).float()
        self.var_2_minus = torch.as_tensor(var_2_minus).float()
        self.var_2_plus = torch.as_tensor(var_2_plus).float()
        self.var_2_plus_256 = torch.as_tensor(var_2_plus_256).float()
        self.var_3_minus_plus = torch.as_tensor(var_3_minus_plus).float()
        self.var_3_minus_plus_256 = torch.as_tensor(var_3_minus_plus_256).float()

        self.corr_for_corr_uncertainty_decoder = corr_for_corr_uncertainty_decoder
        if 'gocor' in self.params.global_corr_type.lower() and 'corr' in self.corr_for_corr_uncertainty_decoder:
            self.corr_module_for_corr_uncertainty_decoder = GlobalFeatureCorrelationLayer(shape='3D')
        self.give_layer_before_flow_to_uncertainty_decoder = give_layer_before_flow_to_uncertainty_decoder
        if self.estimate_three_modes:
            uncertainty_output_channels = 4
        elif self.estimate_one_mode:
            uncertainty_output_channels = 1
        else:
            uncertainty_output_channels = 3

        # 16x16
        nd = 16*16  # global correlation
        od = nd + 2
        decoder4, num_channels_last_conv = self.initialize_mapping_decoder(self.params.global_decoder_type,
                                                                           in_channels=od, output_x=True,
                                                                           batch_norm=self.params.batch_norm)
        self.decoder4 = decoder4
        if not self.use_interp_instead_of_deconv:
            if init_deconv_w_bilinear:
                # initialize the deconv to bilinear weights speeds up the training significantly
                self.deconv4 = BilinearConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
            else:
                self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        else:
            self.deconv4 = None

        if 'gocor' in self.params.global_corr_type.lower() and \
                self.corr_for_corr_uncertainty_decoder == 'corr_and_gocor':
            num_ch = 2
        else:
            num_ch = 1
        self.corr_uncertainty_decoder4 = MixtureDensityEstimatorFromCorr(in_channels=num_ch,
                                                                         batch_norm=self.params.batch_norm,
                                                                         search_size=16, output_channels=6,
                                                                         output_all_channels_together=True)

        if self.give_layer_before_flow_to_uncertainty_decoder:
            uncertainty_input_channels = 6 + num_channels_last_conv
        else:
            uncertainty_input_channels = 6 + 2
        self.uncertainty_decoder4 = MixtureDensityEstimatorFromUncertaintiesAndFlow(in_channels=uncertainty_input_channels,
                                                                                    batch_norm=self.params.batch_norm,
                                                                                    output_channels=uncertainty_output_channels)

        # 32x32
        corr_dim = (2*md+1)**2  # constrained correlation, 4 pixels on each side
        if self.estimate_three_modes:
            nd = corr_dim + 6  # adds the uncertainty part
        elif self.estimate_one_mode:
            nd = corr_dim + 1
        else:
            nd = corr_dim + 4
        decoder3, num_channels_last_conv = self.initialize_flow_decoder(decoder_type=self.params.local_decoder_type,
                                                                        decoder_inputs=self.params.decoder_inputs,
                                                                        nbr_upfeat_channels=0,
                                                                        in_channels_corr=nd)
        self.decoder3 = decoder3
        input_to_refinement = num_channels_last_conv + 2
        self.corr_uncertainty_decoder3 = MixtureDensityEstimatorFromCorr(in_channels=num_ch,
                                                                         batch_norm=self.params.batch_norm,
                                                                         search_size=(self.params.md*2+1),
                                                                         output_all_channels_together=True,
                                                                         output_channels=6)

        if self.give_layer_before_flow_to_uncertainty_decoder:
            uncertainty_input_channels = 6 + num_channels_last_conv
        else:
            uncertainty_input_channels = 6 + 2
        if self.estimate_three_modes:
            # 6 channels uncertainty from previous level
            uncertainty_input_channels += 2 + 6
        elif self.estimate_one_mode:
            # 1 channel uncertainty from previous level
            uncertainty_input_channels += 2 + 1
        else:
            # 4 channels uncertainty from previous level
            uncertainty_input_channels += 2 + 4
        self.uncertainty_decoder3 = MixtureDensityEstimatorFromUncertaintiesAndFlow(in_channels=uncertainty_input_channels,
                                                                                    batch_norm=self.params.batch_norm,
                                                                                    output_channels=uncertainty_output_channels)
        # weights for refinement module
        if self.params.refinement_at_all_levels or self.params.refinement_at_adaptive_reso:
            self.initialize_adaptive_reso_refinement_module(input_to_refinement, self.params.batch_norm)

        # 1/8 of original resolution
        corr_dim = (2*md+1)**2  # constrained correlation, 4 pixels on each side
        if self.estimate_three_modes:
            nd = corr_dim + 6  # adds the uncertainty part
        elif self.estimate_one_mode:
            nd = corr_dim + 1
        else:
            nd = corr_dim + 4
        decoder2, num_channels_last_conv = self.initialize_flow_decoder(decoder_type=self.params.local_decoder_type,
                                                                        decoder_inputs=self.params.decoder_inputs,
                                                                        nbr_upfeat_channels=0,
                                                                        in_channels_corr=nd)
        self.decoder2 = decoder2
        input_to_refinement = num_channels_last_conv
        self.corr_uncertainty_decoder2 = MixtureDensityEstimatorFromCorr(in_channels=num_ch,
                                                                         batch_norm=self.params.batch_norm,
                                                                         search_size=(self.params.md * 2 + 1), output_channels=6,
                                                                         output_all_channels_together=True)
        if self.give_layer_before_flow_to_uncertainty_decoder:
            uncertainty_input_channels = 6 + input_to_refinement
        else:
            uncertainty_input_channels = 6 + 2
        if self.estimate_three_modes:
            # 6 channels uncertainty from previous level
            uncertainty_input_channels += 2 + 6
        elif self.estimate_one_mode:
            # 1 channel uncertainty from previous level
            uncertainty_input_channels += 2 + 1
        else:
            # 4 channels uncertainty from previous level
            uncertainty_input_channels += 2 + 4
        self.uncertainty_decoder2 = MixtureDensityEstimatorFromUncertaintiesAndFlow(in_channels=uncertainty_input_channels,
                                                                                    batch_norm=self.params.batch_norm,
                                                                                    output_channels=uncertainty_output_channels)

        if 'feat' in self.params.decoder_inputs:
            self.upfeat2 = deconv(input_to_refinement, self.params.nbr_upfeat_channels, kernel_size=4, stride=2, padding=1)

        if not self.use_interp_instead_of_deconv:
            if init_deconv_w_bilinear:
                # initialize the deconv to bilinear weights speeds up the training significantly
                self.deconv2 = BilinearConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
            else:
                self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        else:
            self.deconv2 = None

        if self.params.refinement_at_all_levels:
            self.initialize_intermediate_level_refinement_module(input_to_refinement, self.params.batch_norm)

        # 1/4 of original resolution
        corr_dim = (2*md+1)**2 # constrained correlation, 4 pixels on each side
        if self.estimate_three_modes:
            nd = corr_dim + 6 # adds the uncertainty part
        elif self.estimate_one_mode:
            nd = corr_dim + 1 # adds the uncertainty part
        else:
            nd = corr_dim + 4
        decoder1, num_channels_last_conv = self.initialize_flow_decoder(decoder_type=self.params.local_decoder_type,
                                                                        decoder_inputs=self.params.decoder_inputs,
                                                                        in_channels_corr=nd,
                                                                        nbr_upfeat_channels=self.params.nbr_upfeat_channels)
        self.decoder1 = decoder1
        input_to_refinement = num_channels_last_conv + 2
        self.corr_uncertainty_decoder1 = MixtureDensityEstimatorFromCorr(in_channels=num_ch,
                                                                         batch_norm=self.params.batch_norm,
                                                                         search_size=(self.params.md * 2 + 1),
                                                                         output_channels=6,
                                                                         output_all_channels_together=True)

        if self.give_layer_before_flow_to_uncertainty_decoder:
            uncertainty_input_channels = 6 + input_to_refinement - 2
        else:
            uncertainty_input_channels = 6 + 2
        if self.estimate_three_modes:
            # 6 channels uncertainty from previous level
            uncertainty_input_channels += 2 + 6
        elif self.estimate_one_mode:
            # 6 channels uncertainty from previous level
            uncertainty_input_channels += 2 + 1
        else:
            # 4 channels uncertainty from previous level
            uncertainty_input_channels += 2 + 4
        self.uncertainty_decoder1 = MixtureDensityEstimatorFromUncertaintiesAndFlow(in_channels=uncertainty_input_channels,
                                                                                    batch_norm=self.params.batch_norm,
                                                                                    output_channels=uncertainty_output_channels)

        # weights of the final refinement (context network)
        if self.params.refinement_at_finest_level:
            self.initialize_last_level_refinement_module(input_to_refinement, batch_norm)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
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

        self.initialize_global_corr()
        self.initialize_local_corr()

        # features
        # feature finetuning
        self.make_two_feature_copies = make_two_feature_copies
        if pyramid_type == 'VGG':
            if make_two_feature_copies:
                self.pyramid_256 = VGGPyramid(train=train_features)
            feature_extractor = VGGPyramid(train=train_features)
        else:
            raise NotImplementedError('The feature extractor that you selected in not implemented: {}'
                                      .format(pyramid_type))
        self.pyramid = feature_extractor

    def get_local_correlation(self, c_target, warp_source):
        if 'GOCor' in self.params.local_corr_type:
            if self.params.same_local_corr_at_all_levels:
                corr = self.local_corr(c_target, warp_source)
            else:
                # different dicor at each levels
                # this is wrong, need to change
                corr = self.local_corr_3(c_target, warp_source)
        else:
            corr = correlation.FunctionCorrelation(reference_features=c_target, query_features=warp_source)
        corr = self.leakyRELU(corr)
        return corr

    def estimate_uncertainty_components(self, corr_uncertainty_module, uncertainty_predictor,
                                        corr_type, corr, c_target, c_source, flow, up_previous_flow=None,
                                        up_previous_uncertainty=None, global_local='global_corr'):
        # corr uncertainty decoder
        x_second_corr = None
        if 'gocor' in corr_type.lower():
            if self.corr_for_corr_uncertainty_decoder == 'gocor':
                input_corr_uncertainty_dec = corr
            elif self.corr_for_corr_uncertainty_decoder == 'corr':
                input_corr_uncertainty_dec = getattr(self, global_local)(c_target, c_source)

            elif self.corr_for_corr_uncertainty_decoder == 'corr_and_gocor':
                input_corr_uncertainty_dec = getattr(self, global_local)(c_target, c_source)
                x_second_corr = corr
            else:
                raise NotImplementedError
        else:
            input_corr_uncertainty_dec = corr

        corr_uncertainty = corr_uncertainty_module(input_corr_uncertainty_dec, x_second_corr=x_second_corr)

        # final uncertainty decoder
        if up_previous_flow is not None and up_previous_uncertainty is not None:
            input_uncertainty = torch.cat((corr_uncertainty, flow,
                                           up_previous_uncertainty, up_previous_flow), 1)
        else:
            input_uncertainty = torch.cat((corr_uncertainty, flow), 1)

        large_log_var_map, weight_map = uncertainty_predictor(input_uncertainty)
        return large_log_var_map, weight_map

    def estimate_at_mappinglevel(self, corr_uncertainty_module, uncertainty_predictor, c14, c24, h_256, w_256):
        # level 4: 16x16
        ratio_x = 16.0 / float(w_256)
        ratio_y = 16.0 / float(h_256)

        corr4 = self.get_global_correlation(c14, c24)
        b, c, h, w = corr4.size()
        if torch.cuda.is_available():
            init_map = torch.FloatTensor(b, 2, h, w).zero_().cuda()
        else:
            init_map = torch.FloatTensor(b, 2, h, w).zero_()

        # flow decoder, estimating flow
        x4, est_map4 = self.decoder4(x1=corr4, x3=init_map)
        flow4 = unnormalise_and_convert_mapping_to_flow(est_map4) / self.div
        flow4[:, 0, :, :] /= ratio_x
        flow4[:, 1, :, :] /= ratio_y

        # uncertainty decoder
        if self.give_layer_before_flow_to_uncertainty_decoder:
            large_log_var_map4, weight_map4 = self.estimate_uncertainty_components(corr_uncertainty_module,
                                                                                   uncertainty_predictor,
                                                                                   self.params.global_corr_type,
                                                                                   corr4, c14, c24, x4,
                                                                                   global_local='use_global_corr_layer')
        else:
            large_log_var_map4, weight_map4 = self.estimate_uncertainty_components(corr_uncertainty_module,
                                                                                   uncertainty_predictor,
                                                                                   self.params.global_corr_type,
                                                                                   corr4, c14, c24, flow4,
                                                                                   global_local='use_global_corr_layer')

        # constrain the large log var map
        large_log_var_map4 = self.constrain_large_log_var_map(self.var_2_minus, self.var_2_plus_256, large_log_var_map4)
        if self.estimate_three_modes:
            # make the other fixed variances
            small_log_var_map4 = torch.ones_like(large_log_var_map4, requires_grad=False) * torch.log(
                self.var_1_minus_plus)
            outlier_log_var_map4 = torch.ones_like(large_log_var_map4, requires_grad=False) * torch.log(
                self.var_3_minus_plus_256)

            log_var_map4 = torch.cat((small_log_var_map4, large_log_var_map4, outlier_log_var_map4), 1)
        elif self.estimate_one_mode:
            log_var_map4 = large_log_var_map4
        else:
            # only 2 modes
            small_log_var_map4 = torch.ones_like(large_log_var_map4, requires_grad=False) * torch.log(self.var_1_minus_plus)
            log_var_map4 = torch.cat((small_log_var_map4, large_log_var_map4), 1)
        return flow4, log_var_map4, weight_map4, corr4

    def estimate_at_flowlevel(self, ratio, c_t, c_s, up_flow, up_uncertainty_components, decoder, PWCNetRefinement,
                              corr_uncertainty_module, uncertainty_predictor, sigma_max,
                              up_feat=None, div=1.0, refinement=False):
        # same ratio in both directions at the end
        up_flow_warping = up_flow * div
        up_flow_warping[:, 0, :, :] *= ratio
        up_flow_warping[:, 1, :, :] *= ratio
        c_s_warped = self.warp(c_s, up_flow_warping)

        corr = self.get_local_correlation(c_t, c_s_warped)
        if self.params.decoder_inputs == 'corr_flow_feat':
            if up_feat is not None:
                input_flow_dec = torch.cat((corr, up_flow, up_feat), 1)
            else:
                input_flow_dec = torch.cat((corr, up_flow), 1)
        elif self.params.decoder_inputs == 'feature':
            input_flow_dec = torch.cat((corr, c_t), 1)
        elif self.params.decoder_inputs == 'flow_and_feat_and_feature':
            input_flow_dec = torch.cat((corr, c_t, up_flow, up_feat), 1)
        elif self.params.decoder_inputs == 'corr_only':
            input_flow_dec = corr
        elif self.params.decoder_inputs == 'flow':
            input_flow_dec = torch.cat((corr, up_flow), 1)
        else:
            raise NotImplementedError

        input_flow_dec = torch.cat((input_flow_dec, up_uncertainty_components), 1)

        x, res_flow = decoder(input_flow_dec)
        x_ = torch.zeros_like(x.detach())

        if refinement:
            input_refinement = res_flow + up_flow
            x_ = torch.cat((x, input_refinement), 1)
            x_, res_flow_ = getattr(self, PWCNetRefinement)(x_)
            res_flow = res_flow + res_flow_

        flow = res_flow + up_flow

        # uncertainty decoder
        if self.give_layer_before_flow_to_uncertainty_decoder:
            large_log_var_map, weight_map = self.estimate_uncertainty_components(corr_uncertainty_module,
                                                                                 uncertainty_predictor,
                                                                                 self.params.local_corr_type, corr,
                                                                                 c_t, c_s_warped, (x_ + x), up_flow,
                                                                                 up_uncertainty_components,
                                                                                 global_local='use_local_corr_layer')
        else:
            large_log_var_map, weight_map = self.estimate_uncertainty_components(corr_uncertainty_module,
                                                                                 uncertainty_predictor,
                                                                                 self.params.local_corr_type, corr,
                                                                                 c_t, c_s_warped, res_flow, up_flow,
                                                                                 up_uncertainty_components,
                                                                                 global_local='use_local_corr_layer')

        # constraint variance
        large_log_var_map = self.constrain_large_log_var_map(self.var_2_minus, sigma_max, large_log_var_map)
        if self.estimate_three_modes:
            # make the other fixed variances
            small_log_var_map = torch.ones_like(large_log_var_map, requires_grad=False) * torch.log(
                self.var_1_minus_plus)
            outlier_log_var_map = torch.ones_like(large_log_var_map, requires_grad=False) * torch.log(
                self.var_3_minus_plus)

            log_var_map = torch.cat((small_log_var_map, large_log_var_map, outlier_log_var_map), 1)
        elif self.estimate_one_mode:
            log_var_map = large_log_var_map
        else:
            # only 2 modes
            small_log_var_map = torch.ones_like(large_log_var_map, requires_grad=False) * torch.log(
                self.var_1_minus_plus)
            log_var_map = torch.cat((small_log_var_map, large_log_var_map), 1)
        return x, flow, log_var_map, weight_map

    def upscaling(self, x, flow, log_var_map, weight_map, output_size, deconv=None, upfeat_layer=None):
        # up scaling
        output_size = [int(x) for x in output_size]

        if deconv is not None:
            up_flow = deconv(flow)
        else:
            up_flow = F.interpolate(input=flow, size=output_size, mode='bilinear', align_corners=False)

        up_feat = None
        if upfeat_layer is not None:
            up_feat = upfeat_layer(x)

        up_probability_map = F.interpolate(input=weight_map, size=output_size, mode='bilinear', align_corners=False)

        if self.estimate_three_modes:
            up_large_log_var_map = F.interpolate(input=log_var_map[:, 1].unsqueeze(1), size=output_size,
                                                 mode='bilinear', align_corners=False)
            up_small_log_var_map = torch.ones_like(up_large_log_var_map, requires_grad=False) * torch.log(
                self.var_1_minus_plus)
            up_outlier_log_var_map = torch.ones_like(up_large_log_var_map, requires_grad=False) * torch.log(
                self.var_3_minus_plus)
            up_log_var_map = torch.cat((up_small_log_var_map, up_large_log_var_map, up_outlier_log_var_map), 1)
        elif self.estimate_one_mode:
            up_large_log_var_map = F.interpolate(input=log_var_map, size=output_size,
                                                 mode='bilinear', align_corners=False)
            up_log_var_map = up_large_log_var_map
        else:
            up_large_log_var_map = F.interpolate(input=log_var_map[:, 1].unsqueeze(1), size=output_size,
                                                 mode='bilinear', align_corners=False)
            up_small_log_var_map = torch.ones_like(up_large_log_var_map, requires_grad=False) * torch.log(
                self.var_1_minus_plus)
            up_log_var_map = torch.cat((up_small_log_var_map, up_large_log_var_map), 1)

        return up_flow, up_log_var_map, up_probability_map, up_feat

    def forward(self, im_target, im_source, im_target_256, im_source_256, im_target_pyr=None,
                im_source_pyr=None, im_target_pyr_256=None, im_source_pyr_256=None):
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
            output_256: dict with keys 'flow_estimates' and 'uncertainty_estimates'. The first contains the flow field
                        of the two deepest levels corresponding to the L-Net (flow4 and flow3), they are scaled for
                        input resolution of 256x256.
                        The uncertainty estimates correspond to the log_var_map and weight_map for both levels.
            output: dict with keys 'flow_estimates' and 'uncertainty_estimates'. The first contains the flow field
                    of the two shallowest levels corresponding to the H-Net (flow2 and flow1), they are scaled for
                    original (high resolution) input resolution
                    The uncertainty estimates correspond to the log_var_map and weight_map for both levels.
        """
        # im1 is target image, im2 is source image
        b, _, h_original, w_original = im_target.size()
        b, _, h_256, w_256 = im_target_256.size()

        c14, c24, c13, c23, c12, c22, c11, c21 = self.extract_features(im_target, im_source,
                                                                       im_target_256, im_source_256,
                                                                       im_target_pyr=im_target_pyr,
                                                                       im_source_pyr=im_source_pyr,
                                                                       im_target_pyr_256=im_target_pyr_256,
                                                                       im_source_pyr_256=im_source_pyr_256)

        # RESOLUTION 256x256
        # level 4
        flow4, log_var_map4, weight_map4,  corr4 = self.estimate_at_mappinglevel(self.corr_uncertainty_decoder4,
                                                                                 self.uncertainty_decoder4,
                                                                                 c14, c24, h_256, w_256)

        up_flow4, up_log_var_map4, up_probability_map4, up_feat4 = self.upscaling(_, flow4, log_var_map4,
                                                                                  weight_map4, (32, 32),
                                                                                  deconv=self.deconv4)
        if self.estimate_one_mode:
            up_uncertainty_components4 = up_log_var_map4
        else:
            up_uncertainty_components4 = torch.cat((up_log_var_map4, up_probability_map4), 1)

        # level 3: 32x32
        x3, flow3, log_var_map3, weight_map3 = self.estimate_at_flowlevel(ratio=32.0 / float(w_256),
                                                                          c_t=c13, c_s=c23,
                                                                          up_flow=up_flow4, up_uncertainty_components=
                                                                          up_uncertainty_components4,
                                                                          decoder=self.decoder3, PWCNetRefinement=
                                                                          'PWCNetRefinementAdaptiveReso',
                                                                          corr_uncertainty_module=
                                                                          self.corr_uncertainty_decoder3,
                                                                          uncertainty_predictor=self.uncertainty_decoder3,
                                                                          sigma_max=self.var_2_plus_256,
                                                                          up_feat=None, div=1.0,
                                                                          refinement=self.params.refinement_at_adaptive_reso)

        up_flow3, up_log_var_map3, up_probability_map3, up_feat3 = self.upscaling(x3, flow3, log_var_map3,
                                                                                  weight_map3,
                                                                                  (h_original//8.0, w_original//8.0))

        # before the flow was scaled to h_256xw_256. Now, since we go to the high resolution images, we need
        # to scale the flow to h_original x w_original
        up_flow3[:, 0, :, :] *= float(w_original) / float(w_256)
        up_flow3[:, 1, :, :] *= float(h_original) / float(h_256)
        if self.scale_low_resolution:
            # scale the variance from 256 to w_original or h_original
            # APPROXIMATION FOR NON-SQUARE IMAGES --> use the diagonal
            diag_original = math.sqrt(h_original ** 2 + w_original ** 2)
            diag_256 = math.sqrt(h_256 ** 2 + w_256 ** 2)
            up_log_var_map3 += 2 * math.log(diag_original / float(diag_256))

        if self.estimate_one_mode:
            up_uncertainty_components3 = up_log_var_map3
        else:
            up_uncertainty_components3 = torch.cat((up_log_var_map3, up_probability_map3), 1)

        # level 2 : 1/8 of original resolution
        x2, flow2, log_var_map2, weight_map2 = self.estimate_at_flowlevel(ratio=1.0/8.0, c_t=c12, c_s=c22,
                                                                          up_flow=up_flow3,
                                                                          up_uncertainty_components=up_uncertainty_components3,
                                                                          decoder=self.decoder2,
                                                                          PWCNetRefinement='PWCNetRefinementFinal',
                                                                          corr_uncertainty_module=
                                                                          self.corr_uncertainty_decoder2,
                                                                          uncertainty_predictor=self.uncertainty_decoder2,
                                                                          sigma_max=self.var_2_plus,
                                                                          up_feat=up_feat3, div=1.0,
                                                                          refinement=self.params.refinement_at_all_levels)

        up_flow2, up_log_var_map2, up_probability_map2, up_feat2 = self.upscaling(x2, flow2, log_var_map2, weight_map2,
                                                                                  (h_original//4.0, w_original//4.0),
                                                                                  self.deconv2, self.upfeat2)
        if self.estimate_one_mode:
            up_uncertainty_components2 = up_log_var_map2
        else:
            up_uncertainty_components2 = torch.cat((up_log_var_map2, up_probability_map2), 1)

        # level 1: 1/4 of original resolution
        x1, flow1, log_var_map1, weight_map1 = self.estimate_at_flowlevel(ratio=1.0/4.0, c_t=c11, c_s=c21,
                                                                          up_flow=up_flow2,
                                                                          up_uncertainty_components=
                                                                          up_uncertainty_components2, decoder=self.decoder1,
                                                                          PWCNetRefinement='PWCNetRefinementFinal',
                                                                          corr_uncertainty_module=
                                                                          self.corr_uncertainty_decoder1,
                                                                          uncertainty_predictor=self.uncertainty_decoder1,
                                                                          sigma_max=self.var_2_plus,
                                                                          up_feat=up_feat2, div=1.0,
                                                                          refinement=self.params.refinement_at_finest_level)

        if self.scale_low_resolution:
            # Here, we also want to scale the low resolution flows (flow4 and flow3) to the high resolution
            # original image sizes h_original x w_original
            # prepare output dict
            output_256 = {'flow_estimates': [flow4, flow3], 'correlation': corr4,
                          'uncertainty_estimates': [[log_var_map4, weight_map4], [log_var_map3, weight_map3]]}

            # need to scale the log variance of the LNet.
            # also scale the log variance of the small variance ==> it will correspond to a higher variance
            # APPROXIMATION FOR NON-SQUARE IMAGES --> use the diagonal
            diag_original = math.sqrt(h_original ** 2 + w_original ** 2)
            diag_256 = math.sqrt(h_256 ** 2 + w_256 ** 2)

            flow4 = flow4.clone()
            flow4[:, 0] *= float(w_original) / float(w_256)
            flow4[:, 1] *= float(h_original) / float(h_256)
            log_var_map4 = log_var_map4.clone()
            log_var_map4 += 2 * math.log(diag_original / float(diag_256))

            flow3 = flow3.clone()
            flow3[:, 0] *= float(w_original) / float(w_256)
            flow3[:, 1] *= float(h_original) / float(h_256)
            log_var_map3 = log_var_map3.clone()
            log_var_map3 += 2 * math.log(diag_original / float(diag_256))

            if self.estimate_one_mode:
                # unimodal
                output = {'flow_estimates': [flow4, flow3, flow2, flow1],
                          'uncertainty_estimates': [log_var_map4, log_var_map3, log_var_map2, log_var_map1]}
            else:
                # multi-modal
                output = {'flow_estimates': [flow4, flow3, flow2, flow1],
                          'uncertainty_estimates': [[log_var_map4, weight_map4], [log_var_map3, weight_map3],
                                                    [log_var_map2, weight_map2], [log_var_map1, weight_map1]]}
        else:
            if self.estimate_one_mode:
                output_256 = {'flow_estimates': [flow4, flow3], 'correlation': corr4,
                              'uncertainty_estimates': [log_var_map4, log_var_map3]}
                output = {'flow_estimates': [flow2, flow1],
                          'uncertainty_estimates': [log_var_map2, log_var_map1]}
            else:
                # correspond to the L-Net predictions
                output_256 = {'flow_estimates': [flow4, flow3], 'correlation': corr4,
                              'uncertainty_estimates': [[log_var_map4, weight_map4], [log_var_map3, weight_map3]]}
                # correspond to the H-Net
                output = {'flow_estimates': [flow2, flow1],
                          'uncertainty_estimates': [[log_var_map2, weight_map2], [log_var_map1, weight_map1]]}
        return output_256, output


@model_constructor
def PDCNet_vgg16(global_corr_type='feature_corr_layer', global_gocor_arguments=None, normalize='relu_l2norm',
                 cyclic_consistency=False, local_corr_type='feature_corr_layer', local_gocor_arguments=None,
                 same_local_corr_at_all_levels=True, decoder_inputs='corr_flow_feat',
                 local_decoder_type='OpticalFlowEstimator', global_decoder_type='CMDTop',
                 apply_refinement_finest_resolution=True, refinement_at_finest_level=True,
                 corr_for_corr_uncertainty_decoder='gocor', use_interp_instead_of_deconv=False, init_deconv_w_bilinear=True,
                 give_layer_before_flow_to_uncertainty_decoder=True,
                 var_2_plus=0.0, var_2_plus_256=0.0, var_1_minus_plus=1.0, var_2_minus=2.0,
                 estimate_three_modes=False, var_3_minus_plus=520 ** 2, var_3_minus_plus_256=256 ** 2,
                 estimate_one_mode=False, laplace_distr=True,
                 make_two_feature_copies=False, train_features=False, scale_low_resolution=False):

    net = PDCNetModel(global_gocor_arguments=global_gocor_arguments, global_corr_type=global_corr_type, normalize=normalize,
                      normalize_features=True, cyclic_consistency=cyclic_consistency,
                      local_corr_type=local_corr_type, local_gocor_arguments=local_gocor_arguments,
                      same_local_corr_at_all_levels=same_local_corr_at_all_levels,
                      local_decoder_type=local_decoder_type, global_decoder_type=global_decoder_type,
                      batch_norm=True, pyramid_type='VGG', upfeat_channels=2, decoder_inputs=decoder_inputs,
                      refinement_at_all_levels=False, refinement_at_adaptive_reso=True,
                      refinement_at_finest_level=refinement_at_finest_level,
                      use_interp_instead_of_deconv=use_interp_instead_of_deconv,
                      init_deconv_w_bilinear=init_deconv_w_bilinear,
                      apply_refinement_finest_resolution=apply_refinement_finest_resolution,
                      corr_for_corr_uncertainty_decoder=corr_for_corr_uncertainty_decoder,
                      var_1_minus_plus=var_1_minus_plus, var_2_minus=var_2_minus,
                      var_2_plus=var_2_plus, var_2_plus_256=var_2_plus_256,
                      var_3_minus_plus=var_3_minus_plus,
                      var_3_minus_plus_256=var_3_minus_plus_256, estimate_three_modes=estimate_three_modes,
                      estimate_one_mode = estimate_one_mode, laplace_distr=laplace_distr,
                      give_layer_before_flow_to_uncertainty_decoder=give_layer_before_flow_to_uncertainty_decoder,
                      make_two_feature_copies=make_two_feature_copies, train_features=train_features,
                      scale_low_resolution=scale_low_resolution)
    return net

