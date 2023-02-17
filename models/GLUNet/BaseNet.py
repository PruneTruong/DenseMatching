import torch
import torch.nn.functional as F


from models.feature_backbones.VGG_features import VGGPyramid
from models.modules.mod import deconv, unnormalise_and_convert_mapping_to_flow
from models.base_matching_net import Base3LevelsMultiScaleMatchingNet
from models.base_matching_net import set_basenet_parameters
from models.modules.local_correlation import correlation
from admin.model_constructor import model_constructor


class BaseNet(Base3LevelsMultiScaleMatchingNet):
    """
    BaseNet takes fixed input images of 256x256.
    The flows are predicted such that they are scaled to the input image resolution (256x256).
    To obtain the flow from target to source at 256x256 resolution, one just needs to bilinearly upsample
    (without further scaling).
    """
    def __init__(self, global_corr_type='global_corr', normalize='relu_l2norm', normalize_features=True,
                 cyclic_consistency=False, global_gocor_arguments=None,
                 local_corr_type='local_corr', local_gocor_arguments=None, same_local_corr_at_all_levels=True,
                 local_decoder_type='OpticalFlowEstimator', global_decoder_type='CMDTop',
                 refinement=True, input_decoder='corr_flow_feat', refinement_32=False, md=4, nbr_upfeat=2,
                 batch_norm=True, pyramid_type='VGG', train_features=False):

        params = set_basenet_parameters(global_corr_type=global_corr_type, global_gocor_arguments=global_gocor_arguments,
                                        normalize=normalize, normalize_features=normalize_features,
                                        cyclic_consistency=cyclic_consistency, md=md,
                                        local_corr_type=local_corr_type, local_gocor_arguments=local_gocor_arguments,
                                        same_local_corr_at_all_levels=same_local_corr_at_all_levels,
                                        local_decoder_type=local_decoder_type,
                                        global_decoder_type=global_decoder_type, decoder_inputs=input_decoder,
                                        refinement=refinement, refinement_32=refinement_32, batch_norm=batch_norm,
                                        nbr_upfeat_channels=nbr_upfeat)

        super().__init__(params)

        self.nbr_features = self.get_nbr_features_pyramid(pyramid_type)

        # level 16x16, global correlation
        nd = 16 * 16  # global correlation
        if self.params.add_info_correlation:
            nd += 2
        od = nd + 2
        decoder4, num_channels_last_conv = self.initialize_mapping_decoder(self.params.global_decoder_type,
                                                                           in_channels=od,
                                                                           batch_norm=self.params.batch_norm,
                                                                           output_x=True)
        self.decoder4 = decoder4

        # level 32x32, constrained correlation, 4 pixels on each side
        nd = (2 * self.params.md + 1) ** 2
        if self.params.add_info_correlation:
            nd += 2
        decoder3, num_channels_last_conv = self.initialize_flow_decoder(decoder_type=self.params.local_decoder_type,
                                                                        decoder_inputs=self.params.decoder_inputs,
                                                                        nbr_upfeat_channels=0,
                                                                        in_channels_corr=nd)
        input_to_refinement = num_channels_last_conv
        self.decoder3 = decoder3
        if 'feat' in self.params.decoder_inputs:
            self.upfeat3 = deconv(num_channels_last_conv, self.params.nbr_upfeat_channels, kernel_size=4,
                                  stride=2, padding=1)

        if self.params.refinement_32:
            self.initialize_intermediate_level_refinement_module(input_to_refinement, self.params.batch_norm)

        nd = (2 * self.params.md + 1) ** 2  # constrained correlation, 4 pixels on each side
        if self.params.add_info_correlation:
            nd += 2
        decoder2, num_channels_last_conv = self.initialize_flow_decoder(decoder_type=self.params.local_decoder_type,
                                                                        decoder_inputs=self.params.decoder_inputs,
                                                                        nbr_upfeat_channels=2,
                                                                        in_channels_corr=nd)
        self.decoder2 = decoder2
        input_to_refinement = num_channels_last_conv

        if self.params.refinement:
            self.initialize_last_level_refinement_module(input_to_refinement, self.params.batch_norm)

        self.initialize_global_corr()
        self.initialize_local_corr()

        if pyramid_type == 'VGG':
            feature_extractor = VGGPyramid(train=train_features)
            self.pyramid = feature_extractor
        else:
            raise NotImplementedError

    def forward(self, im_target, im_source, im_target_pyr=None, im_source_pyr=None):
        """
        Args:
            im_target: torch Tensor Bx3x256x256, normalized with imagenet weights
            im_source: torch Tensor Bx3x256x256, normalized with imagenet weights
            im_target_pyr: in case the pyramid features are already computed.
            im_source_pyr: in case the pyramid features are already computed.

        Returns:
            output: dict with keys 'flow_estimates'. It contains the flow field of the three pyramid levels, 
                    they are scaled for fixed 256x256 input resolution.
        """

        b, _, h_original, w_original = im_target.size()
        div = 1.0

        c14, c24, c13, c23, c12, c22 = self.extract_features(im_target, im_source, im_target_pyr, im_source_pyr)

        # level 4: 16x16
        # here same procedure then DGC-Net global mapping decoder.
        corr4 = self.get_global_correlation(c14, c24)
        b, c, h, w = corr4.size()
        ratio_x = w / float(w_original)
        ratio_y = h / float(h_original)

        if torch.cuda.is_available():
            init_map = torch.FloatTensor(b, 2, h, w).zero_().cuda()
        else:
            init_map = torch.FloatTensor(b, 2, h, w).zero_()

        # init_map is fed to the decoder to be consistent with decoder of DGC-Net (but not particularly needed)
        x4, est_map4 = self.decoder4(torch.cat((corr4, init_map), 1))
        flow4 = unnormalise_and_convert_mapping_to_flow(est_map4)

        # we want flow4 to be scaled for h_original x w_original, so we multiply by these ratios.
        flow4 = flow4 / div
        flow4[:, 0, :, :] /= ratio_x
        flow4[:, 1, :, :] /= ratio_y
        up_flow4 = F.interpolate(input=flow4, size=(int(h_original / 8.0), int(w_original / 8.0)), mode='bilinear',
                                 align_corners=False)

        # level 3 32x32
        ratio_x = up_flow4.shape[3] / float(w_original)
        ratio_y = up_flow4.shape[2] / float(h_original)
        # the flow needs to be scaled for the current resolution in order to warp.
        # otherwise flow4 and up_flow4 are scaled for h_original x w_original
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
        elif self.params.decoder_inputs == 'flow':
            corr3 = torch.cat((corr3, up_flow4), 1)

        x3, res_flow3 = self.decoder3(corr3)

        if self.params.refinement_32:
            x_, res_flow3_ = self.PWCNetRefinementIntermediateReso(x3)
            res_flow3 = res_flow3 + res_flow3_
        flow3 = res_flow3 + up_flow4

        up_flow3 = F.interpolate(input=flow3, size=(int(h_original / 4.0), int(w_original / 4.0)), mode='bilinear',
                                 align_corners=False)
        if 'feat' in self.params.decoder_inputs:
            # same upfeat than in PWCNet
            up_feat3 = self.upfeat3(x3)

        # level 2 64x64
        ratio_x = up_flow3.shape[3] / float(w_original)
        ratio_y = up_flow3.shape[2] / float(h_original)
        up_flow_3_warping = up_flow3 * div
        up_flow_3_warping[:, 0, :, :] *= ratio_x
        up_flow_3_warping[:, 1, :, :] *= ratio_y
        warp2 = self.warp(c22, up_flow_3_warping)

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
            corr2 = torch.cat((corr2, up_flow3, up_feat3), 1)
        elif self.params.decoder_inputs == 'flow':
            corr2 = torch.cat((corr2, up_flow3), 1)

        x, res_flow2 = self.decoder2(corr2)

        if self.params.refinement:
            x_, res_flow2_ = self.PWCNetRefinementFinal(x)
            res_flow2 = res_flow2 + res_flow2_

        flow2 = res_flow2 + up_flow3

        # prepare output dict
        output = {'flow_estimates': [flow4, flow3, flow2], 'correlation': corr4}
        return output


@model_constructor
def basenet_vgg16(global_corr_type='global_corr', normalize='relu_l2norm', normalize_features=True,
                  cyclic_consistency=False, global_gocor_arguments=None,
                  local_corr_type='local_corr', local_gocor_arguments=None, same_local_corr_at_all_levels=True,
                  local_decoder_type='OpticalFlowEstimator', global_decoder_type='CMDTop',
                  batch_norm=True, pyramid_type='VGG', refinement=True, refinement_32=False,
                  input_decoder='corr_flow_feat', md=4, nbr_upfeat=2, train_features=False):

    net = BaseNet(global_gocor_arguments=global_gocor_arguments, global_corr_type=global_corr_type, normalize=normalize,
                  normalize_features=normalize_features, cyclic_consistency=cyclic_consistency,
                  local_corr_type=local_corr_type, local_gocor_arguments=local_gocor_arguments,
                  same_local_corr_at_all_levels=same_local_corr_at_all_levels,
                  local_decoder_type=local_decoder_type, global_decoder_type=global_decoder_type,
                  batch_norm=batch_norm, pyramid_type=pyramid_type,  refinement=refinement,
                  input_decoder=input_decoder, refinement_32=refinement_32, md=md, nbr_upfeat=nbr_upfeat,
                  train_features=train_features)
    return net
