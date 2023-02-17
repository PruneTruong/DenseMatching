import numpy as np
from models.modules.mod import OpticalFlowEstimator,  OpticalFlowEstimatorNoDenseConnection, CMDTop, \
    CMDTopResidualConnections, OpticalFlowEstimatorResidualConnection


def initialize_mapping_decoder_(decoder_type, in_channels, batch_norm=True, **kwargs):
    if decoder_type == 'CMDTop':
        # original DGC-Net mapping decoder
        nbr_channels_layer_before_last = 32
        decoder = CMDTop(in_channels=in_channels, batch_norm=batch_norm, **kwargs)
    elif decoder_type == 'CMDTopResidualConnection':
        # DGC Net mapping decoder with added residual connections.
        nbr_channels_layer_before_last = 32
        decoder = CMDTopResidualConnections(in_channels=in_channels, batch_norm=batch_norm, **kwargs)
    else:
        raise NotImplementedError('Unknown mapping decoder type: {}'.format(decoder_type))
    return decoder, nbr_channels_layer_before_last


def initialize_flow_decoder_(decoder_type, decoder_inputs,  in_channels_corr, nbr_upfeat_channels, batch_norm=True, **kwargs):
    if decoder_inputs == 'corr_flow_feat':
        # our default choice. The flow decoder will take as input the local correlation, previous estimated flow field,
        # and upfeat from the previous level (from PWCNet).
        od = in_channels_corr + 2 + nbr_upfeat_channels
    elif decoder_inputs == 'corr':
        od = in_channels_corr
    elif decoder_inputs == 'corr_flow':
        od = in_channels_corr + 2
    elif decoder_inputs == 'flow_and_feat_and_outofviewcorr':
        od = in_channels_corr * 2 + 2 + nbr_upfeat_channels
    else:
        raise ValueError('Unknown decoder input: {}'.format(decoder_inputs))
    # choose the decoder of this level
    if decoder_type == 'OpticalFlowEstimator':
        dd = np.cumsum([128, 128, 96, 64, 32])
        decoder = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm, **kwargs)
        nbr_channels_layer_before_last = od + dd[-1]
    elif decoder_type == 'OpticalFlowEstimatorNoDenseConnection':
        decoder = OpticalFlowEstimatorNoDenseConnection(in_channels=od, batch_norm=batch_norm, **kwargs)
        nbr_channels_layer_before_last = 32
    elif decoder_type == 'OpticalFlowEstimatorResidualConnection':
        decoder = OpticalFlowEstimatorResidualConnection(in_channels=od, batch_norm=batch_norm, **kwargs)
        nbr_channels_layer_before_last = 32
    else:
        raise NotImplementedError('Unknown floa decoder type: {}'.format(decoder_type))
    return decoder, nbr_channels_layer_before_last
