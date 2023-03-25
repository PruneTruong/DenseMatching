import os.path as osp
import torch
import os

from models.GLUNet.GLU_Net import GLUNetModel
from models.PWCNet.pwc_net import PWCNetModel
from models.PDCNet.PDCNet import PDCNet_vgg16
from models.GLUNet.Semantic_GLUNet import SemanticGLUNetModel
from models.semantic_matching_models.SFNet import SFNet, SFNetWithBin
from models.semantic_matching_models.NCNet import NCNetWithBin, ImMatchNet
from models.semantic_matching_models.cats import CATs


def load_network(net, checkpoint_path=None, **kwargs):
    """Loads a network checkpoint file.
    args:
        net: network architecture
        checkpoint_path
    outputs:
        net: loaded network
    """

    if not os.path.isfile(checkpoint_path):
        raise ValueError('The checkpoint that you chose does not exist, {}'.format(checkpoint_path))

    # Load checkpoint
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    if 'state_dict' in checkpoint_dict:
        checkpoint_dict = checkpoint_dict['state_dict']

    net.load_state_dict(checkpoint_dict, strict=False)
    return net


model_type = ['GLUNet', 'GLUNet_interp',
              'GLUNet_GOCor', 'PWCNet', 'PWCNet_GOCor',
              'GLUNet_GOCor_star', 'PDCNet', 'PDCNet_plus',
              'GLUNet_star', 'WarpCGLUNet', 'SemanticGLUNet', 'WarpCSemanticGLUNet', 'WarpCGLUNet_interp',
              'UAWarpC',
              'SFNet', 'PWarpCSFNet_WS', 'PWarpCSFNet_SS', 'NCNet', 'PWarpCNCNet_WS', 'PWarpCNCNet_SS',
              'CATs', 'PWarpCCATs_SS', 'CATs_ft_features', 'PWarpCCATs_ft_features_SS',
              ]
pre_trained_model_types = ['static', 'dynamic', 'chairs_things', 'chairs_things_ft_sintel', 'megadepth',
                           'megadepth_stage1', 'pfpascal', 'spair']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_model(model_name, pre_trained_model_type, arguments, global_optim_iter, local_optim_iter,
                 path_to_pre_trained_models='pre_trained_models/'):
    """
    Select, construct and load model
    args:
        model_name
        pre_trained_model_type
        global_optim_iter
        local_optim_iter
        path_to_pre_trained_models
    output:
        network: constructed and loaded network
    """

    print('Model: {}\nPre-trained-model: {}'.format(model_name, pre_trained_model_type))
    if model_name not in model_type:
        raise ValueError(
            'The model that you chose does not exist, you chose {}'.format(model_name))

    if 'GOCor' in model_name or 'PDCNet' in model_name:
        print('GOCor: Local iter {}'.format(local_optim_iter))
        print('GOCor: Global iter {}'.format(global_optim_iter))

    '''
    if pre_trained_model_type not in pre_trained_model_types:
        raise ValueError(
            'The pre trained model that you chose does not exist, you chose {}'.format(pre_trained_model_types))
    '''

    estimate_uncertainty = False
    if model_name == 'GLUNet':
        # GLU-Net uses a global feature correlation layer followed by a cyclic consistency post-processing.
        # local cost volumes are computed by feature correlation layers
        network = GLUNetModel(iterative_refinement=True, global_corr_type='feature_corr_layer',
                              normalize='relu_l2norm', cyclic_consistency=True,
                              local_corr_type='feature_corr_layer')
    elif model_name == 'GLUNet_interp':
        network = GLUNetModel(iterative_refinement=True, global_corr_type='feature_corr_layer',
                              normalize='relu_l2norm', cyclic_consistency=True,
                              local_corr_type='feature_corr_layer', use_interp_instead_of_deconv=True)

    elif model_name == 'GLUNet_GOCor':
        '''
        Default for global and local gocor arguments:
        global_gocor_arguments = {'optim_iter':3, 'num_features': 512, 'init_step_length': 1.0, 
                                  'init_filter_reg': 1e-2, 'min_filter_reg': 1e-5, 'steplength_reg': 0.0, 
                                  'num_dist_bins':10, 'bin_displacement': 0.5, 'init_gauss_sigma_DIMP':1.0,
                                  'v_minus_act': 'sigmoid', 'v_minus_init_factor': 4.0
                                  'apply_query_loss': False, 'reg_kernel_size': 3, 
                                  'reg_inter_dim': 1, 'reg_output_dim': 1.0}
        
        local_gocor_arguments= {'optim_iter':3, 'num_features': 512, 'search_size': 9, 'init_step_length': 1.0,
                                'init_filter_reg': 1e-2, 'min_filter_reg': 1e-5, 'steplength_reg': 0.0, 
                                'num_dist_bins':10, 'bin_displacement': 0.5, 'init_gauss_sigma_DIMP':1.0,
                                'v_minus_act': 'sigmoid', 'v_minus_init_factor': 4.0
                                'apply_query_loss': False, 'reg_kernel_size': 3, 
                                'reg_inter_dim': 1, 'reg_output_dim': 1.0}
        '''
        # for global gocor, we apply L_r and L_q within the optimizer module
        global_gocor_arguments = {'optim_iter': global_optim_iter, 'apply_query_loss': True,
                                  'reg_kernel_size': 3, 'reg_inter_dim': 16, 'reg_output_dim': 16}

        # for global gocor, we apply L_r only
        local_gocor_arguments = {'optim_iter': local_optim_iter}
        network = GLUNetModel(iterative_refinement=True, global_corr_type='GlobalGOCor',
                              global_gocor_arguments=global_gocor_arguments, normalize='leakyrelu',
                              local_corr_type='LocalGOCor', local_gocor_arguments=local_gocor_arguments,
                              same_local_corr_at_all_levels=True)

    elif model_name == 'PWCNet':
        # PWC-Net uses a feature correlation layer at each pyramid level
        network = PWCNetModel(local_corr_type='feature_corr_layer')

    elif model_name == 'PWCNet_GOCor':
        local_gocor_arguments = {'optim_iter': local_optim_iter}
        # We instead replace the feature correlation layers by Local GOCor modules
        network = PWCNetModel(local_corr_type='LocalGOCor', local_gocor_arguments=local_gocor_arguments,
                              same_local_corr_at_all_levels=False)

    elif model_name == 'GLUNet_GOCor_star':
        # different mapping and flow decoders, features are also finetuned with two VGG copies

        # for global gocor, we apply L_r and L_q within the optimizer module
        global_gocor_arguments = {'optim_iter': global_optim_iter,  'steplength_reg': 0.1, 'apply_query_loss': True,
                                  'reg_kernel_size': 3, 'reg_inter_dim': 16, 'reg_output_dim': 16}

        # for global gocor, we apply L_r only
        local_gocor_arguments = {'optim_iter': local_optim_iter, 'steplength_reg': 0.1}
        network = GLUNetModel(iterative_refinement=True, cyclic_consistency=False, global_corr_type='GlobalGOCor',
                              global_gocor_arguments=global_gocor_arguments, normalize='leakyrelu',
                              local_corr_type='LocalGOCor', local_gocor_arguments=local_gocor_arguments,
                              same_local_corr_at_all_levels=True, give_flow_to_refinement_module=True,
                              local_decoder_type='OpticalFlowEstimatorResidualConnection',
                              global_decoder_type='CMDTopResidualConnection', make_two_feature_copies=True)

    elif model_name == 'PDCNet' or model_name == 'PDCNet_plus':
        estimate_uncertainty = True
        # for global gocor, we apply L_r and L_q within the optimizer module
        global_gocor_arguments = {'optim_iter': global_optim_iter, 'steplength_reg': 0.1, 'train_label_map': False,
                                  'apply_query_loss': True,
                                  'reg_kernel_size': 3, 'reg_inter_dim': 16, 'reg_output_dim': 16}

        # for global gocor, we apply L_r only
        local_gocor_arguments = {'optim_iter': local_optim_iter, 'steplength_reg': 0.1}
        if model_name == 'PDCNet':
            network = PDCNet_vgg16(global_corr_type='GlobalGOCor', global_gocor_arguments=global_gocor_arguments,
                                   normalize='leakyrelu', same_local_corr_at_all_levels=True,
                                   local_corr_type='LocalGOCor', local_gocor_arguments=local_gocor_arguments,
                                   local_decoder_type='OpticalFlowEstimatorResidualConnection',
                                   global_decoder_type='CMDTopResidualConnection',
                                   corr_for_corr_uncertainty_decoder='corr',
                                   give_layer_before_flow_to_uncertainty_decoder=True,
                                   var_2_plus=520 ** 2, var_2_plus_256=256 ** 2, var_1_minus_plus=1.0, var_2_minus=2.0)
        else:
            network = PDCNet_vgg16(global_corr_type='GlobalGOCor', global_gocor_arguments=global_gocor_arguments,
                                   normalize='leakyrelu', same_local_corr_at_all_levels=True,
                                   local_corr_type='LocalGOCor', local_gocor_arguments=local_gocor_arguments,
                                   local_decoder_type='OpticalFlowEstimatorResidualConnection',
                                   global_decoder_type='CMDTopResidualConnection',
                                   corr_for_corr_uncertainty_decoder='corr',
                                   give_layer_before_flow_to_uncertainty_decoder=True,
                                   var_2_plus=520 ** 2, var_2_plus_256=256 ** 2, var_1_minus_plus=1.0, var_2_minus=2.0,
                                   make_two_feature_copies=True)
    elif model_name == 'UAWarpC':
        estimate_uncertainty = True
        # probabilistic model with a single mode. Uncertainty predictors are based on PDCNet. Does not use GOCor.
        # the predictions from the LNet (pyramid network taking 256x256 images) are scaled to original resolution
        network = PDCNet_vgg16(global_corr_type='global_corr', normalize='relu_l2norm',
                               cyclic_consistency=True, local_decoder_type='OpticalFlowEstimatorResidualConnection',
                               global_decoder_type='CMDTopResidualConnection',
                               use_interp_instead_of_deconv=True, scale_low_resolution=True,
                               corr_for_corr_uncertainty_decoder='corr', estimate_one_mode=True,
                               laplace_distr=False,  # Gaussian distribution
                               give_layer_before_flow_to_uncertainty_decoder=True)
    elif model_name == 'GLUNet_star' or model_name == 'WarpCGLUNet':
        # replaced the DenseNet connections in original network by residual connections to make the network lighter.
        network = GLUNetModel(iterative_refinement=True, global_corr_type='feature_corr_layer',
                              normalize='relu_l2norm', cyclic_consistency=True,
                              local_corr_type='feature_corr_layer',
                              local_decoder_type='OpticalFlowEstimatorResidualConnection',
                              global_decoder_type='CMDTopResidualConnection')
    elif model_name == 'WarpCGLUNet_interp':
        network = GLUNetModel(iterative_refinement=True, global_corr_type='feature_corr_layer',
                              normalize='relu_l2norm', cyclic_consistency=True,
                              local_corr_type='feature_corr_layer',
                              local_decoder_type='OpticalFlowEstimatorResidualConnection',
                              global_decoder_type='CMDTopResidualConnection', use_interp_instead_of_deconv=True)
    elif model_name == 'SemanticGLUNet' or model_name == 'WarpCSemanticGLUNet':
        network = SemanticGLUNetModel(iterative_refinement=True)

    # ########################## PWarpC Semantic networks #####################################################
    # the architecture is the unchanged compared to original works, only the training is different (and the
    # inference strategy in some cases).
    elif model_name == 'SFNet':
        network = SFNet(forward_pass_strategy='flow', inference_strategy='softargmax_padding')
    elif model_name == 'PWarpCSFNet_SS':
        network = SFNet(forward_pass_strategy='corr_prediction_no_kernel', inference_strategy='argmax')
    elif model_name == 'PWarpCSFNet_WS':
        network = SFNetWithBin(forward_pass_strategy='corr_prediction_no_kernel', inference_strategy='argmax')
    elif model_name == 'NCNet' or model_name == 'PWarpCNCNet_SS':
        network = ImMatchNet(inference_strategy='argmax')
    elif model_name == 'PWarpCNCNet_WS':
        network = NCNetWithBin(inference_strategy='argmax')
    elif 'CATs' in model_name:
        # similar to original work, we use softargmax as the inference_strategy. This is because the kp loss is the
        # EPE after applying softargmax.
        network = CATs(forward_pass_strategy='flow_prediction', inference_strategy='softargmax')
    else:
        raise NotImplementedError('the model that you chose does not exist: {}'.format(model_name))

    if path_to_pre_trained_models.endswith('.pth') or path_to_pre_trained_models.endswith('.pth.tar') \
            or path_to_pre_trained_models.endswith('.pt'):
        # if the path already corresponds to a checkpoint path, we use it directly
        checkpoint_fname = path_to_pre_trained_models
    else:
        # it is the path to the directory containing all checkpoints.
        checkpoint_fname = osp.join(path_to_pre_trained_models, model_name + '_{}'.format(pre_trained_model_type)
                                    + '.pth')
        if not os.path.exists(checkpoint_fname):
            checkpoint_fname = checkpoint_fname + '.tar'

    if not os.path.exists(checkpoint_fname):
        raise ValueError('The checkpoint that you chose does not exist, {}'.format(checkpoint_fname))

    network = load_network(network, checkpoint_path=checkpoint_fname)
    network.eval()
    network = network.to(device)

    # define inference arguments
    if arguments.network_type == 'PDCNet' or arguments.network_type == 'PDCNet_plus':
        # define inference parameters for PDC-Net and particularly the ones needed for multi-stage alignment
        network.set_inference_parameters(confidence_R=arguments.confidence_map_R,
                                         ransac_thresh=arguments.ransac_thresh,
                                         multi_stage_type=arguments.multi_stage_type,
                                         mask_type_for_2_stage_alignment=arguments.mask_type,
                                         homography_visibility_mask=arguments.homography_visibility_mask,
                                         list_resizing_ratios=arguments.scaling_factors,
                                         compute_cyclic_consistency_error=arguments.compute_cyclic_consistency_error)

    '''
    to plot GOCor weights
    if model_name == 'GLUNet_GOCor':
        network.corr.corr_module.filter_optimizer._plot_weights(save_dir='evaluation/')
        network.local_corr.filter_optimizer._plot_weights(save_dir='evaluation/')
    '''
    return network, estimate_uncertainty
