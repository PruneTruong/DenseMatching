from models.GLUNet.GLU_Net import GLUNetModel
from models.PWCNet.pwc_net import PWCNetModel
from models.PDCNet.PDCNet import PDCNet_vgg16
import os.path as osp
import torch
import os


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

    try:
        net.load_state_dict(checkpoint_dict['state_dict'])
    except:
        net.load_state_dict(checkpoint_dict)
    return net


model_type = ['GLUNet', 'GLUNet_GOCor', 'PWCNet', 'PWCNet_GOCor', 'GLUNet_GOCor_star', 'PDCNet']
pre_trained_model_types = ['static', 'dynamic', 'chairs_things', 'chairs_things_ft_sintel', 'megadepth']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_model(model_name, pre_trained_model_type, arguments, global_optim_iter, local_optim_iter,
                 path_to_pre_trained_models='../pre_trained_models/ours/PDCNet'):
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

    if pre_trained_model_type not in pre_trained_model_types:
        raise ValueError(
            'The pre trained model that you chose does not exist, you chose {}'.format(pre_trained_model_types))

    estimate_uncertainty = False
    if model_name == 'GLUNet':
        # GLU-Net uses a global feature correlation layer followed by a cyclic consistency post-processing.
        # local cost volumes are computed by feature correlation layers
        network = GLUNetModel(iterative_refinement=True, global_corr_type='feature_corr_layer',
                              normalize='relu_l2norm', cyclic_consistency=True,
                              local_corr_type='feature_corr_layer')

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

    elif model_name == 'PDCNet':
        estimate_uncertainty = True
        # for global gocor, we apply L_r and L_q within the optimizer module
        global_gocor_arguments = {'optim_iter': global_optim_iter, 'steplength_reg': 0.1, 'train_label_map': False,
                                  'apply_query_loss': True,
                                  'reg_kernel_size': 3, 'reg_inter_dim': 16, 'reg_output_dim': 16}

        # for global gocor, we apply L_r only
        local_gocor_arguments = {'optim_iter': local_optim_iter, 'steplength_reg': 0.1}
        network = PDCNet_vgg16(global_corr_type='GlobalGOCor', global_gocor_arguments=global_gocor_arguments,
                               normalize='leakyrelu', same_local_corr_at_all_levels=True,
                               local_corr_type='LocalGOCor', local_gocor_arguments=local_gocor_arguments,
                               local_decoder_type='OpticalFlowEstimatorResidualConnection',
                               global_decoder_type='CMDTopResidualConnection',
                               corr_for_corr_uncertainty_decoder='corr',
                               give_layer_before_flow_to_uncertainty_decoder=True,
                               var_2_plus=520 ** 2, var_2_plus_256=256 ** 2, var_1_minus_plus=1.0, var_2_minus=2.0)

    else:
        raise NotImplementedError('the model that you chose does not exist: {}'.format(model_name))

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
    if arguments.network_type == 'PDCNet':
        # define inference parameters for PDC-Net and particularly the ones needed for multi-stage alignment
        network.set_inference_parameters(confidence_R=arguments.confidence_map_R,
                                         ransac_thresh=arguments.ransac_thresh,
                                         multi_stage_type=arguments.multi_stage_type,
                                         mask_type_for_2_stage_alignment=arguments.mask_type,
                                         homography_visibility_mask=arguments.homography_visibility_mask,
                                         list_resizing_ratios=arguments.scaling_factors)

    '''
    to plot GOCor weights
    if model_name == 'GLUNet_GOCor':
        network.corr.corr_module.filter_optimizer._plot_weights(save_dir='evaluation/')
        network.local_corr.filter_optimizer._plot_weights(save_dir='evaluation/')
    '''
    return network, estimate_uncertainty
