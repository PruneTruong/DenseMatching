from utils_data.euler_wrapper import prepare_data
from termcolor import colored
import torch.optim as optim
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from training.actors.warp_consistency_actor_GLUNet import GLUNetWarpCUnsupervisedBatchPreprocessing, GLUNetWarpCUnsupervisedActor
from training.losses.basic_losses import L1
from training.losses.multiscale_loss import MultiScaleFlow
from training.trainers.matching_trainer import MatchingTrainer
from utils_data.loaders import Loader
from admin.multigpu import MultiGPU
from utils_data.sampler import RandomSampler, FirstItemsSampler
from utils_data.image_transforms import ArrayToTensor, ScaleToZeroOne
from training.actors.warp_consistency_utils.online_triplet_creation import BatchedImageTripletCreation
from training.actors.warp_consistency_utils.synthetic_flow_generation_from_pair_batch import \
    GetRandomSyntheticAffHomoTPSFlow, SynthecticAffHomoTPSTransfo
from utils_data.geometric_transformation_sampling.synthetic_warps_sampling import CompositionOfFlowCreations
from utils_data.geometric_transformation_sampling.synthetic_warps_sampling import AddElasticTransformsV2
from datasets.MegaDepth.megadepth import MegaDepthDataset
import torch
import numpy as np
import os
from models.GLUNet.GLU_Net import glunet_vgg16
from utils_data.augmentations.color_augmentation_torch import ColorJitter, RandomGaussianBlur


def run(settings):
    settings.description = 'Default train settings for WarpCGLUNet, stage2'
    settings.data_mode = 'euler'
    settings.batch_size = 6
    settings.n_threads = 8
    settings.multi_gpu = True
    settings.print_interval = 300
    settings.lr = 0.00005
    settings.n_epochs = 45
    settings.step_size_scheduler = [20, 30, 40]
    settings.resizing_size = 750
    settings.crop_size = 520
    settings.initial_pretrained_model = os.path.join(settings.env.workspace_dir,
                                                     'train_settings/WarpC/train_WarpC_GLUNet_stage1/',
                                                     'GLUNetModel_model_best.pth.tar')

    # specific training parameters
    # loss applied in non-black target prime regions (to account for warping). It is very important, if applied to
    # black regions, weird behavior. If valid mask, doesn't learn interpolation in non-visibile regions.
    settings.compute_mask_zero_borders = True
    settings.apply_mask = False  # valid visible matches, we apply mask_zero_borders instead
    settings.nbr_plot_images = 5
    settings.dataset_callback_fn = 'sample_new_items'  # use to resample image pair at each epoch

    # loss parameters
    settings.name_of_loss = 'warp_supervision_and_w_bipath'  # the warp consistency objective
    settings.compute_visibility_mask = True  # add the visibility mask in the second stage
    settings.apply_constant_flow_weight = False
    settings.loss_weight = {'warp_supervision': 1.0, 'w_bipath': 1.0,
                            'warp_supervision_constant': 1.0, 'w_bipath_constant': 1.0,
                            'cc_mask_alpha_1': 0.03, 'cc_mask_alpha_2': 0.5 }

    # transfo parameters for triplet creation
    # in the second training stage, transfo strength are increased, and we add elastic transformations
    settings.resizing_size = 750
    settings.crop_size = 520
    settings.parametrize_with_gaussian = False
    settings.transformation_types = ['hom', 'tps', 'afftps']
    settings.random_t = 0.25
    settings.random_s = 0.45
    settings.random_alpha = np.pi / 12
    settings.random_t_tps_for_afftps = 200.0 / float(settings.resizing_size)
    settings.random_t_hom = 300.0 / float(settings.resizing_size)
    settings.random_t_tps = 300.0 / float(settings.resizing_size)
    settings.elastic_parameters_ss = {"max_sigma": 0.08, "min_sigma": 0.1, "min_alpha": 1, "max_alpha": 1.0,
                                      'max_sigma_mask': 40.0, 'min_sigma_mask': 10.0}
    settings.appearance_transfo_target_prime = transforms.Compose([ColorJitter(brightness=0.6, contrast=0.6,
                                                                               saturation=0.6, hue=0.5 / 3.14),
                                                                   RandomGaussianBlur(sigma=(0.2, 2.0),
                                                                                      probability=0.2)])

    # apply pre-processing to the images
    # here pre-processing is done within the function
    flow_transform = transforms.Compose([ArrayToTensor()])  # just put channels first and put it to float
    img_transforms = transforms.Compose([ArrayToTensor(get_float=True)])  # just put channels first
    co_transform = None

    prepare_data(settings.env.megadepth_training_tar, mode=settings.data_mode)
    # original images must be at the resizing size, on which are applied the transformations!
    megadepth_cfg = {'scene_info_path': os.path.join(settings.env.megadepth_training, 'scene_info'),
                     'train_num_per_scene': 300, 'val_num_per_scene': 25,
                     'output_image_size': [settings.resizing_size, settings.resizing_size], 'pad_to_same_shape': False,
                     'output_flow_size': [[settings.resizing_size, settings.resizing_size]]}
    train_dataset = MegaDepthDataset(root=settings.env.megadepth_training, split='train', cfg=megadepth_cfg,
                                     source_image_transform=img_transforms, target_image_transform=img_transforms,
                                     flow_transform=flow_transform, co_transform=co_transform,
                                     store_scene_info_in_memory=True)

    # validation data
    val_dataset = MegaDepthDataset(root=settings.env.megadepth_training, cfg=megadepth_cfg, split='val',
                                   source_image_transform=img_transforms,
                                   target_image_transform=img_transforms,
                                   flow_transform=flow_transform, co_transform=co_transform,
                                   store_scene_info_in_memory=False)

    # dataloader
    train_loader = Loader('train', train_dataset, batch_size=settings.batch_size,
                          sampler=RandomSampler(train_dataset, num_samples=30000),
                          drop_last=True, training=True, num_workers=settings.n_threads)

    val_loader = Loader('val', val_dataset, batch_size=settings.batch_size, shuffle=False, drop_last=True,
                        epoch_interval=1.0, training=False, num_workers=settings.n_threads)

    # models
    model = glunet_vgg16(global_corr_type='global_corr', normalize='relu_l2norm', normalize_features=True,
                         cyclic_consistency=True, local_decoder_type='OpticalFlowEstimatorResidualConnection',
                         global_decoder_type='CMDTopResidualConnection')
    # if Load pre-trained weights !
    if settings.initial_pretrained_model is not None:
        model.load_state_dict(torch.load(settings.initial_pretrained_model)['state_dict'])
        print('Initialised weights')
    print(colored('==> ', 'blue') + 'model created.')

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        model = MultiGPU(model)

    # Loss module

    sample_transfo = SynthecticAffHomoTPSTransfo(size_output_flow=settings.resizing_size, random_t=settings.random_t,
                                                 random_s=settings.random_s,
                                                 random_alpha=settings.random_alpha,
                                                 random_t_tps_for_afftps=settings.random_t_tps_for_afftps,
                                                 random_t_hom=settings.random_t_hom, random_t_tps=settings.random_t_tps,
                                                 transformation_types=settings.transformation_types,
                                                 parametrize_with_gaussian=settings.parametrize_with_gaussian
                                                 )

    # sample a base flow, at the same resizing size than the images.
    base_flow_generator = GetRandomSyntheticAffHomoTPSFlow(settings=settings, transfo_sampling_module=sample_transfo,
                                                           size_output_flow=settings.resizing_size)

    # sample an elastic flow, at the same resizing size than the images
    elastic_flow_generator = AddElasticTransformsV2(settings=settings, size_output_flow=settings.resizing_size,
                                                    max_nbr_perturbations=13, min_nbr_perturbations=5,
                                                    elastic_parameters=settings.elastic_parameters_ss,
                                                    max_sigma_mask=settings.elastic_parameters_ss['max_sigma_mask'],
                                                    min_sigma_mask=settings.elastic_parameters_ss['min_sigma_mask'])

    # the final synthetic sampled flow is a combination of the 2
    synthetic_flow_generator = CompositionOfFlowCreations(transforms_list=[base_flow_generator,
                                                                           elastic_flow_generator])

    # actual module responsable for creating the image triplet from the real image pair.
    triplet_creator = BatchedImageTripletCreation(settings, synthetic_flow_generator=synthetic_flow_generator,
                                                  compute_mask_zero_borders=settings.compute_mask_zero_borders,
                                                  output_size=settings.crop_size, crop_size=settings.crop_size)

    # batch processing module, creates the triplet,  apply appearance transformations and put all the inputs
    # to cuda as well as in the right format
    batch_processing = GLUNetWarpCUnsupervisedBatchPreprocessing(
        settings, apply_mask_zero_borders=settings.compute_mask_zero_borders, online_triplet_creator=triplet_creator,
        appearance_transform_source=None, appearance_transform_target=None,
        appearance_transform_target_prime=settings.appearance_transfo_target_prime)

    # actual objective is L1, with different weights for the different levels.
    objective = L1()
    weights_level_loss = [0.32, 0.08, 0.02, 0.01]
    loss_module_256 = MultiScaleFlow(level_weights=weights_level_loss[:2], loss_function=objective,
                                     downsample_gt_flow=True)
    loss_module = MultiScaleFlow(level_weights=weights_level_loss[2:], loss_function=objective,
                                 downsample_gt_flow=True)

    # actor
    glunet_actor = GLUNetWarpCUnsupervisedActor(model, objective=loss_module, objective_256=loss_module_256,
                                                batch_processing=batch_processing, loss_weight=settings.loss_weight,
                                                name_of_loss=settings.name_of_loss,
                                                compute_visibility_mask=settings.compute_visibility_mask,
                                                nbr_images_to_plot=settings.nbr_plot_images)

    # Optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=settings.lr, weight_decay=0.0004)

    # Scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=settings.step_size_scheduler, gamma=0.5)

    trainer = MatchingTrainer(glunet_actor, [train_loader, val_loader], optimizer, settings, lr_scheduler=scheduler,
                              make_initial_validation=True)

    trainer.train(settings.n_epochs, load_latest=True, fail_safe=True)




