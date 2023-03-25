from termcolor import colored
import torch.optim as optim
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
import torch
import os
import numpy as np


from utils_data.image_transforms import ArrayToTensor
from training.losses.probabilistic_warp_consistency_losses import ProbabilisticWarpConsistencyForGlobalCorr
from training.trainers.matching_trainer import MatchingTrainer
from utils_data.loaders import Loader
from training.actors.PWarpC_actor import ModelWithTripletAndPairWiseProbabilisticWarpConsistency
from utils_data.euler_wrapper import prepare_data
from training.actors.warp_consistency_utils.online_triplet_creation import BatchedImageTripletCreation
from training.actors.warp_consistency_utils.synthetic_flow_generation_from_pair_batch import \
    GetRandomSyntheticAffHomoTPSFlow, SynthecticAffHomoTPSTransfo
from training.actors.warp_consistency_actor_BaseNet import GLOCALNetWarpCUnsupervisedBatchPreprocessing
from datasets.semantic_matching_datasets.spair import SPairDataset
from training.losses.cross_entropy_supervised import SmoothCrossEntropy
from training.losses.cost_volume_losses.losses_with_keypoint_annotations import SparseSupervisedSmoothGT
from utils_data.augmentations.color_augmentation_torch import ColorJitter, RandomGaussianBlur
from models.semantic_matching_models.SFNet import SFNet
from utils_data.sampler import RandomSampler


def run(settings):
    settings.description = 'Default train settings for strongly-supervised PWarpC-SF-Net, finetuned on SPair-71K (' \
                           'from pfpascal).'
    settings.data_mode = 'local'
    settings.batch_size = 16  # fit in memory 24
    settings.n_threads = 8
    settings.multi_gpu = True
    settings.print_interval = 50
    settings.keep_last_checkpoints = 10
    settings.nbr_plot_images = 2
    settings.dataset_callback_fn = 'sample_new_items'  # use to resample image pair at each epoch

    settings.lr = 1e-5
    settings.scheduler_steps = [20, 35]
    settings.n_epochs = 40

    # network param
    # the output of the model is not softmaxed yet, need to apply the activation when computing the loss
    settings.activation_in_loss = 'stable_softmax'
    settings.contrastive_temperature = 1.0 / 50.0
    settings.initial_pretrained_model = os.path.join(settings.env.workspace_dir,
                                                     'train_settings/PWarpC/'
                                                     'train_strongly_supervised_PWarpC_SFNet_pfpascal/'
                                                     'SFNet_model_best.pth.tar')

    # for pw-bipath and pwarp-supervision losses
    settings.loss_type = 'pw_bipath_and_pwarp_supervision'
    settings.balance_pwarpc_losses = True
    settings.loss_module_name = 'LogProb'
    settings.loss_module_name_warp_sup = 'SmoothCrossEntropy'
    # visibility mask settings
    settings.apply_loss_on_top = True
    settings.top_percent = 0.2

    # balance pwarpc losses (pw-bipath ans pwarp-supervision) with keypoint loss
    settings.balance_pwarpc_and_kp_losses = True
    settings.final_loss_weights = {'triplet': 1.0, 'pairwise': 1.0}
    # better results on SPair are obtained when the losses are not balanced, and instead the kp_loss counts for more
    # but generalization to other datasets is not as good in that case, such as
    # settings.balance_pwarpc_and_kp_losses = False
    # settings.final_loss_weights = {'triplet': 1.0, 'pairwise': 3.0}

    # dataset parameters
    settings.dataset_img_size = 320
    # size of images outputted by the dataloader, also size of the sampled synthetic flow for triplet creation

    # synthetic transfo parameters
    settings.crop_size = 20*16
    # size of final images in the triplet, and of the corresponding synthetic flow relating target prime to target
    settings.parametrize_with_gaussian = False
    settings.transformation_types = ['hom', 'tps', 'afftps']
    settings.random_t = 0.25
    settings.random_s = 0.45
    settings.random_alpha = np.pi / 12
    settings.random_t_tps_for_afftps = 0.4
    settings.random_t_hom = 0.4
    settings.random_t_tps = 0.4
    settings.proba_horizontal_flip = 0.15

    # 1. Define datasets
    # images outputted by the dataset must have size equal to settings.dataset_img_size, the same size
    # than the synthetic transformation
    flow_transform = transforms.Compose([ArrayToTensor()])  # just put channels first and put it to float
    image_transforms = transforms.Compose([ArrayToTensor(get_float=True)])  # just put channels first

    if settings.data_mode == 'euler':
        prepare_data(settings.env.spair_tar, mode=settings.data_mode)

    spair_cfg = {'augment_with_crop': True, 'crop_size': [settings.dataset_img_size, settings.dataset_img_size],
                 'augment_with_flip': True, 'proba_of_image_flip': 0.0, 'proba_of_batch_flip': 0.5,
                 'output_image_size': [settings.dataset_img_size, settings.dataset_img_size],
                 'pad_to_same_shape': False, 'output_flow_size': [settings.dataset_img_size, settings.dataset_img_size]}
    train_dataset = SPairDataset(settings.env.spair, output_image_size=settings.dataset_img_size,
                                 source_image_transform=image_transforms, target_image_transform=image_transforms,
                                 flow_transform=flow_transform, split='trn', training_cfg=spair_cfg)

    spair_cfg['augment_with_crop'] = False
    spair_cfg['augment_with_flip'] = False
    val_dataset = SPairDataset(settings.env.spair, output_image_size=settings.dataset_img_size,
                               source_image_transform=image_transforms, target_image_transform=image_transforms,
                               flow_transform=flow_transform, split='val', training_cfg=spair_cfg)

    # 2. Define dataloaders
    train_loader = Loader('train', train_dataset, batch_size=settings.batch_size,
                          sampler=RandomSampler(train_dataset, 15000),
                          drop_last=False, training=True, num_workers=settings.n_threads)

    val_loader = Loader('val', val_dataset, batch_size=settings.batch_size, shuffle=False,
                        epoch_interval=1.0, training=False, num_workers=settings.n_threads)

    # 3. Define model
    model = SFNet(forward_pass_strategy='corr_prediction_no_kernel')
    if settings.initial_pretrained_model is not None:
        checkpoint = torch.load(settings.initial_pretrained_model)['state_dict']
        model.load_state_dict(checkpoint)
    print(colored('==> ', 'blue') + 'model created.')

    # 4. Define batch processing (creates the triplet from the original image pair, then put them to the right
    # format to be processed by the network, ect)

    # synthetic transformation sampling modules (for synthetic flow W in the paper)
    # the flow outputted (size_output_flow) needs to be the same size than the images provided by the dataloader
    # (ie settings.dataset_img_size). Then the flow is applied to the target to create the target prime image.
    # Then all three images (souce, target and target prime) along with the flow are centered cropped
    # (resulting in size 'settings.crop_size').
    sample_transfo = SynthecticAffHomoTPSTransfo(size_output_flow=settings.dataset_img_size, random_t=settings.random_t,
                                                 random_s=settings.random_s,
                                                 random_alpha=settings.random_alpha,
                                                 random_t_tps_for_afftps=settings.random_t_tps_for_afftps,
                                                 random_t_hom=settings.random_t_hom, random_t_tps=settings.random_t_tps,
                                                 transformation_types=settings.transformation_types,
                                                 parametrize_with_gaussian=settings.parametrize_with_gaussian,
                                                 proba_horizontal_flip=settings.proba_horizontal_flip)

    # synthetic flow sampling module. The outputted flow has size settings.dataset_img_size, which is the
    # same size than the image pair given by the dataloader
    synthetic_flow_generator = GetRandomSyntheticAffHomoTPSFlow(settings=settings, transfo_sampling_module=sample_transfo,
                                                                size_output_flow=settings.dataset_img_size)

    # actual module responsable for creating the image triplet from the real image pair and the previously
    # sampled synthetic flow
    batched_triplet_creator = BatchedImageTripletCreation(settings, synthetic_flow_generator=synthetic_flow_generator,
                                                          compute_mask_zero_borders=False,
                                                          output_size=settings.crop_size, crop_size=settings.crop_size)

    batch_image_transform = transforms.Compose([transforms.RandomGrayscale(p=0.2),
                                                ColorJitter(brightness=0.3, contrast=0.3,
                                                            saturation=0.3, hue=0.5 / 3.14, invert_channel=False),
                                                RandomGaussianBlur(sigma=(0.2, 2.0), probability=0.2)])

    batch_image_prime_transform = transforms.Compose([transforms.RandomGrayscale(p=0.2),
                                                      ColorJitter(brightness=0.4, contrast=0.4,
                                                                  saturation=0.4, hue=0.5 / 3.14, invert_channel=True),
                                                      RandomGaussianBlur(sigma=(0.2, 2.0), probability=0.2)])

    # batch processing module, creates the triplet,  apply appearance transformations and put all the inputs
    # to cuda as well as in the right format
    batch_processing = GLOCALNetWarpCUnsupervisedBatchPreprocessing(
        settings, apply_mask_zero_borders=False, apply_mask=True, normalize_images=True,  # imagenet weights
        online_triplet_creator=batched_triplet_creator, appearance_transform_source=batch_image_transform,
        appearance_transform_target=batch_image_transform,
        appearance_transform_target_prime=batch_image_prime_transform)

    # 5. Define loss module
    # kp objective
    kp_objective = SmoothCrossEntropy()
    kp_loss_module = SparseSupervisedSmoothGT(objective=kp_objective, activation=settings.activation_in_loss,
                                              temperature=settings.contrastive_temperature)

    # unsupervised objective
    pwarpc_loss_module = ProbabilisticWarpConsistencyForGlobalCorr(
        occlusion_handling=False, temperature=settings.contrastive_temperature, activation=settings.activation_in_loss,
        name_of_loss=settings.loss_type, balance_losses=settings.balance_pwarpc_losses,
        loss_name=settings.loss_module_name, loss_name_warp_sup=settings.loss_module_name_warp_sup,
        apply_loss_on_top=settings.apply_loss_on_top, top_percent=settings.top_percent)

    actor = ModelWithTripletAndPairWiseProbabilisticWarpConsistency(
        model, triplet_objective=pwarpc_loss_module, batch_processing=batch_processing,
        pairwise_objective=kp_loss_module, balance_triplet_and_pairwise_losses=settings.balance_pwarpc_and_kp_losses,
        loss_weights=settings.final_loss_weights,
        compute_both_directions=True, nbr_images_to_plot=settings.nbr_plot_images)

    # 6. Define optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=settings.lr)

    # 7. Define scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=settings.scheduler_steps, gamma=0.5)

    # 8. Define trainer
    trainer = MatchingTrainer(actor, [train_loader, val_loader], optimizer, settings, lr_scheduler=scheduler,
                              make_initial_validation=True)

    trainer.train(settings.n_epochs, load_latest=True, fail_safe=True)




