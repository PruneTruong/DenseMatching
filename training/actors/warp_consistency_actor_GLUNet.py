import os
import torch
import torch.nn.functional as F
from packaging import version

from training.losses.basic_losses import real_metrics, realEPE
from admin.stats import merge_dictionaries
from .base_actor import BaseActor
from utils_flow.flow_and_mapping_operations import unormalise_and_convert_mapping_to_flow
from training.plot.plot_warp_consistency import plot_flows_warpc
from admin.multigpu import is_multi_gpu
from training.losses.warp_consistency_losses import WBipathLoss, weights_self_supervised_and_unsupervised
from .batch_processing import pre_process_image_glunet


class GetGLUNetPredictions:
    def __call__(self, source_image, target_image, net, device, pre_process_data, *args, **kwargs):
        net.eval()
        # torch.with_no_grad:
        b, _, h, w = target_image.shape
        if pre_process_data:
            source_image, source_image_256 = pre_process_image_glunet(source_image, device)
            target_image, target_image_256 = pre_process_image_glunet(target_image, device)

        net_ = net.module if is_multi_gpu(net) else net
        im_source_pyr = net_.pyramid(source_image, eigth_resolution=True)
        im_target_pyr = net_.pyramid(target_image, eigth_resolution=True)
        im_source_pyr_256 = net_.pyramid(source_image_256)
        im_target_pyr_256 = net_.pyramid(target_image_256)

        output_target_to_source_256, output_target_to_source = net(target_image, source_image,
                                                                   target_image_256, source_image_256,
                                                                   im_target_pyr, im_source_pyr,
                                                                   im_target_pyr_256, im_source_pyr_256)
        estimated_flow_target_to_source = output_target_to_source['flow_estimates'][-1]
        output_source_to_target_256, output_source_to_target = net(source_image, target_image,
                                                                   source_image_256, target_image_256,
                                                                   im_source_pyr, im_target_pyr,
                                                                   im_source_pyr_256, im_target_pyr_256)
        estimated_flow_source_to_target = output_source_to_target['flow_estimates'][-1]

        estimated_flow_target_to_source = F.interpolate(estimated_flow_target_to_source, (h, w),
                                                        mode='bilinear', align_corners=False)
        estimated_flow_source_to_target = F.interpolate(estimated_flow_source_to_target, (h, w),
                                                        mode='bilinear', align_corners=False)

        return estimated_flow_target_to_source, estimated_flow_source_to_target


class GLUNetWarpCUnsupervisedBatchPreprocessing:
    """ Class responsible for processing the mini-batch to create the desired training inputs for GLUNet
     based networks, when training unsupervised using warp consistency.

     Particularly, from the source and target image pair (which are usually not related by a known ground-truth flow),
     the online_triplet_creator creates an image triplet (source, target and target_prime) where the target prime
     image is related to the target by a known synthetically and randomly generated flow field.
     The final image triplet is obtained by cropping central patches in the three images.
     Optionally, two image triplets can be created with different synthetic flow fields, resulting in different
     target_prime images, to apply different losses on each. This is done in 'online_triplet_creator'.

     Here, appearance augmentations can optionally be added, the images are normalized and the
     ground-truth flow field as well as mask used for training are processed.

     In that case, the ground-truth flow field used during training is the synthetic flow field relating
     target prime to target.
     The training inputs are also created at resolution 256x256 for training the L-Net.
    """

    def __init__(self, settings, online_triplet_creator, apply_mask=False, apply_mask_zero_borders=False,
                 sparse_ground_truth=False, mapping=False, appearance_transform_source=None,
                 appearance_transform_target=None, appearance_transform_target_prime=None,
                 appearance_transform_target_prime_ss=None):
        """
        Args:
            settings: settings
            online_triplet_creator: class responsible for the creation of the image triplet used in the warp
                                    consistency graph, from an image pair.
            apply_mask: apply ground-truth correspondence mask for loss computation?
            apply_mask_zero_borders: apply mask zero borders (equal to 0 at black borders in target_prime image) for loss
                                     computation? This is specifically useful when the target image prime is
                                     computed by warping the target with a synthetic transformation.
                                     It can have many black borders due to the warp. That can cause instability
                                     during training, if the loss is applied also in the black areas (where the network
                                     cannot infer any correct predictions).
            sparse_ground_truth: if ground-truth between source and target image is known, indicates that it is a
                                 sparse ground-truth.
            mapping: load correspondence map instead of flow field?
            appearance_transform_source: appearance augmentation applied to batched source image tensors
                                        (before normalizing)
            appearance_transform_target: appearance augmentation applied to batched target image tensors
                                        (before normalizing)
            appearance_transform_target_prime: appearance augmentation applied to batched target prime image tensors
                                               (before normalizing)
            appearance_transform_target_prime_ss: appearance augmentation applied to batched target prime ss image
                                                  tensors (when it exists, here ss stands for
                                                  self-supervised/warp-supervision)
        """

        assert not (apply_mask and apply_mask_zero_borders), \
            'apply_mask and apply_mask_zero_borders cannot both be applied at the same time, choose only one'

        self.apply_mask = apply_mask
        self.apply_mask_zero_borders = apply_mask_zero_borders
        self.sparse_ground_truth = sparse_ground_truth
        self.mapping = mapping

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_triplet_creator = online_triplet_creator

        self.appearance_transform_source = appearance_transform_source
        self.appearance_transform_target = appearance_transform_target
        self.appearance_transform_target_prime = appearance_transform_target_prime
        self.appearance_transform_target_prime_ss = appearance_transform_target_prime_ss

    def __call__(self, mini_batch, net=None, training=False,  *args, **kwargs):

        """
        Args:
            mini_batch: The mini batch input data, should at least contain the fields 'source_image', 'target_image'
            training: bool indicating if we are in training or evaluation mode
            net: network
        Returns:
            mini_batch: output data block with at least the fields 'source_image', 'target_image',
                        'target_image_prime', 'flow_map', 'mask', 'correspondence_mask',
                        'source_image_256', 'target_image_256', 'target_image_prime_256', 'flow_map_256',  'mask_256'.

                        'flow_map' here is the ground-truth synthetic flow relating the target_image_prime to the
                        target. 'correspondence_mask' is the in-view flow regions.
                        'mask' is the mask of where the loss will be applied in the target image prime frame.
                        Same applies for the 256x256 tensors.

                         If ground-truth is known between source and target image, will contain the fields
                        'flow_map_target_to_source', 'correspondence_mask_target_to_source'.
        """

        # create the image triplet from the image pair
        if self.online_triplet_creator is not None:
            mini_batch = self.online_triplet_creator(mini_batch, train=training, net=net)

        # add appearance augmentation to all three images
        # appearance transform on tensor arrays, which are already B, C, H, W
        if self.appearance_transform_source is not None:
            mini_batch['source_image'] = self.appearance_transform_source(mini_batch['source_image'])

        if self.appearance_transform_target is not None:
            mini_batch['target_image'] = self.appearance_transform_target(mini_batch['target_image'])

        if self.appearance_transform_target_prime is not None:
            mini_batch['target_image_prime'] = self.appearance_transform_target_prime(mini_batch['target_image_prime'])

        # pre process data (normalize the images and creates input for L-Net of GLU-Net)
        source_image, source_image_256 = pre_process_image_glunet(mini_batch['source_image'], self.device)
        target_image, target_image_256 = pre_process_image_glunet(mini_batch['target_image'], self.device)
        target_image_prime, target_image_prime_256 = pre_process_image_glunet(mini_batch['target_image_prime'],
                                                                              self.device)

        if self.mapping:
            mapping_gt = mini_batch['correspondence_map_pyro'][-1].to(self.device)
            flow_gt = unormalise_and_convert_mapping_to_flow(mapping_gt.permute(0,3,1,2))
        else:
            flow_gt = mini_batch['flow_map'].to(self.device)  # between target prime and target
        if flow_gt.shape[1] != 2:
            flow_gt.permute(0, 3, 1, 2)
        bs, _, h_original, w_original = flow_gt.shape

        # ground-truth flow field for 256x256 module
        flow_gt_256 = F.interpolate(flow_gt, (256, 256), mode='bilinear', align_corners=False)
        flow_gt_256[:, 0, :, :] *= 256.0 / float(w_original)
        flow_gt_256[:, 1, :, :] *= 256.0 / float(h_original)

        mask = None
        mask_256 = None
        if self.apply_mask_zero_borders:
            if 'mask_zero_borders' not in mini_batch.keys():
                raise ValueError('Mask zero borders not in mini batch, check arguments to triplet creator')
            mask = mini_batch['mask_zero_borders'].to(self.device)
        elif self.apply_mask:
            mask = mini_batch['correspondence_mask'].to(self.device)
        if mask is not None and (mask.shape[1] != h_original or mask.shape[2] != w_original):
            # mask_gt does not have the proper shape
            mask = F.interpolate(mask.float().unsqueeze(1), (h_original, w_original), mode='bilinear',
                                 align_corners=False).squeeze(1).floor()  # bxhxw
            mask = mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask.byte()

        if mask is not None:
            mask_256 = F.interpolate(mask.unsqueeze(1).float(), (256, 256), mode='bilinear',
                                     align_corners=False).squeeze(1).floor()  # bx256x256, flooring
            mask_256 = mask_256.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask_256.byte()

        mini_batch['source_image'] = source_image
        mini_batch['target_image'] = target_image
        mini_batch['target_image_prime'] = target_image_prime
        mini_batch['source_image_256'] = source_image_256
        mini_batch['target_image_256'] = target_image_256
        mini_batch['target_image_prime_256'] = target_image_prime_256

        mini_batch['correspondence_mask'] = mini_batch['correspondence_mask'].to(self.device)
        mini_batch['mask'] = mask
        mini_batch['flow_map'] = flow_gt  # between target_prime and target
        mini_batch['mask_256'] = mask_256
        mini_batch['flow_map_256'] = flow_gt_256  # between target_prime and target

        # different flows for unsupervised (w-bipath) and self-supervised (warp-supervision)
        if 'target_image_prime_ss' in list(mini_batch.keys()):
            if self.appearance_transform_target_prime_ss is not None:
                mini_batch['target_image_prime_ss'] = self.appearance_transform_target_prime_ss(
                    mini_batch['target_image_prime_ss'])
            target_image_prime_ss, target_image_prime_ss_256 = pre_process_image_glunet(
                mini_batch['target_image_prime_ss'], self.device)

            # get synthetic flow field between target image prime ss and target image
            flow_gt_ss = mini_batch['flow_map_ss'].to(self.device)
            if flow_gt_ss.shape[1] != 2:
                flow_gt_ss.permute(0, 3, 1, 2)
            bs, _, h_original, w_original = flow_gt.shape

            flow_gt_ss_256 = F.interpolate(flow_gt_ss, (256, 256), mode='bilinear', align_corners=False)
            flow_gt_ss_256[:, 0, :, :] *= 256.0 / float(w_original)
            flow_gt_ss_256[:, 1, :, :] *= 256.0 / float(h_original)

            # save to mini batch
            mini_batch['target_image_prime_ss'] = target_image_prime_ss
            mini_batch['target_image_prime_ss_256'] = target_image_prime_ss_256
            mini_batch['flow_map_ss'] = flow_gt_ss  # between target_prime and target
            mini_batch['flow_map_ss_256'] = flow_gt_ss_256  # between target_prime and target

            # get correspondence mask and training mask
            mask = None
            mask_256 = None
            if self.apply_mask_zero_borders:
                if 'mask_zero_borders_ss' not in mini_batch.keys():
                    raise ValueError('Mask zero borders ss not in mini batch, check arguments to triplet creator')
                mask = mini_batch['mask_zero_borders_ss'].to(self.device)
            elif self.apply_mask:
                mask = mini_batch['correspondence_mask_ss'].to(self.device)
            if mask is not None and (mask.shape[1] != h_original or mask.shape[2] != w_original):
                # mask_gt does not have the proper shape
                mask = F.interpolate(mask.float().unsqueeze(1), (h_original, w_original), mode='bilinear',
                                     align_corners=False).squeeze(1).floor()  # bxhxw
                mask = mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask.byte()

            if mask is not None:
                mask_256 = F.interpolate(mask.unsqueeze(1).float(), (256, 256), mode='bilinear',
                                         align_corners=False).squeeze(1).floor()  # bx256x256, rounding
                mask_256 = mask_256.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask_256.byte()

            mini_batch['correspondence_mask_ss'] = mini_batch['correspondence_mask_ss'].to(self.device)
            mini_batch['mask_ss'] = mask
            mini_batch['mask_ss_256'] = mask_256

        # if ground-truth between source and target exists, will be used for validation
        if 'flow_map_target_to_source' in list(mini_batch.keys()):
            # gt between target and source (what we are trying to estimate during training unsupervised)
            mini_batch['flow_map_target_to_source'] = mini_batch['flow_map_target_to_source'].to(self.device)
            mini_batch['correspondence_mask_target_to_source'] = mini_batch['correspondence_mask_target_to_source'].to(self.device)

        return mini_batch


class GLUNetWarpCUnsupervisedActor(BaseActor):
    """Actor for training unsupervised GLU-Net based networks with the warp consistency objective."""
    def __init__(self, net, objective, objective_256, batch_processing, name_of_loss, loss_weight=None,
                 apply_constant_flow_weights=False, detach_flow_for_warping=True, compute_visibility_mask=False,
                 best_val_epe=False, nbr_images_to_plot=1, semantic_evaluation=False):
        """
        Args:
            net: The network to train
            objective: The loss function to apply to the H-Net
            objective_256: The loss function to apply to the L-Net
            batch_processing: A processing class which performs the necessary processing of the batched data.
                              Corresponds to creating the image triplet here.
            name_of_loss: 'warp_supervision_and_w_bipath' or 'w_bipath' or 'warp_supervision'
            loss_weight: weights used to balance w_bipath and warp_supervision.
            apply_constant_flow_weights: bool, otherwise, balance both losses according to given weights.
            detach_flow_for_warping: bool
            compute_visibility_mask: bool
            best_val_epe: use AEPE for best value, instead can use PCK
            nbr_images_to_plot: number of images to plot per epoch
            semantic_evaluation: bool, to adapt the thresholds used in the PCK computations
        """
        super().__init__(net, objective, batch_processing)
        self.apply_constant_flow_weights = apply_constant_flow_weights
        self.best_val_epe = best_val_epe
        self.semantic_evaluation = semantic_evaluation
        loss_weight_default = {'w_bipath': 1.0, 'warp_supervision': 1.0,
                               'w_bipath_constant': 1.0, 'warp_supervision_constant': 1.0,
                               'cc_mask_alpha_1': 0.03, 'cc_mask_alpha_2': 0.5}
        self.loss_weight = loss_weight_default
        if loss_weight is not None:
            self.loss_weight.update(loss_weight)

        self.name_of_loss = name_of_loss
        self.objective_256 = objective_256

        self.compute_visibility_mask = compute_visibility_mask
        # define the loss computation modules
        if 'w_bipath' in name_of_loss:
            self.unsupervised_objective = WBipathLoss(
                objective, loss_weight, detach_flow_for_warping,
                compute_cyclic_consistency=self.compute_visibility_mask,
                alpha_1=self.loss_weight['cc_mask_alpha_1'], alpha_2=self.loss_weight['cc_mask_alpha_2'])
            self.unsupervised_objective_256 = WBipathLoss(
                objective_256, loss_weight, detach_flow_for_warping,
                compute_cyclic_consistency=self.compute_visibility_mask, alpha_1=self.loss_weight['cc_mask_alpha_1'],
                alpha_2=self.loss_weight['cc_mask_alpha_2'])
        elif 'warp_supervision' not in name_of_loss:
            raise ValueError('The name of the loss is not correct, you chose {}'.format(self.name_of_loss))

        self.nbr_images_to_plot = nbr_images_to_plot

    def __call__(self, mini_batch, training):
        """
        args:
            mini_batch: The mini batch input data, should at least contain the fields 'source_image', 'target_image',
                        'target_image_prime', 'flow_map', 'mask', 'correspondence_mask',
                        'source_image_256', 'target_image_256', 'target_image_prime_256',
                        'flow_map_256',  'mask_256'.

                        'flow_map' here is the ground-truth synthetic flow relating the target_image_prime to the
                        target. 'correspondence_mask' is the in-view flow regions.
                        'mask' is the mask of where the loss will be applied in the target image prime frame.
                        Same applies for the 256x256 tensors.

                         If ground-truth is known between source and target image, will contain the fields
                        'flow_map_target_to_source', 'correspondence_mask_target_to_source'.
            training: bool indicating if we are in training or evaluation mode
        returns:
            loss: the training loss
            stats: dict containing detailed losses
        """
        # Run network
        epoch = mini_batch['epoch']
        iter = mini_batch['iter']
        mini_batch = self.batch_processing(mini_batch, net=self.net, training=training)
        b, _, h, w = mini_batch['flow_map'].shape
        b, _, h_256, w_256 = mini_batch['flow_map_256'].shape

        # extract features to avoid recomputing them
        net = self.net.module if is_multi_gpu(self.net) else self.net
        im_source_pyr = net.pyramid(mini_batch['source_image'], eigth_resolution=True)
        im_target_pyr = net.pyramid(mini_batch['target_image'], eigth_resolution=True)
        im_target_prime_pyr = net.pyramid(mini_batch['target_image_prime'], eigth_resolution=True)
        im_source_pyr_256 = net.pyramid(mini_batch['source_image_256'])
        im_target_pyr_256 = net.pyramid(mini_batch['target_image_256'])
        im_target_prime_pyr_256 = net.pyramid(mini_batch['target_image_prime_256'])
        if 'target_image_prime_ss' in list(mini_batch.keys()):
            im_target_prime_ss_pyr = net.pyramid(mini_batch['target_image_prime_ss'], eigth_resolution=True)
            im_target_prime_ss_pyr_256 = net.pyramid(mini_batch['target_image_prime_ss_256'])

        # Compute flows
        # just for validation or plotting
        if not training or iter < self.nbr_images_to_plot:
            output_target_to_source_256, output_flow_target_to_source = self.net(
                mini_batch['target_image'], mini_batch['source_image'],
                mini_batch['target_image_256'], mini_batch['source_image_256'],
                im_target_pyr, im_source_pyr, im_target_pyr_256, im_source_pyr_256)
            estimated_flow_target_to_source_256 = output_target_to_source_256['flow_estimates']
            estimated_flow_target_to_source = output_flow_target_to_source['flow_estimates']

        estimated_flow_target_prime_to_target_directly = None
        estimated_flow_target_prime_to_target_directly_256 = None
        if not training or 'warp_supervision' in self.name_of_loss or iter < self.nbr_images_to_plot:
            if 'target_image_prime_ss' in list(mini_batch.keys()):
                # if it use different target prime for self-supervised and unsupervised
                output_target_prime_to_target_directly_256, output_target_prime_to_target_directly = \
                    self.net(mini_batch['target_image_prime_ss'], mini_batch['target_image'],
                             mini_batch['target_image_prime_ss_256'], mini_batch['target_image_256'],
                             im_target_prime_ss_pyr, im_target_pyr, im_target_prime_ss_pyr_256, im_target_pyr_256)
            else:
                output_target_prime_to_target_directly_256, output_target_prime_to_target_directly = \
                    self.net(mini_batch['target_image_prime'], mini_batch['target_image'],
                             mini_batch['target_image_prime_256'], mini_batch['target_image_256'],
                             im_target_prime_pyr, im_target_pyr, im_target_prime_pyr_256, im_target_pyr_256)
            estimated_flow_target_prime_to_target_directly_256 = output_target_prime_to_target_directly_256['flow_estimates']
            estimated_flow_target_prime_to_target_directly = output_target_prime_to_target_directly['flow_estimates']

        # Compute losses
        un_loss_256, un_loss_o, ss_loss_o, ss_loss_256 = 0.0, 0.0, 0.0, 0.0
        if 'warp_supervision' in self.name_of_loss:
            # warp supervision only
            if 'target_image_prime_ss' in list(mini_batch.keys()):
                ss_loss_o, ss_stats_o = self.objective(estimated_flow_target_prime_to_target_directly,
                                                       mini_batch['flow_map_ss'], mask=mini_batch['mask_ss'])
                ss_loss_256, ss_stats_256 = self.objective_256(estimated_flow_target_prime_to_target_directly_256,
                                                               mini_batch['flow_map_ss_256'],
                                                               mask=mini_batch['mask_ss_256'])
            else:
                ss_loss_o, ss_stats_o = self.objective(estimated_flow_target_prime_to_target_directly,
                                                       mini_batch['flow_map'], mask=mini_batch['mask'])
                ss_loss_256, ss_stats_256 = self.objective_256(estimated_flow_target_prime_to_target_directly_256,
                                                               mini_batch['flow_map_256'], mask=mini_batch['mask_256'])

        if 'w_bipath' in self.name_of_loss:
            un_stats_256, un_stats_o, output_un_256, output_un = {}, {}, {}, {}
            # compute the unsupervised loss

            output_target_prime_to_source_256, output_target_prime_to_source = \
                self.net(mini_batch['target_image_prime'], mini_batch['source_image'],
                         mini_batch['target_image_prime_256'], mini_batch['source_image_256'], im_target_prime_pyr,
                         im_source_pyr, im_target_prime_pyr_256, im_source_pyr_256)
            estimated_flow_target_prime_to_source_256 = output_target_prime_to_source_256['flow_estimates']
            estimated_flow_target_prime_to_source = output_target_prime_to_source['flow_estimates']

            output_source_to_target_256, output_source_to_target = self.net(
                mini_batch['source_image'], mini_batch['target_image'],
                mini_batch['source_image_256'], mini_batch['target_image_256'], im_source_pyr, im_target_pyr,
                im_source_pyr_256, im_target_pyr_256)
            estimated_flow_source_to_target_256 = output_source_to_target_256['flow_estimates']
            estimated_flow_source_to_target = output_source_to_target['flow_estimates']

            un_loss_o_, un_stats_o_, output_un_ = self.unsupervised_objective(
                mini_batch['flow_map'], mini_batch['mask'], estimated_flow_target_prime_to_source,
                estimated_flow_source_to_target)

            un_loss_256_, un_stats_256_, output_un_256_ = self.unsupervised_objective_256(
                mini_batch['flow_map_256'], mini_batch['mask_256'], estimated_flow_target_prime_to_source_256,
                estimated_flow_source_to_target_256)

            un_loss_o += un_loss_o_
            un_loss_256 += un_loss_256_
            un_stats_256 = merge_dictionaries(list_dict=[un_stats_256, un_stats_256_])
            un_stats_o = merge_dictionaries(list_dict=[un_stats_o, un_stats_o_])
            output_un = merge_dictionaries(list_dict=[output_un, output_un_])
            output_un_256 = merge_dictionaries(list_dict=[output_un_256, output_un_256_])

        # weights the losses
        if self.name_of_loss == 'warp_supervision':
            # warp supervision only
            loss = ss_loss_o + ss_loss_256

            # Log stats
            stats = merge_dictionaries([ss_stats_o, ss_stats_256], name=['ori', '256'])
            stats['Loss_H_Net/total'] = ss_loss_o.item()
            stats['Loss_L_Net/total'] = ss_loss_256.item()
            stats['Loss/total'] = loss.item()
            output_un = {'estimated_flow_target_prime_to_target_through_composition':
                             estimated_flow_target_prime_to_target_directly}
        elif self.name_of_loss == 'w_bipath':
            # w-bipath only
            loss = un_loss_o + un_loss_256
            stats = merge_dictionaries([un_stats_o, un_stats_256], name=['ori', '256'])
            stats['Loss_H_Net/total'] = un_loss_o.item()
            stats['Loss_L_Net/total'] = un_loss_256.item()
            stats['Loss/total'] = loss.item()
        else:
            # merge stats and losses
            stats_256 = merge_dictionaries([un_stats_256, ss_stats_256], name=['unsup', 'sup'])
            loss_256, stats_256 = weights_self_supervised_and_unsupervised(ss_loss_256, un_loss_256,
                                                                           stats_256, self.loss_weight,
                                                                           self.apply_constant_flow_weights)

            stats_o = merge_dictionaries([un_stats_o, ss_stats_o], name=['unsup', 'sup'])
            loss_o, stats_o = weights_self_supervised_and_unsupervised(ss_loss_o, un_loss_o,
                                                                       stats_o, self.loss_weight,
                                                                       self.apply_constant_flow_weights)

            loss = loss_o + loss_256

            # Log stats
            stats = merge_dictionaries([stats_o, stats_256], name=['ori', '256'])
            stats['Loss_H_Net/total'] = loss_o.item()
            stats['Loss_L_Net/total'] = loss_256.item()
            stats['Loss/total'] = loss.item()

        # Calculates validation stats
        if not training:
            if 'flow_map_target_to_source' in list(mini_batch.keys()):
                mask_gt_target_to_source = mini_batch['correspondence_mask_target_to_source']
                flow_gt_target_to_source = mini_batch['flow_map_target_to_source']
                if self.semantic_evaluation:
                    thresh_1, thresh_2, thresh_3 = max(flow_gt_target_to_source.shape[-2:]) * 0.05, \
                    max(flow_gt_target_to_source.shape[-2:]) * 0.1, max(flow_gt_target_to_source.shape[-2:]) * 0.15
                else:
                    thresh_1, thresh_2, thresh_3 = 1.0, 3.0, 5.0
                for index_reso in range(len(estimated_flow_target_to_source)):
                    EPE, PCK_1, PCK_3, PCK_5 = real_metrics(estimated_flow_target_to_source[-(index_reso + 1)],
                                                            flow_gt_target_to_source, mask_gt_target_to_source,
                                                            thresh_1=thresh_1, thresh_2=thresh_2, thresh_3=thresh_3)

                    stats['EPE_target_to_source_reso_{}/EPE'.format(index_reso)] = EPE.item()
                    stats['PCK_{}_target_to_source_reso_{}/EPE'.format(thresh_1, index_reso)] = PCK_1.item()
                    stats['PCK_{}_target_to_source_reso_{}/EPE'.format(thresh_2, index_reso)] = PCK_3.item()
                    stats['PCK_{}_target_to_source_reso_{}/EPE'.format(thresh_3, index_reso)] = PCK_5.item()

                b, _, h_original, w_original = mini_batch['flow_map'].shape
                for index_reso in range(len(estimated_flow_target_to_source_256)):
                    EPE, PCK_1, PCK_3, PCK_5 = real_metrics(estimated_flow_target_to_source_256[-(index_reso + 1)],
                                                            flow_gt_target_to_source, mask_gt_target_to_source,
                                                            thresh_1=thresh_1, thresh_2=thresh_2, thresh_3=thresh_3,
                                                            ratio_x=float(w_original) / float(256.0),
                                                            ratio_y=float(w_original) / float(256.0))

                    stats['EPE_target_to_source_reso_LNet_{}/EPE'.format(index_reso)] = EPE.item()
                    stats['PCK_{}_target_to_source_reso_LNet_{}/EPE'.format(thresh_1, index_reso)] = PCK_1.item()
                    stats['PCK_{}_target_to_source_reso_LNet_{}/EPE'.format(thresh_2, index_reso)] = PCK_3.item()
                    stats['PCK_{}_target_to_source_reso_LNet_{}/EPE'.format(thresh_3, index_reso)] = PCK_5.item()

            for index_reso in range(len(estimated_flow_target_prime_to_target_directly)):
                if 'target_image_prime_ss' in list(mini_batch.keys()):
                    EPE = realEPE(estimated_flow_target_prime_to_target_directly[-(index_reso + 1)],
                                  mini_batch['flow_map_ss'], mini_batch['correspondence_mask_ss'])
                else:
                    EPE = realEPE(estimated_flow_target_prime_to_target_directly[-(index_reso + 1)],
                                  mini_batch['flow_map'], mini_batch['correspondence_mask'])
                stats['EPE_target_prime_to_target_reso_direct_{}/EPE'.format(index_reso)] = EPE.item()

            if 'estimated_flow_target_prime_to_target_through_composition' in list(output_un.keys()):
                for index_reso in range(len(output_un['estimated_flow_target_prime_to_target_through_composition'])):
                    EPE = realEPE(output_un['estimated_flow_target_prime_to_target_through_composition'][-(index_reso + 1)],
                                  mini_batch['flow_map'], mini_batch['correspondence_mask'])

                    stats['EPE_target_prime_to_target_reso_composition_{}/EPE'.format(index_reso)] = EPE.item()

            if 'flow_map_target_to_source' in list(mini_batch.keys()):
                if self.best_val_epe:
                    stats['best_value'] = stats['EPE_target_to_source_reso_0/EPE']
                else:
                    stats['best_value'] = - stats['PCK_{}_target_to_source_reso_0/EPE'.format(thresh_2)]
            else:
                stats['best_value'] = stats['EPE_target_prime_to_target_reso_composition_0/EPE']

        # plot images
        if iter < self.nbr_images_to_plot:
            training_or_validation = 'train' if training else 'val'
            base_save_dir = os.path.join(mini_batch['settings'].env.workspace_dir, mini_batch['settings'].project_path,
                                         'plot', training_or_validation)
            if not os.path.isdir(base_save_dir):
                os.makedirs(base_save_dir)

            if self.name_of_loss == 'warp_supervision':
                plot_flows_warpc(base_save_dir, 'un_epoch{}_batch{}_reso_520'.format(epoch, iter), h, w,
                                 image_source=mini_batch['source_image'],
                                 image_target=mini_batch['target_image'],
                                 image_target_prime=mini_batch['target_image_prime'],
                                 estimated_flow_target_to_source=None,
                                 estimated_flow_target_prime_to_source=None,
                                 estimated_flow_target_prime_to_target=None,
                                 estimated_flow_target_prime_to_target_directly=
                                 estimated_flow_target_prime_to_target_directly[-1],
                                 gt_flow_target_prime_to_target=mini_batch['flow_map'],
                                 sparse=mini_batch['sparse'], mask=mini_batch['mask'])
            else:
                plot_flows_warpc(base_save_dir, 'un_epoch{}_batch{}_reso_520'.format(epoch, iter), h, w,
                                 image_source=mini_batch['source_image'],
                                 image_target=mini_batch['target_image'],
                                 image_target_prime=mini_batch['target_image_prime'],
                                 estimated_flow_target_to_source=estimated_flow_target_to_source[-1],
                                 estimated_flow_target_prime_to_source=estimated_flow_target_prime_to_source[-1] if
                                 estimated_flow_target_prime_to_source is not None else None,
                                 estimated_flow_target_prime_to_target=
                                 output_un['estimated_flow_target_prime_to_target_through_composition'][-1],
                                 estimated_flow_target_prime_to_target_directly=
                                 estimated_flow_target_prime_to_target_directly[-1],
                                 gt_flow_target_prime_to_target=mini_batch['flow_map'],
                                 gt_flow_target_to_source=mini_batch['flow_map_target_to_source'] if
                                 'flow_map_target_to_source' in list(mini_batch.keys()) else None,
                                 estimated_flow_source_to_target=estimated_flow_source_to_target[-1] if
                                 estimated_flow_source_to_target is not None else None,
                                 sparse=mini_batch['sparse'], mask=output_un['mask_training'][-1],
                                 image_target_prime_ss=mini_batch['target_image_prime_ss'] if 'target_image_prime_ss'
                                 in list(mini_batch.keys()) else None,
                                 mask_cyclic=output_un['mask_cyclic'][-1] if 'mask_cyclic' in list(output_un.keys())
                                 else None)
        return loss, stats

