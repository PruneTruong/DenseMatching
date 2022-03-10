from training.actors.base_actor import BaseActor
import os
from training.plot.plot_features import plot_correlation_for_probabilistic_warpc
from training.plot.plot_sparse_keypoints import plot_sparse_keypoints, plot_matches
from training.plot.plot_corr import plot_correlation
from utils_flow.correlation_to_matches_utils import estimate_epe_from_correlation, correlation_to_flow_w_argmax
import math
import torch
from models.semantic_matching_models.DHPF.base.geometry import Geometry
from training.losses.cost_volume_losses.specific_semantic_network_losses import Evaluator
import numpy as np


class LayerSelectionObjective:
    def __init__(self, target_rate=0.5):
        self.target_rate = target_rate

    def __call__(self, layer_sel):
        r"""Encourages model to select each layer at a certain rate"""
        return (layer_sel.mean(dim=0) - self.target_rate).pow(2).sum()


class NormalizeCorrelation:
    def __init__(self, normalization):
        self.normalization = normalization

    @staticmethod
    def l1normalize(x):
        r"""L1-normalization"""
        vector_sum = torch.sum(x, dim=1, keepdim=True)
        vector_sum[vector_sum == 0] = 1.0
        return x / vector_sum

    @staticmethod
    def maxnormalize(x):
        vector_sum = torch.max(x, dim=1, keepdim=True)[0]
        vector_sum[vector_sum == 0] = 1.0
        return x / vector_sum

    @staticmethod
    def unit_gaussian_normalize_with_detach(x):
        r"""Make each (row) distribution into unit gaussian"""
        correlation_matrix = x - x.detach().mean(dim=1).unsqueeze(1).expand_as(x)

        with torch.no_grad():
            standard_deviation = correlation_matrix.detach().std(dim=1)
            standard_deviation[standard_deviation == 0] = 1.0
        correlation_matrix /= standard_deviation.unsqueeze(1).expand_as(correlation_matrix)

        return correlation_matrix

    @staticmethod
    def unit_gaussian_normalize(x):
        r"""Make each (row) distribution into unit gaussian"""
        correlation_matrix = x - x.mean(dim=1).unsqueeze(1).expand_as(x)

        with torch.no_grad():
            standard_deviation = correlation_matrix.std(dim=1)
            standard_deviation[standard_deviation == 0] = 1.0
        correlation_matrix /= standard_deviation.unsqueeze(1).expand_as(correlation_matrix)

        return correlation_matrix

    def __call__(self, x):
        if self.normalization == 'L1':
            return self.l1normalize(x)
        elif self.normalization == 'max':
            return self.maxnormalize(x)
        elif self.normalization == 'gaussian_unit':
            return self.unit_gaussian_normalize(x)
        elif self.normalization == 'gaussian_unit_detached':
            return self.unit_gaussian_normalize_with_detach(x)
        else:
            return x


class DHPFModelWithTripletProbabilisticWarpConsistency(BaseActor):
    """Actor for training DHPF matching network (with output cost volume/probabilistic mapping representation) using the
     probabilistic w-bipath and probabilistic warp-supervision objective (creates an image triplet).
     Here, no negative loss, only the probabilistic triplet loss applied. """

    def __init__(self, net, triplet_objective, batch_processing, nbr_images_to_plot=3,
                 best_value_based_on_loss=False, plot_every=5, sparse_kp=False):
        """
        Args:
            net: The network to train
            triplet_objective: module responsible for computing the pw-bipath and pwarp-supervision losses
            batch_processing: A processing class which performs the necessary processing of the batched data.
                              Corresponds to creating the image triplet here.
            nbr_images_to_plot: number of images to plot per epoch
            best_value_based_on_loss: track best model (epoch wise) using the loss instead of evaluation metrics
            plot_every: plot every 5 epochs
            sparse_kp:
        """
        super().__init__(net, triplet_objective, batch_processing)

        self.nbr_images_to_plot = nbr_images_to_plot
        self.layer_selection_objective = LayerSelectionObjective()
        self.sparse_kp = sparse_kp
        self.best_value_based_on_loss = best_value_based_on_loss
        self.plot_every = plot_every

    @staticmethod
    def get_val_metrics(stats, correlation, flow_gt, mask_gt, name, alpha_1=0.1, alpha_2=0.15):
        b, _, h, w = correlation.shape
        assert torch.isnan(correlation).sum() == 0
        assert torch.isnan(flow_gt).sum() == 0
        assert torch.isnan(mask_gt).sum() == 0
        pck_tresh_1 = max(flow_gt.shape[-2:]) * alpha_1
        pck_tresh_2 = max(flow_gt.shape[-2:]) * alpha_2
        epe, pck_1, pck_3 = estimate_epe_from_correlation(correlation[:, :h*w], flow_gt=flow_gt, mask_gt=mask_gt,
                                                          pck_thresh_1=pck_tresh_1, pck_thresh_2=pck_tresh_2)
        stats['{}_epe'.format(name, h, w)] = epe
        stats['{}_pck_{}'.format(name, alpha_1)] = pck_1*100
        stats['{}_pck_{}'.format(name, alpha_2)] = pck_3*100
        return stats

    def __call__(self, mini_batch, training):
        """
        args:
            mini_batch: The mini batch input data, should at least contain the fields 'source_image', 'target_image',
                        'flow_map', 'correspondence_mask'
            training: bool indicating if we are in training or evaluation mode
        returns:
            loss: the training loss
            stats: dict containing detailed losses
        """

        # plot images
        plot = False
        epoch = mini_batch['epoch']
        iter = mini_batch['iter']
        training_or_validation = 'train' if training else 'val'
        base_save_dir = os.path.join(mini_batch['settings'].env.workspace_dir,
                                     mini_batch['settings'].project_path,
                                     'plot', training_or_validation)
        # Calculates validation stats
        if epoch % self.plot_every == 0 and iter < self.nbr_images_to_plot:
            plot = True
            if not os.path.isdir(base_save_dir):
                os.makedirs(base_save_dir)

        # Run network
        # creates the image triplet here
        mini_batch = self.batch_processing(mini_batch)  # also put to GPU there

        correlation_matrix_s_to_t, layer_sel_s_to_t = self.net(im_source=mini_batch['target_image'],
                                                               im_target=mini_batch['source_image'])
        h = int(math.sqrt(correlation_matrix_s_to_t.shape[-1])) if len(correlation_matrix_s_to_t.shape) == 3 \
            else correlation_matrix_s_to_t.shape[-1]
        b = correlation_matrix_s_to_t.shape[0]
        correlation_matrix_s_to_t = correlation_matrix_s_to_t.reshape(b, -1, h, h)

        correlation_matrix_tp_to_s, layer_sel_tp_to_s = self.net(im_source=mini_batch['source_image'],
                                                                 im_target=mini_batch['target_image_prime'])
        # shape is b, h_s*w_s, h_tp*w_tp or b, h_s*w_s+1, h_tp*w_tp
        correlation_matrix_tp_to_s = correlation_matrix_tp_to_s.reshape(b, -1, h, h)

        correlation_matrix_tp_to_t, layer_sel_tp_to_t = self.net(im_source=mini_batch['target_image'],
                                                                 im_target=mini_batch['target_image_prime'])
        correlation_matrix_tp_to_t = correlation_matrix_tp_to_t.reshape(b, -1, h, h)

        # Probabilistic warp consistency on positive image pairs
        loss_contrastive, stats, dict_proba_matrices = self.objective(
            A_target_prime_to_source=correlation_matrix_tp_to_s, A_source_to_target=correlation_matrix_s_to_t,
            A_target_prime_to_target=correlation_matrix_tp_to_t,
            flow_gt_full=mini_batch['flow_map'], mask_valid=mini_batch['mask'])

        stats['Loss_triplet/total'] = loss_contrastive.item()

        loss_layer = self.layer_selection_objective(layer_sel_s_to_t) + \
                self.layer_selection_objective(layer_sel_tp_to_t) + \
                self.layer_selection_objective(layer_sel_tp_to_s)
        loss_contrastive += loss_layer

        stats['Loss_layer/total'] = loss_layer.item()
        stats['Loss/total'] = loss_contrastive.item()

        if plot:
            if 'P_w_bipath' in dict_proba_matrices.keys():
                plot_correlation_for_probabilistic_warpc(
                    target_image=mini_batch['target_image'][0], source_image=mini_batch['source_image'][0],
                    target_image_prime=mini_batch['target_image_prime'][0], flow_gt=mini_batch['flow_map'][0],
                    correlation_volume_tp_to_s=dict_proba_matrices['P_target_prime_to_source'].detach()[0],
                    correlation_volume_s_to_t=dict_proba_matrices['P_source_to_target'].detach()[0],
                    correlation_volume_tp_to_t=dict_proba_matrices['P_w_bipath'].detach()[0],
                    correlation_volume_t_to_s=dict_proba_matrices['A_target_to_source'].detach()[0],
                    save_path=base_save_dir, name='global_corr_ep{}_{}'.format(epoch, iter),
                    normalization='softmax', mask=mini_batch['mask'][0], plot_individual=self.plot_ind,
                    mask_top=dict_proba_matrices['mask_top_percent'].detach()[0] if
                    'mask_top_percent' in dict_proba_matrices.keys() else None)
            else:
                if 'flow_map_target_to_source' in list(mini_batch.keys()) and 'A_target_to_source' in \
                        dict_proba_matrices.keys():
                    flow_gt_target_to_source = mini_batch['flow_map_target_to_source']
                    _ = plot_sparse_keypoints(base_save_dir, 'epoch{}_batch{}'.format(epoch, iter),
                                              mini_batch, flow_gt_target_to_source,
                                              correlation_to_flow_w_argmax(
                                                  dict_proba_matrices['A_target_to_source'].detach(),
                                                  output_shape=mini_batch['flow_map'].shape[-2:]))

        if not training:
            if 'P_warp_supervision' in dict_proba_matrices.keys():
                stats = self.get_val_metrics(stats, dict_proba_matrices['P_warp_supervision'].detach(),
                                             mini_batch['flow_map'], mini_batch['correspondence_mask'],
                                             name='pwarp_supervision')

            if 'P_w_bipath' in dict_proba_matrices.keys():
                stats = self.get_val_metrics(stats, dict_proba_matrices['P_w_bipath'].detach(),
                                             mini_batch['flow_map'], mini_batch['correspondence_mask'],
                                             name='pw_bipath')

            if 'flow_map_target_to_source' in list(mini_batch.keys()):
                with torch.no_grad():
                    model_output_t_to_s, layer_sel_t_to_s = self.net(
                        im_source=mini_batch['source_image'], im_target=mini_batch['target_image'])

                mask_gt_target_to_source = mini_batch['correspondence_mask_target_to_source']
                flow_gt_target_to_source = mini_batch['flow_map_target_to_source']
                stats = self.get_val_metrics(stats, model_output_t_to_s.detach().view(b, -1, h, h),
                                             flow_gt_target_to_source, mask_gt_target_to_source,
                                             name='real_gt')

            if self.best_value_based_on_loss:
                stats['best_value'] = - stats['w_bipath_pck_0.1'] if 'P_w_bipath' in dict_proba_matrices.keys() else \
                        - stats['pwarp_supervision_pck_0.1']
            else:
                if 'flow_map_target_to_source' in list(mini_batch.keys()) and 'A_target_to_source' in \
                        dict_proba_matrices.keys():
                    stats['best_value'] = - stats['real_gt_pck_0.1']
                    # we want PCK to increase but best_value must decrease
                else:
                    stats['best_value'] = - stats['w_bipath_pck_0.1'] if 'P_w_bipath' in dict_proba_matrices.keys() else \
                        - stats['pwarp_supervision_pck_0.1']

        return loss_contrastive, stats


class DHPFModelWithTripletAndPairWiseProbabilisticWarpConsistency(BaseActor):
    """Actor for training matching networks (with output cost volume/probabilistic mapping representation) using the
     pw-bipath and pwarp-supervision objective (triplet objective), combined with the negative or keypoint loss
     (pairwise objective).
    """
    def __init__(self, net, triplet_objective, batch_processing, pairwise_objective, loss_weights=None,
                 compute_both_directions=False, balance_triplet_and_pairwise_losses=False,
                 best_value_based_on_loss=False, plot_every=5, nbr_images_to_plot=1):
        """
        Args:
            net: The network to train
            triplet_objective: module responsible for computing the pw-bipath and pwarp-supervision losses
            batch_processing: A processing class which performs the necessary processing of the batched data.
                              Corresponds to creating the image triplet here.
            pairwise_objective: objective on matching and non-matching image pairs (could be supervised cross entropy)
            compute_both_directions: bool, compute the loss corresponding to the pairwise_objective in both directions?
            loss_weights: dictionary with the loss weights to apply to the different losses
            balance_triplet_and_pairwise_losses: balance loss of original_objective with PWarpC so they all have
                                                 the same magnitude?
            best_value_based_on_loss: track best model (epoch wise) using the loss instead of evaluation metrics
            plot_every: plot every 5 epochs
            nbr_images_to_plot: number of images to plot per epoch
        """
        super().__init__(net, triplet_objective, batch_processing)

        default_loss_weights = {'triplet': 1.0, 'pairwise': 1.0, 'layer': 1.0}
        if loss_weights is not None:
            default_loss_weights.update(loss_weights)
        self.loss_weights = default_loss_weights

        self.nbr_images_to_plot = nbr_images_to_plot

        self.pairwise_objective = pairwise_objective
        self.layer_selection_objective = LayerSelectionObjective()
        self.balance_losses = balance_triplet_and_pairwise_losses

        self.best_value_based_on_loss = best_value_based_on_loss
        self.plot_every = plot_every
        self.compute_both_directions = compute_both_directions

    @staticmethod
    def get_val_metrics(stats, correlation, flow_gt, mask_gt, name, alpha_1=0.1, alpha_2=0.15):
        b, _, h, w = correlation.shape
        # correlation = Correlation.mutual_nn_filter(correlation.view(b, -1, h*w))
        # correlation = correlation.view(b, -1, h, w)
        assert torch.isnan(correlation).sum() == 0
        assert torch.isnan(flow_gt).sum() == 0
        assert torch.isnan(mask_gt).sum() == 0
        pck_tresh_1 = max(flow_gt.shape[-2:]) * alpha_1
        pck_tresh_2 = max(flow_gt.shape[-2:]) * alpha_2
        epe, pck_1, pck_3 = estimate_epe_from_correlation(correlation[:, :h*w], flow_gt=flow_gt, mask_gt=mask_gt,
                                                          pck_thresh_1=pck_tresh_1, pck_thresh_2=pck_tresh_2)
        stats['{}_epe'.format(name, h, w)] = epe
        stats['{}_pck_{}'.format(name, alpha_1)] = pck_1*100
        stats['{}_pck_{}'.format(name, alpha_2)] = pck_3*100
        return stats

    def __call__(self, mini_batch, training):
        """
        args:
            mini_batch: The mini batch input data, should at least contain the fields 'source_image', 'target_image',
                        'flow_map', 'correspondence_mask'
            training: bool indicating if we are in training or evaluation mode
        returns:
            loss: the training loss
            stats: dict containing detailed losses
        """

        # plot images
        plot = False
        epoch = mini_batch['epoch']
        iter = mini_batch['iter']
        training_or_validation = 'train' if training else 'val'
        base_save_dir = os.path.join(mini_batch['settings'].env.workspace_dir,
                                     mini_batch['settings'].project_path,
                                     'plot', training_or_validation)
        # Calculates validation stats
        if epoch % self.plot_every == 0 and iter < self.nbr_images_to_plot:
            plot = True
            if not os.path.isdir(base_save_dir):
                os.makedirs(base_save_dir)

        loss = 0.0

        # Run network
        # creates the image triplet here
        mini_batch = self.batch_processing(mini_batch)  # also put to GPU there

        # I. objective on matching and non-matching image pairs :
        # can be negative loss, or loss based on keypoint matches
        # get the image pairs, possibly with negative pairs.
        src_img_w_neg, trg_img_w_neg = self.pairwise_objective.get_image_pair(mini_batch, training)
        # if need to have negative pairs
        nb_pos = self.pairwise_objective.bsz
        correlation_matrix_s_to_t_with_neg, layer_sel_s_to_t_with_neg = self.net(im_source=trg_img_w_neg,
                                                                                 im_target=src_img_w_neg)

        loss_neg, stats_ = self.pairwise_objective.compute_loss(correlation_matrix_s_to_t_with_neg,
                                                                batch=mini_batch, training=training,
                                                                t_to_s=False)
        if self.compute_both_directions:
            correlation_matrix_t_to_s_with_neg, layer_sel_t_to_s_with_neg = self.net(im_source=src_img_w_neg,
                                                                                     im_target=trg_img_w_neg)

            loss_neg_, stats_o = self.pairwise_objective.compute_loss(correlation_matrix_t_to_s_with_neg,
                                                                      batch=mini_batch, training=training,
                                                                      t_to_s=True)
            loss_neg += loss_neg_
            stats_.update(stats_o)

        # model predictions
        correlation_matrix_tp_to_s, layer_sel_tp_to_s = self.net(im_source=mini_batch['source_image'],
                                                                 im_target=mini_batch['target_image_prime'])
        # shape is b, h_s*w_s, h_tp*w_tp
        h = int(math.sqrt(correlation_matrix_tp_to_s.shape[-1])) if len(correlation_matrix_tp_to_s.shape) == 3 \
            else correlation_matrix_tp_to_s.shape[-1]
        b = correlation_matrix_tp_to_s.shape[0]
        correlation_matrix_tp_to_s = correlation_matrix_tp_to_s.reshape(b, -1, h, h)

        correlation_matrix_s_to_t = correlation_matrix_s_to_t_with_neg[:nb_pos]
        correlation_matrix_s_to_t = correlation_matrix_s_to_t.reshape(b, -1, h, h)

        correlation_matrix_tp_to_t, layer_sel_tp_to_t = self.net(im_source=mini_batch['target_image'],
                                                                 im_target=mini_batch['target_image_prime'])
        correlation_matrix_tp_to_t = correlation_matrix_tp_to_t.reshape(b, -1, h, h)

        # Probabilistic warp consistency on positive image pairs
        loss_contrastive, stats, dict_proba_matrices = self.objective(
            A_target_prime_to_source=correlation_matrix_tp_to_s, A_source_to_target=correlation_matrix_s_to_t,
            A_target_prime_to_target=correlation_matrix_tp_to_t,
            flow_gt_full=mini_batch['flow_map'], mask_valid=mini_batch['mask'])

        stats.update(stats_)
        stats['Loss_triplet/total'] = loss_contrastive.item()

        loss_layer = self.layer_selection_objective(layer_sel_s_to_t_with_neg) + \
                self.layer_selection_objective(layer_sel_tp_to_t) + \
                self.layer_selection_objective(layer_sel_tp_to_s)
        if self.balance_losses:
            w_original = torch.abs(loss_contrastive.detach()) / torch.abs(loss_neg.detach() + 1e-6)
            loss_neg = w_original * loss_neg
        else:
            loss_contrastive = loss_contrastive * self.loss_weights['triplet']
            loss_layer = loss_layer * self.loss_weights['layer']
            loss_neg = loss_neg * self.loss_weights['pairwise']

        loss += loss_contrastive
        loss += loss_layer
        loss += loss_neg
        stats['Loss_neg_pos_after_balancing/total'] = loss_neg.item()
        stats['Loss_layer_after_balancing/total'] = loss_layer.item()
        stats['Loss/total'] = loss.item()

        if plot and 'P_w_bipath' in dict_proba_matrices.keys():
            plot_correlation_for_probabilistic_warpc(
                target_image=mini_batch['target_image'][0], source_image=mini_batch['source_image'][0],
                target_image_prime=mini_batch['target_image_prime'][0], flow_gt=mini_batch['flow_map'][0],
                correlation_volume_tp_to_s=dict_proba_matrices['P_target_prime_to_source'].detach()[0],
                correlation_volume_s_to_t=dict_proba_matrices['P_source_to_target'].detach()[0],
                correlation_volume_tp_to_t=dict_proba_matrices['P_w_bipath'].detach()[0],
                correlation_volume_t_to_s=dict_proba_matrices['A_target_to_source'].detach()[0],
                save_path=base_save_dir, name='global_corr_ep{}_{}'.format(epoch, iter),
                normalization='softmax', mask=mini_batch['mask'][0],
                mask_top=dict_proba_matrices['mask_top_percent'].detach()[0] if
                'mask_top_percent' in dict_proba_matrices.keys() else None)

            if 'flow_map_target_to_source' in list(mini_batch.keys()) and 'A_target_to_source' in \
                    dict_proba_matrices.keys():
                flow_gt_target_to_source = mini_batch['flow_map_target_to_source']
                _ = plot_sparse_keypoints(base_save_dir, 'epoch{}_batch{}'.format(epoch, iter),
                                          mini_batch, flow_gt_target_to_source,
                                          correlation_to_flow_w_argmax(
                                              dict_proba_matrices['A_target_to_source'].detach(),
                                              output_shape=mini_batch['flow_map'].shape[-2:]))

        if not training:
            if 'P_warp_supervision' in dict_proba_matrices.keys():
                stats = self.get_val_metrics(stats, dict_proba_matrices['P_warp_supervision'].detach(),
                                             mini_batch['flow_map'], mini_batch['correspondence_mask'],
                                             name='pwarp_supervision')

            if 'P_w_bipath' in dict_proba_matrices.keys():
                stats = self.get_val_metrics(stats, dict_proba_matrices['P_w_bipath'].detach(),
                                             mini_batch['flow_map'], mini_batch['correspondence_mask'],
                                             name='pw_bipath')

            if 'flow_map_target_to_source' in list(mini_batch.keys()):
                with torch.no_grad():
                    if self.compute_both_directions:
                        model_output_t_to_s = correlation_matrix_t_to_s_with_neg
                    else:
                        model_output_t_to_s, layer_sel_t_to_s = self.net(im_source=mini_batch['source_image'],
                                                                         im_target=mini_batch['target_image'])

                mask_gt_target_to_source = mini_batch['correspondence_mask_target_to_source']
                flow_gt_target_to_source = mini_batch['flow_map_target_to_source']
                stats = self.get_val_metrics(stats, model_output_t_to_s.detach().view(b, -1, h, h),
                                             flow_gt_target_to_source, mask_gt_target_to_source,
                                             name='real_gt')

            if self.best_value_based_on_loss:
                stats['best_value'] = - stats['w_bipath_pck_0.1'] if 'P_w_bipath' in dict_proba_matrices.keys() else \
                    - stats['pwarp_supervision_pck_0.1']
            else:
                if 'flow_map_target_to_source' in list(mini_batch.keys()) and 'A_target_to_source' in \
                        dict_proba_matrices.keys():
                    stats['best_value'] = - stats['real_gt_pck_0.1']
                    # we want PCK to increase but best_value must decrease
                else:
                    stats['best_value'] = - stats['w_bipath_pck_0.1'] if 'P_w_bipath' in dict_proba_matrices.keys() else \
                        - stats['pwarp_supervision_pck_0.1']

        return loss, stats



