
import os
import torch
import math

from training.actors.base_actor import BaseActor
from training.plot.plot_features import plot_correlation_for_probabilistic_warpc
from training.plot.plot_sparse_keypoints import plot_sparse_keypoints, plot_matches
from training.plot.plot_corr import plot_correlation
from utils_flow.correlation_to_matches_utils import estimate_epe_from_correlation, correlation_to_flow_w_argmax


class ModelWithPairWiseLoss(BaseActor):
    """Actor for training matching networks (with output cost volume/probabilistic mapping representation) from a
    pair of matching image pairs. """

    def __init__(self, net, objective, batch_processing, compute_both_directions=False,
                 plot_every=5, nbr_images_to_plot=3):
        """
        Args:
            net: The network to train
            objective: pairwise objective
            batch_processing: A processing class which performs the necessary processing of the batched data.
            compute_both_directions: bool, compute the loss corresponding to the pairwise_objective in both directions?
            nbr_images_to_plot: number of images to plot per epoch
        """
        super().__init__(net, objective, batch_processing)

        self.nbr_images_to_plot = nbr_images_to_plot
        self.plot_every = plot_every
        self.compute_both_directions = compute_both_directions

    @staticmethod
    def get_val_metrics(stats, correlation, flow_gt, mask_gt, name, alpha_1=0.05, alpha_2=0.1):
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
                        'flow_map', 'correspondence_mask'.
                        'flow_map' is the ground-truth flow relating the target to the source.

            training: bool indicating if we are in training or evaluation mode
        returns:
            loss: the training loss
            stats: dict containing detailed losses
        """
        # Run network
        # normalize the image pair here
        mini_batch = self.batch_processing(mini_batch)  # also put to GPU there

        # if need to have negative pairs
        src_img_w_neg, trg_img_w_neg = self.objective.get_image_pair(mini_batch, training)
        model_output = self.net(im_source=src_img_w_neg, im_target=trg_img_w_neg)

        # model_output_t_to_s can be directly correlation_from_t_to_s for NC-Net or dict for DCC-Net
        if isinstance(model_output, dict):
            model_output_t_to_s = model_output['correlation_from_t_to_s']
        else:
            model_output_t_to_s = model_output

        if len(model_output_t_to_s.shape) == 3:
            hw = model_output_t_to_s.shape[-1]
            h = w = int(math.sqrt(hw))
        else:
            h, w = model_output_t_to_s.shape[-2:]
        b = model_output_t_to_s[0]
        model_output_t_to_s = model_output_t_to_s.reshape(b, -1, h, w)

        # 4. Compute loss to update weights
        loss, stats = self.objective.compute_loss(model_output_t_to_s,  model=self.net, batch=mini_batch,
                                                  training=training, t_to_s=True)

        if self.compute_both_directions:
            if isinstance(model_output, dict) and 'correlation_from_s_to_t' in model_output.keys():
                model_output_s_to_t = model_output['correlation_from_s_to_t']
            else:
                model_output_s_to_t = self.net(im_source=trg_img_w_neg, im_target=src_img_w_neg)

                # model_output_t_to_s can be directly correlation_from_t_to_s for NC-Net or dict for DCC-Net
                if isinstance(model_output_s_to_t, dict):
                    model_output_s_to_t = model_output_s_to_t['correlation_from_t_to_s']
                model_output_s_to_t = model_output_s_to_t.reshape(b, -1, h, w)

            # 4. Compute loss to update weights
            loss_, stats_ = self.objective.compute_loss(model_output_s_to_t, model=self.net, batch=mini_batch,
                                                        training=training, t_to_s=False)
            loss += loss_
            stats.update(stats_)

        # plot images
        epoch = mini_batch['epoch']
        iter = mini_batch['iter']
        training_or_validation = 'train' if training else 'val'
        base_save_dir = os.path.join(mini_batch['settings'].env.workspace_dir,
                                     mini_batch['settings'].project_path,
                                     'plot', training_or_validation)

        # Calculates validation stats
        if epoch % self.plot_every == 0 and iter < self.nbr_images_to_plot:
            if not os.path.isdir(base_save_dir):
                os.makedirs(base_save_dir)

            plot_correlation(target_image=mini_batch['target_image'][0],  source_image=mini_batch['source_image'][0],
                             flow_gt=mini_batch['flow_map'][0],
                             correlation_volume=self.objective.get_correlation(model_output_t_to_s)[:, :h*w].view(-1, h*w, h, w)[0],
                             # only positive pair
                             save_path=base_save_dir, normalization='min',
                             name='global_corr_ep{}_{}'.format(epoch, iter))

            _ = plot_sparse_keypoints(base_save_dir, 'epoch{}_batch{}'.format(epoch, iter),
                                      mini_batch, mini_batch['flow_map'],
                                      output_net=correlation_to_flow_w_argmax(
                                          self.objective.get_correlation(model_output_t_to_s)[:, :h*w].view(-1, h*w, h, w),
                                          output_shape=mini_batch['flow_map'].shape[-2:]))

        if not training:
            if 'flow_map' in mini_batch.keys():
                # there is a ground-truth flow
                stats = self.get_val_metrics(stats,
                                             self.objective.get_correlation(model_output_t_to_s)[:, :h*w].view(-1, h * w, h, w),
                                             mini_batch['flow_map'], mini_batch['correspondence_mask'],
                                             name='real_gt')
                stats['best_value'] = - stats['real_gt_pck_0.1']
                # we want PCK to increase but best_value must decrease
            else:
                stats['best_value'] = loss.item()

        return loss, stats


class ModelWithTripletProbabilisticWarpConsistency(BaseActor):
    """Actor for training matching networks (with output cost volume/probabilistic mapping representation) using the
     probabilistic w-bipath and probabilistic warp-supervision objective (creates an image triplet).
     Here, no negative loss, only the probabilistic triplet loss applied. """
    def __init__(self, net, triplet_objective, batch_processing, nbr_images_to_plot=3,
                 best_value_based_on_loss=False, plot_every=5, plot_ind=False):
        """
        Args:
            net: The network to train
            triplet_objective: module responsible for computing the pw-bipath and pwarp-supervision losses
            batch_processing: A processing class which performs the necessary processing of the batched data.
                              Corresponds to creating the image triplet here.
            nbr_images_to_plot: number of images to plot per epoch
            best_value_based_on_loss: track best model (epoch wise) using the loss instead of evaluation metrics
            plot_every: plot every 5 epochs
        """
        super().__init__(net, triplet_objective, batch_processing)

        self.nbr_images_to_plot = nbr_images_to_plot
        self.best_value_based_on_loss = best_value_based_on_loss
        self.plot_every = plot_every
        self.plot_ind = plot_ind

    @staticmethod
    def get_val_metrics(stats, correlation, flow_gt, mask_gt, name, alpha_1=0.05, alpha_2=0.1):
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
                        'flow_map', 'correspondence_mask'.
                        'flow_map' is the ground-truth flow relating the target to the source.
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
        # creates the image triplet, and puts all inputs to GPU
        mini_batch = self.batch_processing(mini_batch)
        """
        output data block with at least the fields 'source_image', 'target_image', 'target_image_prime', 
        'flow_map', 'correspondence_mask'.
        ATTENTION: 'flow_map' contains the synthetic flow field, relating the target_image_prime to
        the target. This is NOT the same flow_map than in the original mini_batch. Similarly,
        'correspondence_mask' identifies the valid (in-view) flow regions of the synthetic flow_map.

        If ground-truth between source and target image is known (was provided), will contain the fields
        'flow_map_target_to_source', 'correspondence_mask_target_to_source'.

        If ground-truth keypoints in source and target were provided, will contain the fields
        'target_kps' and 'source_kps'.
        
        """

        # computes flows
        # model_output can be directly correlation_from_t_to_s for NC-Net or dict for DCC-Net
        model_output_s_to_t = self.net(im_source=mini_batch['target_image'], im_target=mini_batch['source_image'])

        if isinstance(model_output_s_to_t, dict):
            correlation_matrix_s_to_t = model_output_s_to_t['correlation_from_t_to_s']
        else:
            correlation_matrix_s_to_t = model_output_s_to_t

        if len(correlation_matrix_s_to_t.shape) == 3:
            h = w = int(math.sqrt(correlation_matrix_s_to_t.shape[-1]))
        else:
            h, w = correlation_matrix_s_to_t.shape[-2:]
        b = correlation_matrix_s_to_t.shape[0]
        correlation_matrix_s_to_t = correlation_matrix_s_to_t.reshape(b, -1, h, w)

        # tp to s
        model_output_tp_to_s = self.net(im_source=mini_batch['source_image'],
                                        im_target=mini_batch['target_image_prime'])
        if isinstance(model_output_tp_to_s, dict):
            correlation_matrix_tp_to_s = model_output_tp_to_s['correlation_from_t_to_s']
        else:
            correlation_matrix_tp_to_s = model_output_tp_to_s
        correlation_matrix_tp_to_s = correlation_matrix_tp_to_s.reshape(b, -1, h, w)

        # tp tp t
        model_output_tp_to_t = self.net(im_source=mini_batch['target_image'],
                                        im_target=mini_batch['target_image_prime'])
        if isinstance(model_output_tp_to_t, dict):
            correlation_matrix_tp_to_t = model_output_tp_to_t['correlation_from_t_to_s']
        else:
            correlation_matrix_tp_to_t = model_output_tp_to_t
        correlation_matrix_tp_to_t = correlation_matrix_tp_to_t.reshape(b, -1, h, w)

        # Probabilistic warp consistency on positive image pairs
        loss_triplet, stats, dict_proba_matrices = self.objective(
            A_target_prime_to_source=correlation_matrix_tp_to_s, A_source_to_target=correlation_matrix_s_to_t,
            A_target_prime_to_target=correlation_matrix_tp_to_t,
            flow_gt_full=mini_batch['flow_map'], mask_valid=mini_batch['mask'])

        stats['Loss_triplet/total'] = loss_triplet.item()

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
                                             name='warp_supervision')

            if 'P_w_bipath' in dict_proba_matrices.keys():
                stats = self.get_val_metrics(stats, dict_proba_matrices['P_w_bipath'].detach(),
                                             mini_batch['flow_map'], mini_batch['correspondence_mask'],
                                             name='w_bipath')

            if 'flow_map_target_to_source' in list(mini_batch.keys()):
                with torch.no_grad():
                    model_output_t_to_s = self.net(im_source=mini_batch['source_image'],
                                                   im_target=mini_batch['target_image'])
                    if isinstance(model_output_t_to_s, dict):
                        correlation_matrix_t_to_s = model_output_t_to_s['correlation_from_t_to_s']
                    else:
                        correlation_matrix_t_to_s = model_output_t_to_s

                mask_gt_target_to_source = mini_batch['correspondence_mask_target_to_source']
                flow_gt_target_to_source = mini_batch['flow_map_target_to_source']
                stats = self.get_val_metrics(stats, correlation_matrix_t_to_s.detach().view(b, -1, h, w),
                                             flow_gt_target_to_source, mask_gt_target_to_source,
                                             name='real_gt')

            if self.best_value_based_on_loss:
                stats['best_value'] = - stats['w_bipath_pck_0.1'] if 'P_w_bipath' in dict_proba_matrices.keys() else \
                        - stats['pwarp_supervision_pck_0.1']
            else:
                if 'flow_map_target_to_source' in list(mini_batch.keys()):
                    stats['best_value'] = - stats['real_gt_pck_0.1']
                    # we want PCK to increase but best_value must decrease
                else:
                    stats['best_value'] = - stats['w_bipath_pck_0.1'] if 'P_w_bipath' in dict_proba_matrices.keys() \
                        else - stats['pwarp_supervision_pck_0.1']

        return loss_triplet, stats


class ModelWithTripletAndPairWiseProbabilisticWarpConsistency(BaseActor):
    """Actor for training matching networks (with output cost volume/probabilistic mapping representation) using the
     pw-bipath and pwarp-supervision objective (triplet objective), combined with the negative or keypoint loss
     (pairwise objective).
    """
    def __init__(self, net, triplet_objective, batch_processing, pairwise_objective, loss_weights=None,
                 compute_both_directions=False, balance_triplet_and_pairwise_losses=False, compute_flow=False,
                 plot_ind=False, best_value_based_on_loss=False, plot_every=5, nbr_images_to_plot=3):
        """
        Args:
            net: The network to train
            triplet_objective: module responsible for computing the pw-bipath and pwarp-supervision losses
            batch_processing: A processing class which performs the necessary processing of the batched data.
                              Corresponds to creating the image triplet here.
            pairwise_objective: objective on matching and non-matching image pairs (could be supervised cross entropy)
            compute_both_directions: bool, compute the loss corresponding to the pairwise_objective in both directions?
            compute_flow: bool, loss will be on the flow field?
            loss_weights: dictionary with the loss weights to apply to the different losses
            balance_triplet_and_pairwise_losses: balance loss of original_objective with PWarpC so they all have the same magnitude?
            best_value_based_on_loss: track best model (epoch wise) using the loss instead of evaluation metrics
            plot_every: plot every 5 epochs
            nbr_images_to_plot: number of images to plot per epoch
        """
        super().__init__(net, triplet_objective, batch_processing)

        default_loss_weights = {'triplet': 1.0, 'pairwise': 1.0}
        if loss_weights is not None:
            default_loss_weights.update(loss_weights)
        self.loss_weights = default_loss_weights
        self.compute_both_directions = compute_both_directions

        self.nbr_images_to_plot = nbr_images_to_plot
        self.pairwise_objective = pairwise_objective
        self.balance_triplet_and_pairwise_losses = balance_triplet_and_pairwise_losses
        self.best_value_based_on_loss = best_value_based_on_loss
        self.plot_every = plot_every
        self.compute_flow = compute_flow
        if self.compute_flow:
            self.compute_both_directions = False  # currently not possible
        self.plot_ind = plot_ind

    @staticmethod
    def get_val_metrics(stats, correlation, flow_gt, mask_gt, name, alpha_1=0.05, alpha_2=0.1):
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

        # creates the image triplet here, and put all inputs to gpu and right format
        mini_batch = self.batch_processing(mini_batch)
        """
        output data block with at least the fields 'source_image', 'target_image', 'target_image_prime', 
        'flow_map', 'correspondence_mask'.
        ATTENTION: 'flow_map' contains the synthetic flow field, relating the target_image_prime to
        the target. This is NOT the same flow_map than in the original mini_batch. Similarly,
        'correspondence_mask' identifies the valid (in-view) flow regions of the synthetic flow_map.

        If ground-truth between source and target image is known (was provided), will contain the fields
        'flow_map_target_to_source', 'correspondence_mask_target_to_source'.

        If ground-truth keypoints in source and target were provided, will contain the fields
        'target_kps' and 'source_kps'.

        """

        loss = 0.0

        # I. objective on matching and non-matching image pairs :
        # can be negative loss, or loss based on keypoint matches
        # get the image pairs, possibly with negative pairs.
        src_img_w_neg, trg_img_w_neg = self.pairwise_objective.get_image_pair(mini_batch, training)
        nb_pos = self.pairwise_objective.bsz

        # network prediction from s to t, potentially augmented with negatives
        model_output_s_to_t_with_neg = self.net(im_source=trg_img_w_neg, im_target=src_img_w_neg)

        # model_output can be directly correlation_from_t_to_s for NC-Net or dict for DCC-Net
        if isinstance(model_output_s_to_t_with_neg, dict):
            correlation_matrix_s_to_t_with_neg = model_output_s_to_t_with_neg['correlation_from_t_to_s']

        else:
            correlation_matrix_s_to_t_with_neg = model_output_s_to_t_with_neg

        flow_t_to_s_with_neg = None
        if self.compute_flow:
            # need the flow from t to s.
            if isinstance(model_output_s_to_t_with_neg, dict) and \
                    'flow_from_s_to_t' in model_output_s_to_t_with_neg.keys():
                # corresponds to opposite direction, hence 'flow_from_s_to_t'
                flow_t_to_s_with_neg = model_output_s_to_t_with_neg['flow_from_s_to_t']
            else:
                model_output_t_to_s_with_neg = self.net(im_source=src_img_w_neg, im_target=trg_img_w_neg)
                flow_t_to_s_with_neg = model_output_t_to_s_with_neg['flow_from_t_to_s']

        loss_pairwise, stats_neg = self.pairwise_objective.compute_loss(
            correlation_matrix=correlation_matrix_s_to_t_with_neg, est_flow_t_to_s=flow_t_to_s_with_neg,
            model=self.net, batch=mini_batch, training=training, t_to_s=False)

        if self.compute_both_directions:
            # not possible with flow currently
            if isinstance(model_output_s_to_t_with_neg, dict) and \
                    'correlation_from_s_to_t' in model_output_s_to_t_with_neg.keys():
                # corresponds to opposite direction, hence 'correlation_from_s_to_t'
                correlation_matrix_t_to_s_with_neg = model_output_s_to_t_with_neg['correlation_from_s_to_t']
            else:
                correlation_matrix_t_to_s_with_neg = self.net(im_source=src_img_w_neg, im_target=trg_img_w_neg)

                if isinstance(correlation_matrix_t_to_s_with_neg, dict):
                    correlation_matrix_t_to_s_with_neg = correlation_matrix_t_to_s_with_neg['correlation_from_t_to_s']

            loss_neg_, stats_neg_ = self.pairwise_objective.compute_loss(
                correlation_matrix=correlation_matrix_t_to_s_with_neg, model=self.net, batch=mini_batch,
                training=training, t_to_s=True)

            loss_pairwise += loss_neg_
            stats_neg.update(stats_neg_)

        # II. probabilistic warp-consistency
        # s to t, with keeping only the positive
        correlation_matrix_s_to_t = correlation_matrix_s_to_t_with_neg[:nb_pos]
        if len(correlation_matrix_s_to_t.shape) == 3:
            h = w = int(math.sqrt(correlation_matrix_s_to_t.shape[-1]))
        else:
            h, w = correlation_matrix_s_to_t.shape[-2:]
        b = correlation_matrix_s_to_t.shape[0]
        correlation_matrix_s_to_t = correlation_matrix_s_to_t.reshape(b, -1, h, w)

        # tp to s
        model_output_tp_to_s = self.net(im_source=mini_batch['source_image'],
                                        im_target=mini_batch['target_image_prime'])
        if isinstance(model_output_tp_to_s, dict):
            correlation_matrix_tp_to_s = model_output_tp_to_s['correlation_from_t_to_s']
        else:
            correlation_matrix_tp_to_s = model_output_tp_to_s
        correlation_matrix_tp_to_s = correlation_matrix_tp_to_s.reshape(b, -1, h, w)

        # tp tp t
        model_output_tp_to_t = self.net(im_source=mini_batch['target_image'],
                                        im_target=mini_batch['target_image_prime'])
        if isinstance(model_output_tp_to_t, dict):
            correlation_matrix_tp_to_t = model_output_tp_to_t['correlation_from_t_to_s']
        else:
            correlation_matrix_tp_to_t = model_output_tp_to_t
        correlation_matrix_tp_to_t = correlation_matrix_tp_to_t.reshape(b, -1, h, w)

        loss_triplet, stats, dict_proba_matrices = self.objective(
            A_target_prime_to_source=correlation_matrix_tp_to_s, A_source_to_target=correlation_matrix_s_to_t,
            A_target_prime_to_target=correlation_matrix_tp_to_t,
            flow_gt_full=mini_batch['flow_map'], mask_valid=mini_batch['mask'])

        stats.update(stats_neg)
        
        stats['Loss_triplet/total'] = loss_triplet.item()

        if self.balance_triplet_and_pairwise_losses:
            w_original = torch.abs(loss_triplet.detach()) / torch.abs(loss_pairwise.detach() + 1e-6)
            loss_pairwise = w_original * loss_pairwise
        else:
            loss_triplet = loss_triplet * self.loss_weights['triplet']
            loss_pairwise = loss_pairwise * self.loss_weights['pairwise']

        loss += loss_triplet
        loss += loss_pairwise
        stats['Loss_pairwise_after_balancing/total'] = loss_pairwise.item()
        stats['Loss/total'] = loss.item()

        training_or_validation = 'train' if training else 'val'
        base_save_dir = os.path.join(mini_batch['settings'].env.workspace_dir,
                                     mini_batch['settings'].project_path,
                                     'plot', training_or_validation)
        # Calculates validation stats
        if epoch % self.plot_every == 0 and iter < self.nbr_images_to_plot:
            plot = True
            if not os.path.isdir(base_save_dir):
                os.makedirs(base_save_dir)
        if plot and 'P_w_bipath' in dict_proba_matrices.keys():
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
                    if self.compute_both_directions and isinstance(model_output_s_to_t_with_neg, dict) and \
                                'correlation_from_s_to_t' in model_output_s_to_t_with_neg.keys():
                        model_output_t_to_s = model_output_s_to_t_with_neg['correlation_from_s_to_t']
                    else:
                        model_output_t_to_s = self.net(im_source=mini_batch['source_image'],
                                                       im_target=mini_batch['target_image'])
                    if isinstance(model_output_t_to_s, dict):
                        correlation_matrix_t_to_s = model_output_t_to_s['correlation_from_t_to_s']
                    else:
                        correlation_matrix_t_to_s = model_output_t_to_s

                mask_gt_target_to_source = mini_batch['correspondence_mask_target_to_source']
                flow_gt_target_to_source = mini_batch['flow_map_target_to_source']
                stats = self.get_val_metrics(stats, correlation_matrix_t_to_s.detach().view(b, -1, h, w),
                                             flow_gt_target_to_source, mask_gt_target_to_source,
                                             name='real_gt')

        if not training:
            if self.best_value_based_on_loss:
                stats['best_value'] = - stats['w_bipath_pck_0.1'] if 'P_w_bipath' in dict_proba_matrices.keys() else \
                        - stats['pwarp_supervision_pck_0.1']
            else:
                if 'flow_map_target_to_source' in list(mini_batch.keys()):
                    stats['best_value'] = - stats['real_gt_pck_0.1']
                    # we want PCK to increase but best_value must decrease
                else:
                    stats['best_value'] = - stats['w_bipath_pck_0.1'] if 'P_w_bipath' in dict_proba_matrices.keys() else \
                        - stats['pwarp_supervision_pck_0.1']
        return loss, stats

