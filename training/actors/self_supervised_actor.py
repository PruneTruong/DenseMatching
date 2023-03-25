import os


from training.losses.basic_losses import realEPE, real_metrics
from admin.stats import merge_dictionaries
from .base_actor import BaseActor
from training.plot.plot_GLOCALNet import plot_basenet_during_training
from training.plot. plot_sparse_keypoints import plot_sparse_keypoints
from training.plot.plot_GLUNet import plot_during_training_with_uncertainty, plot_sparse_keypoints_GLUNet


class GLOCALNetActor(BaseActor):
    """Actor for training the GLOCALNet or BaseNet network with a self-supervised or supervised strategy."""
    def __init__(self, net, objective, batch_processing, nbr_images_to_plot=1):
        """
        Args:
            net: The network to train
            objective: The loss function
            batch_processing: A processing class which performs the necessary processing of the batched data.
            nbr_images_to_plot: number of images to plot per epoch
        """
        super().__init__(net, objective, batch_processing)
        self.nbr_images_to_plot = nbr_images_to_plot

    def __call__(self, mini_batch, training):
        """
        Args:
            mini_batch: The mini batch input data, should at least contain the fields 'source_image', 'target_image',
                        'flow_map', 'mask', 'correspondence_mask'.
                        'flow_map' is the ground-truth flow relating the target to the source. 'mask' is the mask
                        where the loss will be applied (in coordinate system of the target).
            training: bool indicating if we are in training or evaluation mode
        returns:
            loss: the training loss
            stats: dict containing detailed losses
        """
        # Run network
        mini_batch = self.batch_processing(mini_batch)
        if 'target_image_256' in mini_batch.keys():
            output_net_256, output_net = self.net(mini_batch['target_image'], mini_batch['source_image'],
                                                  mini_batch['target_image_256'], mini_batch['source_image_256'])
        else:
            output_net = self.net(im_target=mini_batch['target_image'], im_source=mini_batch['source_image'])

        loss, stats = self.objective(output_net, mini_batch['flow_map'], mask=mini_batch['mask'])

        # Log stats
        stats['Loss/total'] = loss.item()

        # Calculates validation stats
        if not training:
            for index_reso in range(len(output_net['flow_estimates'])):
                EPE, PCK_1, PCK_3, PCK_5 = real_metrics(output_net['flow_estimates'][index_reso],
                                                        mini_batch['flow_map'], mini_batch['correspondence_mask'])

                h_, w_ = output_net['flow_estimates'][index_reso].shape[-2:]
                stats['EPE_reso_{}x{}/EPE'.format(h_, w_)] = EPE.item()
                stats['PCK_1_reso_{}x{}/EPE'.format(h_, w_)] = PCK_1.item()
                stats['PCK_3_reso_{}x{}/EPE'.format(h_, w_)] = PCK_3.item()
                stats['PCK_5_reso_{}x{}/EPE'.format(h_, w_)] = PCK_5.item()

            stats['best_value'] = stats['EPE_reso_{}x{}/EPE'.format(mini_batch['flow_map'].shape[-2]//4,
                                                                    mini_batch['flow_map'].shape[-1]//4)]

        # plot images
        epoch = mini_batch['epoch']
        iter = mini_batch['iter']
        if iter < self.nbr_images_to_plot:
            training_or_validation = 'train' if training else 'val'
            base_save_dir = os.path.join(mini_batch['settings'].env.workspace_dir, mini_batch['settings'].project_path,
                                         'plot', training_or_validation)
            if not os.path.isdir(base_save_dir):
                os.makedirs(base_save_dir)

            if mini_batch['sparse'][0]:
                _ = plot_sparse_keypoints(base_save_dir, 'epoch{}_batch{}'.format(epoch, iter), mini_batch,
                                          mini_batch['flow_map'], output_net['flow_estimates'][-1])
            else:
                _ = plot_basenet_during_training(base_save_dir, 'epoch{}_batch{}'.format(epoch, iter),
                                                 mini_batch, mini_batch['flow_map'],
                                                 output_net['flow_estimates'][-1],
                                                 mask_gt=mini_batch['mask'],
                                                 uncertainty_info=
                                                 output_net['uncertainty_estimates'][-1] if
                                                 'uncertainty_estimates' in list(output_net.keys()) else None,
                                                 warping_mask=output_net['warping_mask'] if 'warping_mask' in
                                                              list(output_net.keys()) else None,
                                                 )

        return loss, stats


class GLUNetBasedActor(BaseActor):
    """Actor for training the GLU-Net based networks with a self-supervised or supervised strategy."""

    def __init__(self, net, objective, objective_256, batch_processing, best_val_epe=True, nbr_images_to_plot=1):
        """
        Args:
            net: The network to train
            objective: The loss function to apply to the H-Net
            objective_256: The loss function to apply to the L-Net
            batch_processing: A processing class which performs the necessary processing of the batched data.
            best_val_epe: use AEPE for best value, instead can use PCK
            nbr_images_to_plot: number of images to plot per epoch
        """
        super().__init__(net, objective, batch_processing)

        self.batch_processing = batch_processing
        self.objective_256 = objective_256
        self.nbr_images_to_plot = nbr_images_to_plot
        self.best_val_epe = best_val_epe

    def __call__(self, mini_batch, training):
        """
        args:
            mini_batch: The mini batch input data, should at least contain the fields 'source_image', 'target_image',
                        'flow_map',  'mask', 'source_image_256', 'target_image_256', 'flow_map_256',
                        'mask_256', 'correspondence_mask'

                        'flow_map' is the ground-truth flow relating the target to the source. 'mask' is the mask
                        where the loss will be applied (in coordinate system of the target).
                        Similar for the 256x256 tensors.
            training: bool indicating if we are in training or evaluation mode
        returns:
            loss: the training loss
            stats: dict containing detailed losses
        """
        epoch = mini_batch['epoch']
        iter = mini_batch['iter']

        # Run network
        mini_batch = self.batch_processing(mini_batch)  # also put to GPU there
        output_net_256, output_net_original = self.net(mini_batch['target_image'], mini_batch['source_image'],
                                                       mini_batch['target_image_256'], mini_batch['source_image_256'])

        loss_o, stats_o = self.objective(output_net_original, mini_batch['flow_map'], mask=mini_batch['mask'])
        loss_256, stats_256 = self.objective_256(output_net_256, mini_batch['flow_map_256'], mask=mini_batch['mask_256'])
        loss = loss_o + loss_256

        # Log stats
        stats = merge_dictionaries([stats_o, stats_256])
        stats['Loss_H_Net/total'] = loss_o.item()
        stats['Loss_L_Net/total'] = loss_256.item()
        stats['Loss/total'] = loss.item()

        # Calculates validation stats
        if not training:
            b, _, h_original, w_original = mini_batch['flow_map'].shape
            for index_reso_original in range(len(output_net_original['flow_estimates'])):
                EPE, PCK_1, PCK_3, PCK_5 = real_metrics(output_net_original['flow_estimates'][-(index_reso_original+1)],
                                                        mini_batch['flow_map'], mini_batch['correspondence_mask'])
                h_, w_ = output_net_original['flow_estimates'][-(index_reso_original+1)].shape[-2:]
                stats['EPE_HNet_reso_{}x{}/EPE'.format(h_, w_)] = EPE.item()
                stats['PCK_1_HNet_reso_{}x{}/EPE'.format(h_, w_)] = PCK_1.item()
                stats['PCK_3_HNet_reso_{}x{}/EPE'.format(h_, w_)] = PCK_3.item()
                stats['PCK_5_HNet_reso_{}x{}/EPE'.format(h_, w_)] = PCK_5.item()

            for index_reso_256 in range(len(output_net_256['flow_estimates'])):
                EPE = realEPE(output_net_256['flow_estimates'][-(index_reso_256+1)], mini_batch['flow_map'],
                              mini_batch['correspondence_mask'],
                              ratio_x=float(w_original) / 256.0,
                              ratio_y=float(h_original) / 256.0)
                h_, w_ = output_net_256['flow_estimates'][-(index_reso_256+1)].shape[-2:]
                stats['EPE_LNet_reso_{}x{}/EPE'.format(h_, w_)] = EPE.item()

            if self.best_val_epe:
                stats['best_value'] = stats['EPE_HNet_reso_{}x{}/EPE'.format(h_original//4, w_original//4)]
            else:
                stats['best_value'] = - stats['PCK_1_HNet_reso_{}x{}/EPE'.format(h_original//4, w_original//4)]

        # plot images
        if iter < self.nbr_images_to_plot:
            training_or_validation = 'train' if training else 'val'
            base_save_dir = os.path.join(mini_batch['settings'].env.workspace_dir,
                                         mini_batch['settings'].project_path,
                                         'plot', training_or_validation)
            if not os.path.isdir(base_save_dir):
                os.makedirs(base_save_dir)

            if mini_batch['sparse'][0]:
                _ = plot_sparse_keypoints_GLUNet(base_save_dir, epoch, iter,
                                                 mini_batch['source_image'], mini_batch['target_image'],
                                                 mini_batch['source_image_256'], mini_batch['target_image_256'],
                                                 mini_batch['flow_map'], mini_batch['flow_map_256'],
                                                 output_net_original['flow_estimates'][-1],
                                                 output_net_256['flow_estimates'][-1],
                                                 normalization=True,
                                                 uncertainty_info_original=output_net_original['uncertainty_estimates'][
                                                     -1] if 'uncertainty_estimates' in list(output_net_original.keys()) else None,
                                                 uncertainty_info_256=output_net_256['uncertainty_estimates'][-1]
                                                 if 'uncertainty_estimates' in list(output_net_original.keys()) else None)
            else:
                _ = plot_during_training_with_uncertainty(base_save_dir, epoch, iter,
                                                          mini_batch['source_image'], mini_batch['target_image'],
                                                          mini_batch['source_image_256'],
                                                          mini_batch['target_image_256'],
                                                          mini_batch['flow_map'], mini_batch['flow_map_256'],
                                                          output_net=output_net_original['flow_estimates'][-1],
                                                          output_net_256=output_net_256['flow_estimates'][-1],
                                                          mask=mini_batch['mask'], mask_256=mini_batch['mask_256'],
                                                          uncertainty_info_original=
                                                          output_net_original['uncertainty_estimates'][-1] if
                                                          'uncertainty_estimates' in list(output_net_original.keys()) else None,
                                                          uncertainty_info_256=output_net_256['uncertainty_estimates'][-1]
                                                          if 'uncertainty_estimates' in list(output_net_original.keys()) else None)

        return loss, stats
