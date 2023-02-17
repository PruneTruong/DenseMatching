import torch
import os


from training.losses.basic_losses import realEPE
from admin.stats import merge_dictionaries
from training.actors.base_actor import BaseActor
from training.plot.plot_GLUNet import plot_predictions
from admin.multigpu import is_multi_gpu
from training.actors.batch_processing import pre_process_image_glunet


class GLUNetImageBatchPreprocessing:
    """ Class responsible for processing the mini-batch to create the desired training inputs for GLU-Net based
    networks when training unsupervised (for example photometric loss). Only required are source and target images.
    Particularly, from the source and target images at original resolution,
    needs to create the source, target at resolution 256x256 for training the L-Net.
    """

    def __init__(self, settings, resizing=None):
        """
        Args:
            settings: settings
            resizing: resizing of the images?
        """

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        if resizing is not None:
            if not isinstance(resizing, (tuple, list)):
                resizing = (resizing, resizing)
        self.resizing = resizing

    def __call__(self, mini_batch, *args, **kwargs):
        """
        args:
            mini_batch: The mini batch input data, should at least contain the fields 'source_image', 'target_image'.
            training: bool indicating if we are in training or evaluation mode
        returns:
            TensorDict: output data block with following fields:
                        'source_image', 'target_image'
                        Can optionally also contain 'flow_map', 'correspondence_mask', 'flow_map_target_to_source',
                        'correspondence_mask_target_to_source'
        """

        if self.resizing is not None:
            mini_batch['source_image'] = torch.nn.functional.interpolate(input=mini_batch['source_image'].float()
                                                                         .to(self.device),
                                                                         size=self.resizing, mode='area')
            mini_batch['target_image'] = torch.nn.functional.interpolate(input=mini_batch['target_image'].float()
                                                                         .to(self.device),
                                                                         size=self.resizing, mode='area')

        source_image, source_image_256 = pre_process_image_glunet(mini_batch['source_image'], self.device)
        target_image, target_image_256 = pre_process_image_glunet(mini_batch['target_image'], self.device)

        # At original resolution
        if 'flow_map' in list(mini_batch.keys()):
            mini_batch['flow_map'] = mini_batch['flow_map'].to(self.device)
            mini_batch['correspondence_mask'] = mini_batch['correspondence_mask'].to(self.device)
        if 'flow_map_target_to_source' in list(mini_batch.keys()):
            mini_batch['flow_map_target_to_source'] = mini_batch['flow_map_target_to_source'].to(self.device)
            mini_batch['correspondence_mask_target_to_source'] = \
                mini_batch['correspondence_mask_target_to_source'].to(self.device)

        mini_batch['source_image'] = source_image
        mini_batch['target_image'] = target_image
        mini_batch['source_image_256'] = source_image_256
        mini_batch['target_image_256'] = target_image_256
        return mini_batch


class GLUNetPhotometricUnsupervisedActor(BaseActor):
    """Actor for training with photometric based losses GLU-Net based networks."""

    def __init__(self, net, objective, objective_256, batch_processing, nbr_images_to_plot=2):
        """
        Args:
            net: The network to train
            objective: The loss function
            batch_processing: A processing class which performs the necessary processing of the batched data.
            nbr_images_to_plot: number of images to plot per epoch
        """
        super().__init__(net, objective, batch_processing)

        self.objective_256 = objective_256
        self.nbr_images_to_plot = nbr_images_to_plot

    def __call__(self, mini_batch, training):
        """
        args:
            mini_batch: The mini batch input data, should at least contain the fields 'source_image', 'target_image'.
            training: bool indicating if we are in training or evaluation mode
        returns:
            loss: the training loss
            stats: dict containing detailed losses
        """
        # Run network
        epoch = mini_batch['epoch']
        iter = mini_batch['iter']
        mini_batch = self.batch_processing(mini_batch, net=self.net, training=training)
        net = self.net.module if is_multi_gpu(self.net) else self.net

        im_source_pyr = net.pyramid(mini_batch['source_image'], eigth_resolution=True)
        im_target_pyr = net.pyramid(mini_batch['target_image'], eigth_resolution=True)
        im_source_pyr_256 = net.pyramid(mini_batch['source_image_256'])
        im_target_pyr_256 = net.pyramid(mini_batch['target_image_256'])

        output_flow_target_to_source_256, output_flow_target_to_source = self.net(
            mini_batch['target_image'], mini_batch['source_image'],
            mini_batch['target_image_256'], mini_batch['source_image_256'],
            im_target_pyr, im_source_pyr, im_target_pyr_256, im_source_pyr_256)
        estimated_flow_target_to_source_256 = output_flow_target_to_source_256['flow_estimates']
        estimated_flow_target_to_source = output_flow_target_to_source['flow_estimates']

        output_flow_source_to_target_256, output_flow_source_to_target = self.net(
            mini_batch['source_image'], mini_batch['target_image'],
            mini_batch['source_image_256'], mini_batch['target_image_256'],
            im_source_pyr, im_target_pyr, im_source_pyr_256, im_target_pyr_256)
        estimated_flow_source_to_target_256 = output_flow_source_to_target_256['flow_estimates']
        estimated_flow_source_to_target = output_flow_source_to_target['flow_estimates']

        loss_o, stats_o = self.objective(source_image=mini_batch['source_image'],
                                         target_image=mini_batch['target_image'],
                                         estimated_flow_target_to_source=estimated_flow_target_to_source,
                                         estimated_flow_source_to_target=estimated_flow_source_to_target)
        loss_256, stats_256 = self.objective_256(source_image=mini_batch['source_image_256'],
                                                 target_image=mini_batch['target_image_256'],
                                                 estimated_flow_target_to_source=estimated_flow_target_to_source_256,
                                                 estimated_flow_source_to_target=estimated_flow_source_to_target_256)
        loss = loss_o + loss_256

        # Log stats
        stats = merge_dictionaries([stats_o, stats_256])
        stats['Loss_H_Net/total'] = loss_o.item()
        stats['Loss_L_Net/total'] = loss_256.item()
        stats['Loss/total'] = loss.item()

        # Calculates validation stats
        if not training:
            if 'flow_map' in list(mini_batch.keys()):
                b, _, h_original, w_original = mini_batch['flow_map'].shape
                for index_reso_original in range(len(estimated_flow_target_to_source)):
                    EPE = realEPE(estimated_flow_target_to_source[-(index_reso_original + 1)], mini_batch['flow_map'],
                                  mini_batch['correspondence_mask'])
                    stats['EPE_HNet_reso_{}/EPE'.format(index_reso_original)] = EPE.item()

                stats['best_value'] = stats['EPE_HNet_reso_0/EPE']

            elif 'flow_map_target_to_source' in list(mini_batch.keys()):
                mask_gt_target_to_source = mini_batch['correspondence_mask_target_to_source']
                flow_gt_target_to_source = mini_batch['flow_map_target_to_source']
                for index_reso in range(len(estimated_flow_target_to_source)):
                    EPE = realEPE(estimated_flow_target_to_source[-(index_reso + 1)],
                                  flow_gt_target_to_source, mask_gt_target_to_source)

                    stats['EPE_target_to_source_reso_{}/EPE'.format(index_reso)] = EPE.item()

                stats['best_value'] = stats['EPE_target_to_source_reso_0/EPE']
            else:
                stats['best_value'] = stats['Loss/total']

        # plot images
        epoch = mini_batch['epoch']
        iter = mini_batch['iter']
        if iter < self.nbr_images_to_plot:
            training_or_validation = 'train' if training else 'val'
            base_save_dir = os.path.join(mini_batch['settings'].env.workspace_dir,
                                         mini_batch['settings'].project_path,
                                         'plot', training_or_validation)
            if not os.path.isdir(base_save_dir):
                os.makedirs(base_save_dir)

            _ = plot_predictions(base_save_dir, epoch, iter,
                                 mini_batch['source_image'], mini_batch['target_image'],
                                 mini_batch['source_image_256'],
                                 mini_batch['target_image_256'],
                                 output_net=output_flow_target_to_source['flow_estimates'][-1],
                                 output_net_256=
                                 output_flow_target_to_source_256['flow_estimates'][-1])
        return loss, stats






