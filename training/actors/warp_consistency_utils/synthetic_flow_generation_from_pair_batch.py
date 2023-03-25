import torch
import torch.nn.functional as F
import cv2
import numpy as np


from validation.utils import matches_from_flow
from utils_data.geometric_transformation_sampling.homography_parameters_sampling import RandomHomography, \
    from_homography_to_pixel_wise_mapping
from utils_flow.pixel_wise_mapping import warp
from utils_flow.flow_and_mapping_operations import convert_mapping_to_flow
from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from matplotlib import pyplot as plt
from utils_data.geometric_transformation_sampling.synthetic_warps_sampling import SynthecticAffHomoTPSTransfo


# for debugging
def plot_synthetic_flow_creation(save_path, epoch, batch, source_image, target_image,
                                 estimated_flow_target_to_source, estimated_flow_source_to_target, mask):

    flow_target_to_source_x = estimated_flow_target_to_source.detach().permute(0, 2, 3, 1)[0, :, :, 0]
    flow_target_to_source_y = estimated_flow_target_to_source.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    flow_source_to_target_x = estimated_flow_source_to_target.detach().permute(0, 2, 3, 1)[0, :, :, 0]
    flow_source_to_target_y = estimated_flow_source_to_target.detach().permute(0, 2, 3, 1)[0, :, :, 1]

    image_1 = source_image.detach()[0].cpu().permute(1, 2, 0)
    image_2 = target_image.detach()[0].cpu().permute(1, 2, 0)
    remapped_gt = remap_using_flow_fields(image_1.numpy(),
                                          flow_target_to_source_x.cpu().numpy(),
                                          flow_target_to_source_y.cpu().numpy())
    remapped_est = remap_using_flow_fields(image_2.numpy(), flow_source_to_target_x.cpu().numpy(),
                                           flow_source_to_target_y.cpu().numpy())

    fig, axis = plt.subplots(2, 5, figsize=(20, 20))
    axis[0][0].imshow(image_1.numpy(), vmin=0, vmax=1)
    axis[0][0].set_title("original reso: \nsource image")
    axis[0][1].imshow(image_2.numpy(), vmin=0, vmax=1)
    axis[0][1].set_title("original reso: \ntarget image")
    if mask is not None:
        mask = mask.detach()[0].squeeze().cpu().numpy().astype(np.float32)
    axis[0][2].imshow(mask, vmin=0.0, vmax=1.0)
    axis[0][2].set_title("original reso: \nmask applied during training")
    axis[0][3].imshow(remapped_gt)
    axis[0][3].set_title("original reso : \nsource remapped with ground truth")
    axis[0][4].imshow(remapped_est)
    axis[0][4].set_title("original reso: \nsource remapped with network")
    fig.savefig('{}/epoch{}_batch{}.png'.format(save_path, epoch, batch),
                bbox_inches='tight')
    plt.close(fig)
    return True


def get_flow_from_predictions(estimated_flow_target_to_source, consistent_matches, scaling,
                              size_output_flow, min_nbr_points=50):

    pts_s, pts_t = matches_from_flow(estimated_flow_target_to_source, consistent_matches)  # homo from source to target
    pts_s *= scaling
    pts_t *= scaling

    flow_gt = None
    if len(pts_s) > min_nbr_points:
        H, inliers = cv2.findHomography(pts_s, pts_t, cv2.RANSAC, 3.0, maxIters=2000)
        if H is not None:
            mapping_from_homography_x, mapping_from_homography_y = from_homography_to_pixel_wise_mapping(
                size_output_flow, H)
            mapping_from_homography_numpy = np.dstack((mapping_from_homography_x, mapping_from_homography_y))
            flow_gt = convert_mapping_to_flow(torch.from_numpy(mapping_from_homography_numpy)
                                              .unsqueeze(0).permute(0, 3, 1, 2))
    return flow_gt


class GetSyntheticFlowFromNetPredictions:
    """For all image in a batch, computes the homography relating each pair of images based on the predicted flow
    field by a network. If the network prediction is not good enough to compute a homography, a synthetic
    transformation is instead sampled for each image pair independently, according to provided
    sampling and generation module. """
    def __init__(self, settings, size_output_flow, alternative_synthetic_flow_generator, get_net_prediction,
                 pre_process_data_for_net_prediction=True,
                 image_prediction_size=256, first_epochs_alternative=1, cyclic_cons_thresh=10.0, min_nbr_points=50):
        """
        Args:
            settings:
            size_output_flow:
            alternative_synthetic_flow_generator: synthetic flow generator to use in case the network predictions are
                                                  not good enough to estimate a homography
            get_net_prediction: class returning the network predictions
            pre_process_data_for_net_prediction: module to pre process the data, given as input to the net predictor
                                                 module
            image_prediction_size: size of the images used for network prediction
            first_epochs_alternative: epochs for which to use directly the alternative flow sampling generator instead
                                      of the network predictions
            cyclic_cons_thresh: threshold used for cyclic consistency mask
            min_nbr_points: minimum number of points for which cyclic consistency condition is respected, in order to
                            compute the homography
        """
        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        if not isinstance(image_prediction_size, (tuple, list)):
            image_prediction_size = (image_prediction_size, image_prediction_size)
        self.image_prediction_size = image_prediction_size

        if not isinstance(size_output_flow, (tuple, list)):
            size_output_flow = (size_output_flow, size_output_flow)
        self.size_output_flow = size_output_flow

        self.get_net_prediction = get_net_prediction
        self.pre_process_data_for_net_prediction = pre_process_data_for_net_prediction
        # when prediction is not accurate enough, use other synthetic flow generator
        # (could be random, or old CAD transformations and so on)
        self.alternative_synthetic_flow_generator = alternative_synthetic_flow_generator
        self.first_epochs_alternative = first_epochs_alternative
        self.min_nbr_points = min_nbr_points
        self.cyclic_cons_thresh = cyclic_cons_thresh

    def __call__(self, mini_batch, net, training=True, *args, **kwargs):

        # take original images, have them at resolution 520x520
        source_image = mini_batch['source_image'].to(self.device) # not pre processed yet
        b, _, h, w = source_image.shape
        target_image = mini_batch['target_image'].to(self.device)

        if mini_batch['epoch'] >= self.first_epochs_alternative:
            # resized original images to resolution to get prediction
            source_image_for_prediction = F.interpolate(source_image.float(),
                                                        self.image_prediction_size, mode='area')
            target_image_for_prediction = F.interpolate(target_image.float(),
                                                        self.image_prediction_size, mode='area')

            # get flow predictions
            estimated_flow_target_to_source, estimated_flow_source_to_target = self.get_net_prediction(
                source_image_for_prediction, target_image_for_prediction, net, self.device,
                self.pre_process_data_for_net_prediction)
            b, _, h_f, w_f = estimated_flow_target_to_source.shape
            cyclic_consistency = estimated_flow_target_to_source + warp(estimated_flow_source_to_target,
                                                                        estimated_flow_target_to_source)

            # get mask of valid correspondences
            consistent_matches = torch.norm(cyclic_consistency, p=2, dim=1, keepdim=True).le(self.cyclic_cons_thresh)
            # b, 1, h, w

            # get scaling
            scaling_x = float(self.size_output_flow[1]) / float(w_f)
            scaling_y = float(self.size_output_flow[0]) / float(h_f)
            scaling = [scaling_x, scaling_y]

        # then currently must do image by image to check that RANSAC gets the correct flow:
        synthetic_flow = []
        for b_ in range(b):

            flow_gt_ = None
            if mini_batch['epoch'] >= self.first_epochs_alternative:
                # only try the predictions after a certain amount of epochs
                flow_gt_ = get_flow_from_predictions(estimated_flow_target_to_source[b_].unsqueeze(0),
                                                     consistent_matches[b_], scaling,
                                                     self.size_output_flow, min_nbr_points=self.min_nbr_points)
            # would be size 1, 2, h, w
            if flow_gt_ is None:
                # prediction was not accurate !
                flow_gt_ = self.alternative_synthetic_flow_generator(mini_batch=mini_batch, training=training)
                flow_gt_ = flow_gt_.to(self.device).requires_grad_(False)
                # must be getting a unique flow there !

                if flow_gt_.shape[1] != 2:
                    flow_gt_.permute(0, 3, 1, 2)
                _, _, h_f, w_f = flow_gt_.shape

                ratio_x = float(self.size_output_flow[1]) / float(w_f)
                ratio_y = float(self.size_output_flow[0]) / float(h_f)
            else:
                if flow_gt_.shape[1] != 2:
                    flow_gt_.permute(0, 3, 1, 2)
                ratio_x = float(self.size_output_flow[1]) / float(self.image_prediction_size[1])
                ratio_y = float(self.size_output_flow[0]) / float(self.image_prediction_size[0])
            flow_gt_ = flow_gt_.to(self.device)

            # resize the unit flow to desired shape if needed
            if h_f != self.size_output_flow[0] or w_f != self.size_output_flow[1]:
                # reshape and rescale to desired size
                flow_gt_ = F.interpolate(flow_gt_, self.size_output_flow, mode='bilinear', align_corners=False)
                # interpolation to image_prediction_size is taken into account already
                flow_gt_[:, 0] *= ratio_x
                flow_gt_[:, 1] *= ratio_y
            synthetic_flow.append(flow_gt_)

        flow_gt = torch.cat(synthetic_flow, dim=0)
        return flow_gt


class GetSyntheticFlowFromCAD:
    """For all image in a batch, get the flow field from CAD pre-loaded dataset. """
    def __init__(self, settings, size_output_flow, homo_dataloader, homo_dataloader_eval=None):
        """
        Args:
            settings:
            size_output_flow:  size of the outputted flow field
            homo_dataloader: dataloader used for getting the saved flow fields corresponding to CAD, used during
                             training
            homo_dataloader_eval: dataloader used for getting the saved flow fields corresponding to CAD, used during
                                  validation
        """

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.homo_dataloader = homo_dataloader
        self.homography_dataloader_iterator = iter(homo_dataloader)

        if homo_dataloader_eval is not None:
            self.homo_dataloader_eval = homo_dataloader_eval
            self.homography_dataloader_eval_iterator = iter(homo_dataloader_eval)

        if not isinstance(size_output_flow, (tuple, list)):
            size_output_flow = (size_output_flow, size_output_flow)
        self.size_output_flow = size_output_flow

    def __call__(self, training=False, *args, **kwargs):

        # at each call, sample from the next of the dataset
        if training:
            try:
                flow_gt = next(self.homography_dataloader_iterator)['flow_map'].to(self.device).requires_grad_(False)
            except StopIteration:
                self.homography_dataloader_iterator = iter(self.homo_dataloader)  # from the start
                flow_gt = next(self.homography_dataloader_iterator)['flow_map'].to(self.device).requires_grad_(False)
        else:
            try:
                flow_gt = next(self.homography_dataloader_eval_iterator)['flow_map'].to(self.device).requires_grad_(False)
            except StopIteration:
                self.homography_dataloader_eval_iterator = iter(self.homo_dataloader_eval)  # from the start
                flow_gt = next(self.homography_dataloader_eval_iterator)['flow_map'].to(self.device).requires_grad_(False)

        if flow_gt.shape[1] != 2:
            flow_gt.permute(0, 3, 1, 2)

        # reshape and rescale to desired size
        bs, _, h_f, w_f = flow_gt.shape
        flow_gt = F.interpolate(flow_gt, self.size_output_flow, mode='bilinear', align_corners=False)
        flow_gt[:, 0] *= float(self.size_output_flow[1]) / float(w_f)
        flow_gt[:, 1] *= float(self.size_output_flow[0]) / float(h_f)
        return flow_gt


class GetRandomSyntheticHomographyFlow:
    """ For all image in a batch, computes the flow fields corresponding to random homography transforms. A different
    homography is sampled for each image pair in the batch.They have a size corresponding to argument size_output_flow.
    The range of sampling parameters are provided through the homo sampling module. """
    def __init__(self, settings, size_output_flow, homo_sampling_module=None):
        """
        Args:
            settings:
            size_output_flow: size of the outputted flow field
            homo_sampling_module: module to sample the homogaphy transform parameters.
                                  If None, we use the default module.
        """

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        if homo_sampling_module is None:
            homo_sampling_module = RandomHomography(p_flip=0.0, max_rotation=10.0, max_shear=0.1,
                                                    max_scale=1.1, max_ar_factor=0.1,
                                                    min_perspective=0.0005, max_perspective=0.0009,
                                                    max_translation=10, pad_amount=0)
        self.homography_transform = homo_sampling_module

        if not isinstance(size_output_flow, (tuple, list)):
            size_output_flow = (size_output_flow, size_output_flow)
        self.size_output_flow = size_output_flow

    def __call__(self,  mini_batch, training=True, *args, **kwargs):

        with torch.no_grad():
            source_image = mini_batch['source_image'].to(self.device)
            b = source_image.shape[0]
            synthetic_flow = []

            for b_ in range(b):
                do_flip, rot, shear_values, scale_factors, perpective_factor, tx, ty = self.homography_transform.roll()
                H = self.homography_transform._construct_t_mat(self.size_output_flow, do_flip, rot,
                                                               shear_values, scale_factors, tx=tx, ty=ty,
                                                               perspective_factor=perpective_factor)
                mapping_from_homography_x, mapping_from_homography_y = from_homography_to_pixel_wise_mapping(
                    self.size_output_flow, H)
                mapping_from_homography_numpy = np.dstack((mapping_from_homography_x, mapping_from_homography_y))
                flow_gt_ = convert_mapping_to_flow(torch.from_numpy(mapping_from_homography_numpy)
                                                   .unsqueeze(0).permute(0, 3, 1, 2))

                synthetic_flow.append(flow_gt_)

            flow_gt = torch.cat(synthetic_flow, dim=0)
        return flow_gt.to(self.device).requires_grad_(False)


class GetRandomSyntheticAffHomoTPSFlow:
    """ For all image in a batch, computes flow fields corresponding to random Affine, homography or TPS transforms.
    They have a size corresponding to argument size_output_flow.
    A different geometric transformation is sampled for each image pair in the batch.
    The range of sampling parameters are provided through the transfo sampling module. """
    def __init__(self, settings, size_output_flow, transfo_sampling_module=None):
        """
        Args:
            settings:
            size_output_flow:  size of the outputted flow field
            transfo_sampling_module: module to sample the transform parameters. If None, we use the default module.
        """

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        if not isinstance(size_output_flow, (tuple, list)):
            size_output_flow = (size_output_flow, size_output_flow)
        self.size_output_flow = size_output_flow

        if transfo_sampling_module is None:
            transfo_sampling_module = SynthecticAffHomoTPSTransfo(size_output_flow=self.size_output_flow, random_t=0.25,
                                                                  random_s=0.5, random_alpha=np.pi / 12,
                                                                  random_t_hom=0.4, random_t_tps=0.4,
                                                                  transformation_types=['affine', 'hom',
                                                                                        'tps', 'afftps'])
            # this is quite strong transformations
        self.sample_transfo = transfo_sampling_module

    def __call__(self,  mini_batch, training=True, *args, **kwargs):

        with torch.no_grad():
            source_image = mini_batch['source_image'].to(self.device)
            b = source_image.shape[0]
            synthetic_flow = []

            for b_ in range(b):
                flow_gt_ = self.sample_transfo()
                synthetic_flow.append(flow_gt_)

            flow_gt = torch.cat(synthetic_flow, dim=0)
        return flow_gt.to(self.device).requires_grad_(False)

