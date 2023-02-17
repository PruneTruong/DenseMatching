import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import torch.nn as nn
import imageio
import numpy as np
import sys

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from utils_flow.visualization_utils import overlay_semantic_mask
from utils_flow.flow_and_mapping_operations import get_gt_correspondence_mask
from datasets.geometric_matching_datasets.dataset_video import DatasetNoGT
from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from model_selection import select_model
from models.inference_utils import estimate_mask
from utils_data.image_transforms import ArrayToTensor
from validation.test_parser import define_model_parser
from demos.utils import make_matching_and_warping_plot_fast
from utils_flow.visualization_utils import make_and_save_video


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Matching of a sequence according to middle frame')
    # Paths
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to folder containing image sequence. ')
    define_model_parser(parser)  # model parameters
    parser.add_argument('--pre_trained_model', type=str, required=True,
                        help='Name of pre trained model')
    parser.add_argument('--name_of_sequence', type=str, required=True, help='Name to give to the sequence')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to directory to save the results')
    parser.add_argument('--save_individual_outputs', action='store_true', help='Save individual outputs.')
    parser.add_argument('--start', type=int, default=0, help='Start of the sequence')
    parser.add_argument('--middle', type=int, default=None, help='Position of reference frame, to which all the other'
                                                                 'images are matched to. ')
    parser.add_argument('--end', type=int, default=None, help='End of the sequence')
    parser.add_argument(
        '--save_video', action='store_true',
        help='Save output (with match visualizations) to a video.')
    parser.add_argument(
        '--mask_uncertain_regions', action='store_true',
        help='When warping the source to the target image, we zeros out the regions for which the matches are '
             'not predicted as confident.')

    # selection of confident match points
    parser.add_argument('--confident_mask_type', default='proba_interval_1_above_10', type=str,
                        help='mask type for filtering confident matches (when confidence is between 0 and 1)')
    parser.add_argument(
        '--cyclic_consistency_mask_threshold', type=int, default=2,
        help='Threshold used to filter matches based on their cyclic consistency error (in pixels)'
        ' (Must be positive)')  # not used here
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # either gpu or cpu

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # image_target transformations for the dataset
    co_transform = None
    target_transform = transforms.Compose([ArrayToTensor()]) # only put channel first
    input_transform = transforms.Compose([ArrayToTensor(get_float=False)])  # only put channel first

    test_set = DatasetNoGT(args.data_dir, source_image_transform=input_transform,
                           target_image_transform=input_transform,
                           middle_image=args.middle, start=args.start, end=args.end)  # only test

    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    save_dict = {}
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    name_of_sequence = args.name_of_sequence + '_{}_{}_mid_{}'.format(test_set.start, test_set.end,
                                                                      test_set.middle_image)
    with torch.no_grad():
        # define the network to use
        network, estimate_uncertainty = select_model(
            args.model, args.pre_trained_model, args, args.optim_iter, args.local_optim_iter,
            path_to_pre_trained_models=args.path_to_pre_trained_models)

        # name of the model + pre trained model
        name_to_save = args.model
        if 'PDCNet' in args.model:
            name_to_save += '_{}'.format(args.multi_stage_type)
        name_to_save += '_{}'.format(args.pre_trained_model)

        path_to_save = os.path.join(args.save_dir, name_of_sequence, name_to_save)
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        if args.save_individual_outputs:
            if not os.path.exists(os.path.join(path_to_save, 'source')):
                os.makedirs(os.path.join(path_to_save, 'source'))
            if not os.path.exists(os.path.join(path_to_save, 'target')):
                os.makedirs(os.path.join(path_to_save, 'target'))
            if not os.path.exists(os.path.join(path_to_save, 'warped_source')):
                os.makedirs(os.path.join(path_to_save, 'warped_source'))
            if not os.path.exists(os.path.join(path_to_save, 'warped_source_masked')):
                os.makedirs(os.path.join(path_to_save, 'warped_source_masked'))
            if not os.path.exists(os.path.join(path_to_save, 'uncertainty')):
                os.makedirs(os.path.join(path_to_save, 'uncertainty'))
            if not os.path.exists(os.path.join(path_to_save, 'warped_source_overlaid')):
                os.makedirs(os.path.join(path_to_save, 'warped_source_overlaid'))

        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        for i_batch, mini_batch in pbar:
            source_img = mini_batch['source_image']
            target_img = mini_batch['target_image']
            save_name = '{}_{}'.format(name_of_sequence, i_batch)  # name of image
            if 'image_shape' in mini_batch.keys():
                h_g, w_g = mini_batch['image_shape'][:2]
            else:
                b, ch_g, h_g, w_g = target_img.shape

            flow_estimated,  uncertainty_est_padded = network.estimate_flow_and_confidence_map(
                source_img, target_img)

            mask_padded = estimate_mask(args.confident_mask_type, uncertainty_est_padded,
                                        list_item=-1)

            if 'warping_mask' in list(uncertainty_est_padded.keys()):
                # get mask from internal multi stage alignment, if it took place
                mask_warping = uncertainty_est_padded['warping_mask']

                # get the mask according to uncertainty estimation
                mask_padded *= mask_warping

            if flow_estimated.shape[2] != h_g or flow_estimated.shape[3] != w_g:
                ratio_h = float(h_g) / float(flow_estimated.shape[2])
                ratio_w = float(w_g) / float(flow_estimated.shape[3])
                flow_estimated = nn.functional.interpolate(flow_estimated, size=(h_g, w_g), mode='bilinear',
                                                           align_corners=False)
                flow_estimated[:, 0, :, :] *= ratio_w
                flow_estimated[:, 1, :, :] *= ratio_h
            assert flow_estimated.shape[2] == h_g and flow_estimated.shape[3] == w_g

            mask_padded = mask_padded * get_gt_correspondence_mask(flow_estimated)
            flow_est_x = flow_estimated.permute(0, 2, 3, 1)[:, :, :, 0]
            flow_est_y = flow_estimated.permute(0, 2, 3, 1)[:, :, :, 1]

            I_2_est_remapped = remap_using_flow_fields(source_img[0].permute(1, 2, 0).cpu().numpy(),
                                                       flow_est_x[0].cpu().numpy(),
                                                       flow_est_y[0].cpu().numpy())

            source_shape = mini_batch['source_size']
            target_shape = mini_batch['target_size']

            source_img = source_img[0].permute(1, 2, 0).cpu().numpy()[:source_shape[0], :source_shape[1]]
            target_img = target_img[0].permute(1, 2, 0).cpu().numpy()[:target_shape[0], :target_shape[1]]
            mask_padded = mask_padded.squeeze().cpu().numpy()[:target_shape[0], :target_shape[1]]
            I_2_est_remapped = I_2_est_remapped[:target_shape[0], :target_shape[1]]

            text = [
                '{}'.format(name_to_save),
                'Mask type: {}'.format(args.confident_mask_type)
            ]

            if args.mask_uncertain_regions:
                mask_color = I_2_est_remapped * np.tile(np.expand_dims(mask_padded, axis=2), (1, 1, 3))
                text.append('Only confident warped regions are shown in the 3rd image.')

            else:
                mask_color = overlay_semantic_mask(I_2_est_remapped, 255 - mask_padded * 255, color=[255, 102, 51])
                text.append('Red regions are predicted uncertain in the 3rd image')

            small_text = [
                '{}'.format(save_name),
                'Start: {}, end: {}, fixed frame: {}'.format(test_set.start, test_set.end, test_set.middle_image)
            ]
            plot = make_matching_and_warping_plot_fast(source_img, target_img,
                                                       warped_and_overlay_image=mask_color, text=text,
                                                       small_text=small_text)
            imageio.imwrite('{}/{}.jpg'.format(path_to_save, save_name), plot)

            if args.save_individual_outputs:
                imageio.imwrite(os.path.join(path_to_save, 'source',  "{}_source_images_{}.jpg".format(name_of_sequence, save_name)),
                                source_img)
                imageio.imwrite(os.path.join(path_to_save, 'target', "{}_target_images_{}.jpg".format(name_of_sequence, save_name)),
                                target_img)
                imageio.imwrite(os.path.join(path_to_save, 'warped_source',
                                             "{}_warped_source_images_{}.jpg".format(name_of_sequence, save_name)),
                                I_2_est_remapped)
                imageio.imwrite(os.path.join(path_to_save, 'warped_source_masked',
                                             "{}_warped_source_images_{}.jpg".format(name_of_sequence, save_name)),
                                I_2_est_remapped * np.tile(np.expand_dims(mask_padded, axis=2), (1, 1, 3)))

                imageio.imwrite(os.path.join(path_to_save, 'warped_source_overlaid',
                                             "{}_warped_source_images_overlaid{}.jpg".format(name_of_sequence, save_name)),
                                mask_color)

        if args.save_video:
            print('Saving video...')
            make_and_save_video(path_to_save, os.path.join(path_to_save, save_name + '.mp4'), rate=10)
