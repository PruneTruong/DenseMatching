import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import datasets
from utils_data.image_transforms import ArrayToTensor
import json
import admin.settings as ws_settings
from admin.stats import merge_dictionaries
from validation.evaluate_per_dataset import run_evaluation_generic, run_evaluation_kitti, run_evaluation_megadepth_or_robotcar, \
    run_evaluation_sintel, run_evaluation_eth3d, run_evaluation_semantic
from model_selection import select_model
dataset_names = sorted(name for name in datasets.__all__)
dataset_names.append('VOC')


# Argument parsing
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def main(args, settings):
    # image transformations for the dataset
    co_transform = None
    target_transform = transforms.Compose([ArrayToTensor()])  # only put channel first
    input_transform = transforms.Compose([ArrayToTensor(get_float=False)])  # only put channel first

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args.network_type == 'PDCNet' and args.multi_stage_type != 'direct':
        # add sub-possibility with mask threshold of internal multi-stage alignment
        save_dir = os.path.join(save_dir, 'mask_for_homo_align_' + args.mask_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    save_dict = {}
    for pre_trained_model_type in args.pre_trained_models:
        # define the network to use
        network, estimate_uncertainty = select_model(
            args.model, pre_trained_model_type, args, args.optim_iter, local_optim_iter,
            path_to_pre_trained_models=args.path_to_pre_trained_models)

        name_to_save = args.model
        with torch.no_grad():
            # choosing the different dataset !
            path_to_save = os.path.join(save_dir, '{}_{}'.format(name_to_save, pre_trained_model_type))
            if 'PDCNet' in args.model or 'GOCor' in args.model:
                path_to_save = path_to_save + '_optim{}_localoptin{}'.format(args.optim_iter, local_optim_iter)

            if not os.path.isdir(path_to_save) and args.plot:
                os.makedirs(path_to_save)

            if args.datasets == 'megadepth':
                output = run_evaluation_megadepth_or_robotcar(network, settings.env.megadepth,
                                                              testCSV=settings.env.megadepth_csv,
                                                              estimate_uncertainty=estimate_uncertainty,
                                                              path_to_save=path_to_save, plot=args.plot)

            elif args.datasets == 'robotcar':
                output = run_evaluation_megadepth_or_robotcar(network, settings.env.robotcar,
                                                              testCSV=settings.env.robotcar_csv,
                                                              estimate_uncertainty=estimate_uncertainty,
                                                              path_to_save=path_to_save, plot=args.plot)

            elif 'hp' in args.datasets:
                original_size = True
                if args.datasets == 'hp-240':
                    original_size = False
                number_of_scenes = 5 + 1
                list_of_outputs = []
                # loop over scenes (1-2, 1-3, 1-4, 1-5, 1-6)
                for id, k in enumerate(range(2, number_of_scenes + 2)):
                    if id == 5:
                        _, test_set = datasets.HPatchesdataset(settings.env.hp,
                                                               os.path.join('assets',
                                                                            'hpatches_all.csv'.format(k)), \
                                                               input_transform, target_transform, co_transform,
                                                               original_size=original_size, split=0)
                    else:
                        _, test_set = datasets.HPatchesdataset(settings.env.hp,
                                                               os.path.join('assets',
                                                                            'hpatches_1_{}.csv'.format(k)), \
                                                               input_transform, target_transform, co_transform,
                                                               original_size=original_size, split=0)
                    test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
                    output_scene = run_evaluation_generic(network, test_dataloader, device,
                                                          estimate_uncertainty=estimate_uncertainty)
                    list_of_outputs.append(output_scene)

                output = {'scene_1': list_of_outputs[0], 'scene_2': list_of_outputs[1], 'scene_3': list_of_outputs[2],
                          'scene_4': list_of_outputs[3], 'scene_5': list_of_outputs[4], 'all': list_of_outputs[5]}

            elif args.datasets == 'kitti2012':
                _, test_set = datasets.KITTI_occ(settings.env.kitti2012, source_image_transform=input_transform,
                                                 target_image_transform=input_transform,
                                                 flow_transform=target_transform, co_transform=co_transform, split=0)
                test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
                output = run_evaluation_kitti(network, test_dataloader, device,
                                              estimate_uncertainty=estimate_uncertainty, path_to_save=path_to_save,
                                              plot=args.plot)

            elif args.datasets == 'kitti2015':
                _, test_set = datasets.KITTI_occ(settings.env.kitti2015, source_image_transform=input_transform,
                                                 target_image_transform=input_transform,
                                                 flow_transform=target_transform, co_transform=co_transform, split=0)
                test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
                output = run_evaluation_kitti(network, test_dataloader, device,
                                              estimate_uncertainty=estimate_uncertainty, path_to_save=path_to_save,
                                              plot=args.plot)

            elif args.datasets == 'TSS':
                output = {}
                for sub_data in ['FG3DCar', 'JODS', 'PASCAL']:
                    path_to_save_ = os.path.join(path_to_save, sub_data)
                    if not os.path.exists(path_to_save_) and args.plot:
                        os.makedirs(path_to_save_)
                    _, test_set = datasets.TSS(os.path.join(settings.env.tss, sub_data),
                                               source_image_transform=input_transform, target_image_transform=input_transform,
                                               flow_transform=target_transform, co_transform=co_transform, split=0)
                    test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
                    results = run_evaluation_semantic(network, test_dataloader, device, estimate_uncertainty=estimate_uncertainty,
                                                      flipping_condition=args.flipping_condition,
                                                      path_to_save=path_to_save_, plot=args.plot)
                    output[sub_data] = results

            elif args.datasets == 'PFPascal':
                test_set = datasets.PFPascalDataset(settings.env.PFPascal_root, source_image_transform=input_transform,
                                                    target_image_transform=input_transform,
                                                    flow_transform=target_transform, pck_procedure='image_size')
                test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
                output = run_evaluation_semantic(network, test_dataloader, device, estimate_uncertainty=estimate_uncertainty,
                                                 flipping_condition=args.flipping_condition,
                                                 path_to_save=path_to_save, plot=args.plot)

            elif args.datasets == 'PFWillow':
                test_set = datasets.PFWillowDataset(settings.env.PFWillow_root, source_image_transform=input_transform,
                                                    target_image_transform=input_transform, flow_transform=target_transform)
                test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
                output = run_evaluation_semantic(network, test_dataloader, device, estimate_uncertainty=estimate_uncertainty,
                                                 flipping_condition=args.flipping_condition,
                                                 path_to_save=path_to_save, plot=args.plot)

            elif args.datasets == 'sintel':
                output = {}
                for dstype in ['clean', 'final']:
                    _, test_set = datasets.mpi_sintel(settings.env.sintel, source_image_transform=input_transform,
                                                      target_image_transform=input_transform,
                                                      flow_transform=target_transform, co_transform=co_transform, split=0,
                                                      load_occlusion_mask=True, dstype=dstype)
                    test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
                    results = run_evaluation_sintel(network, test_dataloader, device,
                                                    estimate_uncertainty=estimate_uncertainty)
                    output[dstype] = results

            elif args.datasets == 'eth3d':
                output = run_evaluation_eth3d(network, settings.env.eth3d, input_transform, target_transform, co_transform,
                                              device, estimate_uncertainty=estimate_uncertainty)

            else:
                raise ValueError('Unknown dataset, {}'.format(args.datasets))

            save_dict['{}'.format(pre_trained_model_type)] = output

    if 'PDCNet' in args.model or 'GOCor' in args.model:
        name_save_metrics = 'metrics_{}_iter_{}_{}'.format(name_to_save, args.optim_iter, local_optim_iter)
    else:
        name_save_metrics = 'metrics_{}'.format(name_to_save)

    path_file = '{}/{}.txt'.format(save_dir, name_save_metrics)
    if os.path.exists(path_file):
        with open(path_file, 'r') as outfile:
            save_dict_existing = json.load(outfile)
        save_dict = merge_dictionaries([save_dict_existing, save_dict])

    with open(path_file, 'w') as outfile:
        json.dump(save_dict, outfile, ensure_ascii=False, separators=(',', ':'))
        print('written to file ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Correspondence evaluation')
    # Paths
    parser.add_argument('--datasets', type=str, required=True, help='Dataset name')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to use')
    parser.add_argument('--flipping_condition', dest='flipping_condition',  default=False, type=boolean_string,
                        help='Apply flipping condition for semantic data and GLU-Net-based networks ? ')
    parser.add_argument('--optim_iter', type=int, default=3,
                        help='number of optim iter for global GOCor, when applicable')
    parser.add_argument('--local_optim_iter', dest='local_optim_iter', default=None,
                        help='number of optim iter for local GOCor, when applicable')
    parser.add_argument('--path_to_pre_trained_models', type=str, default='pre_trained_models/',
                        help='path to the folder containing pre trained models')
    parser.add_argument('--pre_trained_models', nargs='+', required=True,
                        help='name of pre trained models')
    parser.add_argument('--plot', default=False, type=boolean_string,
                        help='plot? default is False')
    parser.add_argument('--save_dir', type=str, default='evaluation/',
                        help='path to directory to save the text files and results')
    parser.add_argument('--seed', type=int, default=1984, help='Pseudo-RNG seed')

    subprasers = parser.add_subparsers(dest='network_type')
    PDCNet = subprasers.add_parser('PDCNet', help='inference parameters for PDCNet')
    PDCNet.add_argument(
        '--confidence_map_R', default=1.0, type=float,
        help='R used for confidence map computation',
    )
    PDCNet.add_argument(
        '--multi_stage_type', default='direct', type=str, choices=['direct', 'homography_from_last_level_uncertainty',
                                                                   'homography_from_quarter_resolution_uncertainty',
                                                                   'multiscale_homo_from_quarter_resolution_uncertainty'],
        help='multi stage type',
    )
    PDCNet.add_argument(
        '--ransac_thresh', default=1.0, type=float,
        help='ransac threshold used for multi-stages alignment',
    )
    PDCNet.add_argument(
        '--mask_type', default='proba_interval_1_above_5', type=str,
        help='mask computation for multi-stage alignment',
    )
    PDCNet.add_argument(
        '--homography_visibility_mask', default=True, type=boolean_string,
        help='apply homography visibility mask for multi-stage computation ?',
    )
    PDCNet.add_argument('--scaling_factors', type=float, nargs='+', default=[0.5, 0.6, 0.88, 1, 1.33, 1.66, 2],
                        help='scaling factors')
    args = parser.parse_args()

    if not args.local_optim_iter:
        local_optim_iter = args.optim_iter
    else:
        local_optim_iter = int(args.local_optim_iter)

    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # either gpu or cpu

    # settings containing paths to datasets
    settings = ws_settings.Settings()
    main(args=args, settings=settings)
