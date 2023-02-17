import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import datasets
import json

from utils_data.image_transforms import ArrayToTensor
from validation.flow_evaluation.evaluate_per_dataset import (run_evaluation_generic, run_evaluation_kitti,
                                                             run_evaluation_megadepth_or_robotcar,
                                                             run_evaluation_sintel, run_evaluation_eth3d,
                                                             run_evaluation_semantic, run_evaluation_caltech)
from model_selection import select_model
import admin.settings as ws_settings
from admin.stats import merge_dictionaries
from validation.test_parser import define_model_parser, boolean_string

torch.set_grad_enabled(False)


def main(args, settings):
    # image transformations for the dataset
    co_transform = None
    target_transform = transforms.Compose([ArrayToTensor()])  # only put channel first
    input_transform = transforms.Compose([ArrayToTensor(get_float=False)])  # only put channel first

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args.network_type == 'PDCNet' and ('d' not in args.multi_stage_type.lower()):
        # add sub-possibility with mask threshold of internal multi-stage alignment
        save_dir = os.path.join(save_dir, 'mask_for_homo_align_' + args.mask_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    name_to_save = args.model
    save_dict = {}
    for pre_trained_model_type in args.pre_trained_models:
        # define the network to use
        network, estimate_uncertainty = select_model(
            args.model, pre_trained_model_type, args, args.optim_iter, local_optim_iter,
            path_to_pre_trained_models=args.path_to_pre_trained_models)

        # for networks that inherently predict an uncertainty measure, automatically evaluate it. Can optionally
        # evaluate uncertainty based on cyclic consistency error
        estimate_uncertainty = estimate_uncertainty or args.compute_metrics_uncertainty

        # choosing the different dataset !
        path_to_save = os.path.join(save_dir, '{}_{}'.format(name_to_save, pre_trained_model_type))
        if 'gocor' in args.model.lower() or 'PDCNet' in args.model:
            path_to_save = path_to_save + '_globaloptim{}_localoptim{}'.format(args.optim_iter, local_optim_iter)

        if not os.path.isdir(path_to_save) and (args.plot or args.plot_100):
            os.makedirs(path_to_save)

        if args.dataset == 'megadepth':
            output = run_evaluation_megadepth_or_robotcar(network, settings.env.megadepth,
                                                          path_to_csv=settings.env.megadepth_csv,
                                                          estimate_uncertainty=estimate_uncertainty,
                                                          path_to_save=path_to_save, plot=args.plot,
                                                          plot_100=args.plot_100,
                                                          plot_ind_images=args.plot_individual_images)

        elif args.dataset == 'robotcar':
            output = run_evaluation_megadepth_or_robotcar(network, settings.env.robotcar,
                                                          path_to_csv=settings.env.robotcar_csv,
                                                          estimate_uncertainty=estimate_uncertainty,
                                                          path_to_save=path_to_save, plot=args.plot,
                                                          plot_100=args.plot_100,
                                                          plot_ind_images=args.plot_individual_images)

        elif 'hp' in args.dataset:
            original_size = True
            if args.dataset == 'hp-240':
                original_size = False
            number_of_scenes = 5 + 1
            list_of_outputs = []
            # loop over scenes (1-2, 1-3, 1-4, 1-5, 1-6)
            for id, k in enumerate(range(2, number_of_scenes + 2)):
                if id == 5:
                    _, test_set = datasets.HPatchesdataset(settings.env.hp,
                                                           os.path.join('assets',
                                                                        'hpatches_all.csv'.format(k)),
                                                           input_transform, target_transform, co_transform,
                                                           use_original_size=original_size, split=0)
                else:
                    _, test_set = datasets.HPatchesdataset(settings.env.hp,
                                                           os.path.join('assets',
                                                                        'hpatches_1_{}.csv'.format(k)),
                                                           input_transform, target_transform, co_transform,
                                                           use_original_size=original_size, split=0)
                test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
                output_scene = run_evaluation_generic(network, test_dataloader, device,
                                                      estimate_uncertainty=estimate_uncertainty)
                list_of_outputs.append(output_scene)

            output = {'scene_1': list_of_outputs[0], 'scene_2': list_of_outputs[1], 'scene_3': list_of_outputs[2],
                      'scene_4': list_of_outputs[3], 'scene_5': list_of_outputs[4], 'all': list_of_outputs[5]}

        elif args.dataset == 'kitti2012':
            _, test_set = datasets.KITTI_occ(settings.env.kitti2012, source_image_transform=input_transform,
                                             target_image_transform=input_transform,
                                             flow_transform=target_transform, co_transform=co_transform, split=0)
            test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
            output = run_evaluation_kitti(network, test_dataloader, device,
                                          estimate_uncertainty=estimate_uncertainty, path_to_save=path_to_save,
                                          plot=args.plot, plot_100=args.plot_100,
                                          plot_ind_images=args.plot_individual_images)

        elif args.dataset == 'kitti2015':
            _, test_set = datasets.KITTI_occ(settings.env.kitti2015, source_image_transform=input_transform,
                                             target_image_transform=input_transform,
                                             flow_transform=target_transform, co_transform=co_transform, split=0)
            test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)

            output = run_evaluation_kitti(network, test_dataloader, device,
                                          estimate_uncertainty=estimate_uncertainty, path_to_save=path_to_save,
                                          plot=args.plot, plot_100=args.plot_100,
                                          plot_ind_images=args.plot_individual_images)

        elif args.dataset == 'TSS':
            output = {}
            for sub_data in ['FG3DCar', 'JODS', 'PASCAL']:
                path_to_save_ = os.path.join(path_to_save, sub_data)
                if not os.path.exists(path_to_save_) and (args.plot or args.plot_100):
                    os.makedirs(path_to_save_)
                test_set = datasets.TSSDataset(os.path.join(settings.env.tss, sub_data),
                                               source_image_transform=input_transform,
                                               target_image_transform=input_transform, flow_transform=target_transform,
                                               co_transform=co_transform)
                test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
                results = run_evaluation_semantic(network, test_dataloader, device,
                                                  estimate_uncertainty=estimate_uncertainty,
                                                  flipping_condition=args.flipping_condition,
                                                  path_to_save=path_to_save_, plot=args.plot, plot_100=args.plot_100,
                                                  plot_ind_images=args.plot_individual_images)
                output[sub_data] = results

        elif args.dataset == 'PFPascal':
            test_set = datasets.PFPascalDataset(settings.env.PFPascal, source_image_transform=input_transform,
                                                target_image_transform=input_transform, split='test',
                                                flow_transform=target_transform)
            test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
            output = run_evaluation_semantic(network, test_dataloader, device, estimate_uncertainty=estimate_uncertainty,
                                             flipping_condition=args.flipping_condition,
                                             path_to_save=path_to_save, plot=args.plot, plot_100=args.plot_100,
                                             plot_ind_images=args.plot_individual_images)

        elif args.dataset == 'PFWillow':
            test_set = datasets.PFWillowDataset(settings.env.PFWillow, source_image_transform=input_transform,
                                                target_image_transform=input_transform, split='test',
                                                flow_transform=target_transform)
            test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
            output = run_evaluation_semantic(network, test_dataloader, device,
                                             estimate_uncertainty=estimate_uncertainty,
                                             flipping_condition=args.flipping_condition,
                                             path_to_save=path_to_save, plot=args.plot, plot_100=args.plot_100,
                                             plot_ind_images=args.plot_individual_images)
        elif args.dataset == 'spair':
            test_set = datasets.SPairDataset(settings.env.spair, source_image_transform=input_transform,
                                             target_image_transform=input_transform, split='test',
                                             flow_transform=target_transform)
            test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
            output = run_evaluation_semantic(network, test_dataloader, device,
                                             estimate_uncertainty=estimate_uncertainty,
                                             flipping_condition=args.flipping_condition,
                                             path_to_save=path_to_save, plot=args.plot, plot_100=args.plot_100,
                                             plot_ind_images=args.plot_individual_images)

        elif args.dataset == 'caltech':
            test_set = datasets.CaltechDataset(settings.env.caltech, source_image_transform=input_transform,
                                               target_image_transform=input_transform, split='test',
                                               flow_transform=target_transform)
            test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
            output = run_evaluation_caltech(network, test_dataloader, device, estimate_uncertainty=estimate_uncertainty,
                                            flipping_condition=args.flipping_condition, path_to_save=path_to_save,
                                            plot_ind_images=args.plot_individual_images)

        elif args.dataset == 'sintel':
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

        elif args.dataset == 'eth3d':
            output = run_evaluation_eth3d(network, settings.env.eth3d, input_transform, target_transform, co_transform,
                                          device, estimate_uncertainty=estimate_uncertainty)

        else:
            raise ValueError('Unknown dataset, {}'.format(args.dataset))

        save_dict['{}'.format(pre_trained_model_type)] = output

    if 'gocor' in args.model.lower() or 'PDCNet' in args.model:
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
    parser.add_argument('--dataset', type=str, help='Dataset name', required=True)
    define_model_parser(parser)  # model parameters
    parser.add_argument('--pre_trained_models', nargs='+', required=True,
                        help='name of pre trained models')

    parser.add_argument('--compute_metrics_uncertainty', default=False, type=boolean_string,
                        help='compute metrics uncertainty? default is False')
    parser.add_argument('--plot', default=False, type=boolean_string,
                        help='plot? default is False')
    parser.add_argument('--plot_100', default=False, type=boolean_string,
                        help='plot 100 first images? default is False')
    parser.add_argument('--plot_individual_images', default=False, type=boolean_string,
                        help='plot individual images? default is False')

    parser.add_argument('--save_dir', type=str,
                        help='path to directory to save the text files and results')
    parser.add_argument('--seed', type=int, default=1984, help='Pseudo-RNG seed')

    args = parser.parse_args()
    local_optim_iter = int(args.local_optim_iter) if args.local_optim_iter else args.optim_iter

    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # either gpu or cpu

    # settings containing paths to datasets
    settings = ws_settings.Settings()
    main(args=args, settings=settings)
