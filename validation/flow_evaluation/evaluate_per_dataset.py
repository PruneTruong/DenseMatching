import numpy as np
import torch
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader


from utils_flow.img_processing_utils import pad_to_same_shape
from validation.flow_evaluation.metrics_uncertainty import (compute_average_of_uncertainty_metrics, compute_aucs,
                                                            compute_uncertainty_per_image)
from datasets.geometric_matching_datasets.ETH3D_interval import ETHInterval
from validation.plot import plot_sparse_keypoints, plot_flow_and_uncertainty, plot_individual_images
from .metrics_segmentation_matching import poly_str_to_mask, intersection_over_union, label_transfer_accuracy
from utils_flow.pixel_wise_mapping import warp


def resize_images_to_min_resolution(min_size, img, x, y, stride_net=16):  # for consistency with RANSAC-Flow
    """
    Function that resizes the image according to the minsize, at the same time resize the x,y coordinate.
    Extracted from RANSAC-Flow (https://github.com/XiSHEN0220/RANSAC-Flow/blob/master/evaluation/evalCorr/getResults.py)
    We here use exactly the same function that they used, for fair comparison. Even through the index_valid could
    also theoretically include the lower bound x = 0 or y = 0.
    """
    # Is is source image resized
    # Xs contains the keypoint x coordinate in source image
    # Ys contains the keypoints y coordinate in source image
    # valids is bool on wheter the keypoint is contained in the source image
    x = np.array(list(map(float, x.split(';')))).astype(np.float32)  # contains all the x coordinate
    y = np.array(list(map(float, y.split(';')))).astype(np.float32)

    w, h = img.size
    ratio = min(w / float(min_size), h / float(min_size))
    new_w, new_h = round(w / ratio), round(h / ratio)
    new_w, new_h = new_w // stride_net * stride_net, new_h // stride_net * stride_net

    ratioW, ratioH = new_w / float(w), new_h / float(h)
    img = img.resize((new_w, new_h), resample=Image.LANCZOS)

    x, y = x * ratioW, y * ratioH  # put coordinate in proper size after resizing the images
    index_valid = (x > 0) * (x < new_w) * (y > 0) * (y < new_h)

    return img, x, y, index_valid


def compute_pck_sparse_data(x_s, y_s, x_r, y_r, flow, pck_thresholds, dict_list_uncertainties, uncertainty_est=None):

    flow_x = flow[0, 0].cpu().numpy()
    flow_y = flow[0, 1].cpu().numpy()

    # remove points for which xB, yB are outside of the image
    h, w = flow_x.shape
    index_valid = (np.int32(np.round(x_r)) >= 0) * (np.int32(np.round(x_r)) < w) * \
                  (np.int32(np.round(y_r)) >= 0) * (np.int32(np.round(y_r)) < h)
    x_s, y_s, x_r, y_r = x_s[index_valid], y_s[index_valid], x_r[index_valid], y_r[index_valid]
    nbr_valid_corr = index_valid.sum()

    # calculates the PCK
    if nbr_valid_corr > 0:
        # more accurate to compute the flow like this, instead of rounding both coordinates as in RANSAC-Flow
        flow_gt_x = x_s - x_r
        flow_gt_y = y_s - y_r
        flow_est_x = flow_x[np.int32(np.round(y_r)), np.int32(np.round(x_r))]
        flow_est_y = flow_y[np.int32(np.round(y_r)), np.int32(np.round(x_r))]
        EPE = ((flow_gt_x - flow_est_x) ** 2 + (flow_gt_y - flow_est_y) ** 2) ** 0.5
        EPE = EPE.reshape((-1, 1))
        AEPE = np.mean(EPE)
        count_pck = np.sum(EPE <= pck_thresholds, axis=0)
        # here compares the EPE of the pixels to be inferior to some value pixelGrid
    else:
        count_pck = np.zeros(pck_thresholds.shape[1])
        AEPE = np.nan

    results = {'count_pck': count_pck, 'nbr_valid_corr': nbr_valid_corr, 'aepe': AEPE}

    # calculates sparsification plot information
    if uncertainty_est is not None:
        flow_est = torch.from_numpy(np.concatenate((flow_est_x.reshape(-1, 1), flow_est_y.reshape(-1, 1)), axis=1))
        flow_gt = torch.from_numpy(np.concatenate((flow_gt_x.reshape(-1, 1), flow_gt_y.reshape(-1, 1)), axis=1))

        # uncert shape is #number_of_elements
        for uncertainty_name in uncertainty_est.keys():
            if uncertainty_name == 'inference_parameters' or uncertainty_name == 'log_var_map' or \
                    uncertainty_name == 'weight_map' or uncertainty_name == 'warping_mask':
                continue

            if 'p_r' == uncertainty_name:
                # convert confidence map to uncertainty
                uncert = (1.0 / (uncertainty_est['p_r'] + 1e-6)).squeeze()[np.int32(np.round(y_r)),
                                                                           np.int32(np.round(x_r))]
            else:
                uncert = uncertainty_est[uncertainty_name].squeeze()[np.int32(np.round(y_r)), np.int32(np.round(x_r))]
            # compute metrics based on uncertainty
            uncertainty_metric_dict = compute_aucs(flow_gt, flow_est, uncert, intervals=50)
            if uncertainty_name not in dict_list_uncertainties.keys():
                # for first image, create the list for each possible uncertainty type
                dict_list_uncertainties[uncertainty_name] = []
            dict_list_uncertainties[uncertainty_name].append(uncertainty_metric_dict)
    return results, dict_list_uncertainties


def run_evaluation_megadepth_or_robotcar(network, root, path_to_csv, estimate_uncertainty=False,
                                         min_size=480, stride_net=16, pre_processing=None,
                                         path_to_save=None, plot=False, plot_100=False,
                                         plot_ind_images=False):
    """
    Extracted from RANSAC-Flow (https://github.com/XiSHEN0220/RANSAC-Flow/blob/master/evaluation/evalCorr/getResults.py)
    We here recreate the same functions that they used, for fair comparison, but add additional metrics.
    """

    df = pd.read_csv(path_to_csv, dtype=str)
    nbImg = len(df)

    # pixelGrid = np.around(np.logspace(0, np.log10(36), 8).reshape(-1, 8))
    # looks at different distances for the keypoint
    pixelGrid = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 13.0, 22.0, 36.0]).reshape(-1, 9)
    # for metric calculation
    count_pck = np.zeros_like(pixelGrid)
    nbr_valid_corr = 0
    aepe_list = []
    dict_list_uncertainties = {}

    for i in tqdm(range(nbImg)):
        scene = df['scene'][i]

        # read and open the source and target image
        Is_original = Image.open(os.path.join(os.path.join(root, scene), df['source_image'][i])).convert('RGB') \
            if scene != '/' else Image.open(os.path.join(root, df['source_image'][i])).convert('RGB')
        It_original = Image.open(os.path.join(os.path.join(root, scene), df['target_image'][i])).convert('RGB') \
            if scene != '/' else Image.open(os.path.join(root, df['target_image'][i])).convert('RGB')

        # resize images and scale corresponding keypoints
        Is_original, Xs, Ys, valids = resize_images_to_min_resolution(min_size, Is_original, df['XA'][i],
                                                                      df['YA'][i], stride_net)
        It_original, Xt, Yt, validt = resize_images_to_min_resolution(min_size, It_original, df['XB'][i],
                                                                      df['YB'][i], stride_net)
        It_original = np.array(It_original)
        Is_original = np.array(Is_original)

        # removes points that are not contained in the source or the target
        index_valid = valids * validt
        Xs, Ys, Xt, Yt = Xs[index_valid], Ys[index_valid], Xt[index_valid], Yt[index_valid]

        # padd the images to the same shape to be fed to network + convert them to Tensors
        Is_original_padded_numpy, It_original_padded_numpy = pad_to_same_shape(Is_original, It_original)
        Is = torch.Tensor(Is_original_padded_numpy).permute(2, 0, 1).unsqueeze(0)
        It = torch.Tensor(It_original_padded_numpy).permute(2, 0, 1).unsqueeze(0)

        if pre_processing is not None:
            uncertainty_est = None
            flow_estimated = pre_processing.combine_with_est_flow_field(i, Is_original_padded_numpy,
                                                                        It_original_padded_numpy, Is, It, network)
        else:
            if estimate_uncertainty:
                flow_estimated, uncertainty_est = network.estimate_flow_and_confidence_map(Is, It)
            else:
                uncertainty_est = None
                flow_estimated = network.estimate_flow(Is, It)

        dict_results, dict_list_uncertainties = compute_pck_sparse_data(Xs, Ys, Xt, Yt, flow_estimated, pixelGrid,
                                                                        uncertainty_est=uncertainty_est,
                                                                        dict_list_uncertainties=dict_list_uncertainties)
        count_pck = count_pck + dict_results['count_pck']
        if dict_results['aepe'] != np.nan:
            aepe_list.append(dict_results['aepe'])
        nbr_valid_corr += dict_results['nbr_valid_corr']

        if plot_ind_images:
            plot_individual_images(path_to_save, 'image_{}'.format(i), Is, It, flow_estimated)
        if plot or (plot_100 and i < 100):
            plot_sparse_keypoints(path_to_save, 'image_{}'.format(i), Is, It, flow_estimated, Xs, Ys, Xt, Yt,
                                  uncertainty_comp_est=uncertainty_est)

    # Note that the PCK is over the whole dataset, for consistency with RANSAC-Flow computation.
    output = {'pixel-threshold': pixelGrid.tolist(), 'PCK': (count_pck / (nbr_valid_corr + 1e-6)).tolist(),
              'AEPE': np.mean(aepe_list).astype(np.float64)}
    print("Validation MegaDepth: {}".format(output['PCK']))
    if estimate_uncertainty:
        for uncertainty_name in dict_list_uncertainties.keys():
            output['uncertainty_dict_{}'.format(uncertainty_name)] = compute_average_of_uncertainty_metrics(
                dict_list_uncertainties[uncertainty_name])
    return output


def run_evaluation_kitti(network, test_dataloader, device, estimate_uncertainty=False,
                         path_to_save=None, plot=False, plot_100=False, plot_ind_images=False):
    out_list, epe_list = [], []
    dict_list_uncertainties = {}
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for i_batch, mini_batch in pbar:
        source_img = mini_batch['source_image']
        target_img = mini_batch['target_image']
        flow_gt = mini_batch['flow_map'].to(device)
        mask_valid = mini_batch['correspondence_mask'].to(device)

        if estimate_uncertainty:
            flow_est, uncertainty_est = network.estimate_flow_and_confidence_map(source_img, target_img)
        else:
            flow_est = network.estimate_flow(source_img, target_img)

        if plot or (plot_100 and i_batch < 100):
            plot_flow_and_uncertainty(path_to_save, 'image_{}'.format(i_batch), source_img, target_img,
                                      flow_gt, flow_est, compute_rgb_flow=True)
        if plot_ind_images:
            plot_individual_images(path_to_save, 'image_{}'.format(i_batch), source_img, target_img, flow_est)

        flow_est = flow_est.permute(0, 2, 3, 1)[mask_valid]
        flow_gt = flow_gt.permute(0, 2, 3, 1)[mask_valid]

        epe = torch.sum((flow_est - flow_gt) ** 2, dim=1).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe.mean().item())
        out_list.append(out.cpu().numpy())

        if estimate_uncertainty:
            dict_list_uncertainties = compute_uncertainty_per_image(uncertainty_est, flow_gt, flow_est, mask_valid,
                                                                    dict_list_uncertainties)

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)  # AEPE is per image, and then averaged over the dataset.
    fl = 100 * np.mean(out_list)  # fl is over the whole dataset
    print("Validation KITTI: aepe: %f, fl: %f" % (epe, fl))
    output = {'AEPE': epe, 'kitti-fl': fl}
    if estimate_uncertainty:
        for uncertainty_name in dict_list_uncertainties.keys():
            output['uncertainty_dict_{}'.format(uncertainty_name)] = compute_average_of_uncertainty_metrics(
                dict_list_uncertainties[uncertainty_name])
    return output


def run_evaluation_sintel(network, test_dataloader, device, estimate_uncertainty=False):
    epe_list, pck_1_list, pck_3_list, pck_5_list = [], [], [], []
    dict_list_uncertainties = {}
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for i_batch, mini_batch in pbar:
        source_img = mini_batch['source_image']
        target_img = mini_batch['target_image']
        flow_gt = mini_batch['flow_map'].to(device)
        mask_valid = mini_batch['correspondence_mask'].to(device)

        if estimate_uncertainty:
            flow_est, uncertainty_est = network.estimate_flow_and_confidence_map(source_img, target_img)
        else:
            flow_est = network.estimate_flow(source_img, target_img)

        flow_est = flow_est.permute(0, 2, 3, 1)[mask_valid]
        flow_gt = flow_gt.permute(0, 2, 3, 1)[mask_valid]

        epe = torch.sum((flow_est - flow_gt) ** 2, dim=1).sqrt()
        epe_list.append(epe.view(-1).cpu().numpy())
        pck_1_list.append(epe.le(1.0).float().mean().item())
        pck_3_list.append(epe.le(3.0).float().mean().item())
        pck_5_list.append(epe.le(5.0).float().mean().item())

        if estimate_uncertainty:
            dict_list_uncertainties = compute_uncertainty_per_image(uncertainty_est, flow_gt, flow_est, mask_valid,
                                                                    dict_list_uncertainties)

    epe_all = np.concatenate(epe_list).astype(np.float64)
    epe = np.mean(epe_all)
    pck1 = np.mean(epe_all <= 1)
    pck3 = np.mean(epe_all <= 3)
    pck5 = np.mean(epe_all <= 5)

    output = {'AEPE': epe, 'PCK_1': pck1, 'PCK_3': pck3, 'PCK5': pck5,
              'PCK_1_per_image': np.mean(pck_1_list),
              'PCK_3_per_image': np.mean(pck_3_list), 'PCK_5_per_image': np.mean(pck_5_list),
              }
    print("Validation EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (epe, pck1, pck3, pck5))
    if estimate_uncertainty:
        for uncertainty_name in dict_list_uncertainties.keys():
            output['uncertainty_dict_{}'.format(uncertainty_name)] = compute_average_of_uncertainty_metrics(
                dict_list_uncertainties[uncertainty_name])
    return output


def run_evaluation_generic(network, test_dataloader, device, estimate_uncertainty=False):
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    mean_epe_list, epe_all_list, pck_1_list, pck_3_list, pck_5_list = [], [], [], [], []
    dict_list_uncertainties = {}
    for i_batch, mini_batch in pbar:
        source_img = mini_batch['source_image']
        target_img = mini_batch['target_image']
        flow_gt = mini_batch['flow_map'].to(device)
        mask_valid = mini_batch['correspondence_mask'].to(device)

        if estimate_uncertainty:
            flow_est, uncertainty_est = network.estimate_flow_and_confidence_map(source_img, target_img)
        else:
            flow_est = network.estimate_flow(source_img, target_img)

        flow_est = flow_est.permute(0, 2, 3, 1)[mask_valid]
        flow_gt = flow_gt.permute(0, 2, 3, 1)[mask_valid]

        epe = torch.sum((flow_est - flow_gt) ** 2, dim=1).sqrt()

        epe_all_list.append(epe.view(-1).cpu().numpy())
        mean_epe_list.append(epe.mean().item())
        pck_1_list.append(epe.le(1.0).float().mean().item())
        pck_3_list.append(epe.le(3.0).float().mean().item())
        pck_5_list.append(epe.le(5.0).float().mean().item())

        if estimate_uncertainty:
            dict_list_uncertainties = compute_uncertainty_per_image(uncertainty_est, flow_gt, flow_est, mask_valid,
                                                                    dict_list_uncertainties)

    epe_all = np.concatenate(epe_all_list)
    pck1_dataset = np.mean(epe_all <= 1)
    pck3_dataset = np.mean(epe_all <= 3)
    pck5_dataset = np.mean(epe_all <= 5)
    output = {'AEPE': np.mean(mean_epe_list), 'PCK_1_per_image': np.mean(pck_1_list),
              'PCK_3_per_image': np.mean(pck_3_list), 'PCK_5_per_image': np.mean(pck_5_list),
              'PCK_1_per_dataset': pck1_dataset, 'PCK_3_per_dataset': pck3_dataset,
              'PCK_5_per_dataset': pck5_dataset, 'num_pixels_pck_1': np.sum(epe_all <= 1).astype(np.float64),
              'num_pixels_pck_3': np.sum(epe_all <= 3).astype(np.float64),
              'num_pixels_pck_5': np.sum(epe_all <= 5).astype(np.float64),
              'num_valid_corr': len(epe_all)
              }
    print("Validation EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (np.mean(mean_epe_list), pck1_dataset,
                                                             pck3_dataset, pck5_dataset))
    if estimate_uncertainty:
        for uncertainty_name in dict_list_uncertainties.keys():
            output['uncertainty_dict_{}'.format(uncertainty_name)] = compute_average_of_uncertainty_metrics(
                dict_list_uncertainties[uncertainty_name])
    return output


def run_evaluation_eth3d(network, data_dir, input_images_transform, gt_flow_transform, co_transform, device,
                         estimate_uncertainty):
    # ETH3D dataset information
    dataset_names = ['lakeside', 'sand_box', 'storage_room', 'storage_room_2', 'tunnel', 'delivery_area', 'electro',
                     'forest', 'playground', 'terrains']
    rates = list(range(3, 16, 2))
    dict_results = {}
    for rate in rates:
        print('Computing results for interval {}...'.format(rate))
        dict_results['rate_{}'.format(rate)] = {}
        list_of_outputs_per_rate = []
        num_pck_1 = 0.0
        num_pck_3 = 0.0
        num_pck_5 = 0.0
        num_valid_correspondences = 0.0
        for name_dataset in dataset_names:
            print('looking at dataset {}...'.format(name_dataset))
            test_set = ETHInterval(root=data_dir,
                                   path_list=os.path.join(data_dir, 'info_ETH3D_files',
                                                          '{}_every_5_rate_of_{}'.format(name_dataset, rate)),
                                   source_image_transform=input_images_transform,
                                   target_image_transform=input_images_transform,
                                   flow_transform=gt_flow_transform,
                                   co_transform=co_transform)  # only test
            test_dataloader = DataLoader(test_set,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=8)
            print(test_set.__len__())
            output = run_evaluation_generic(network, test_dataloader, device, estimate_uncertainty)
            # to save the intermediate results
            # dict_results['rate_{}'.format(rate)][name_dataset] = output
            list_of_outputs_per_rate.append(output)
            num_pck_1 += output['num_pixels_pck_1']
            num_pck_3 += output['num_pixels_pck_3']
            num_pck_5 += output['num_pixels_pck_5']
            num_valid_correspondences += output['num_valid_corr']

        # average over all datasets for this particular rate of interval
        avg = {'AEPE': np.mean([list_of_outputs_per_rate[i]['AEPE'] for i in range(len(dataset_names))]),
               'PCK_1_per_image': np.mean([list_of_outputs_per_rate[i]['PCK_1_per_image'] for i in
                                           range(len(dataset_names))]),
               'PCK_3_per_image': np.mean([list_of_outputs_per_rate[i]['PCK_3_per_image'] for i in
                                           range(len(dataset_names))]),
               'PCK_5_per_image': np.mean([list_of_outputs_per_rate[i]['PCK_5_per_image'] for i in
                                           range(len(dataset_names))]),
               'pck-1-per-rate': num_pck_1 / (num_valid_correspondences + 1e-6),
               'pck-3-per-rate': num_pck_3 / (num_valid_correspondences + 1e-6),
               'pck-5-per-rate': num_pck_5 / (num_valid_correspondences + 1e-6),
               'num_valid_corr': num_valid_correspondences
               }
        dict_results['rate_{}'.format(rate)] = avg

    avg_rates = {'AEPE': np.mean([dict_results['rate_{}'.format(rate)]['AEPE'] for rate in rates]),
                 'PCK_1_per_image': np.mean(
                     [dict_results['rate_{}'.format(rate)]['PCK_1_per_image'] for rate in rates]),
                 'PCK_3_per_image': np.mean(
                     [dict_results['rate_{}'.format(rate)]['PCK_3_per_image'] for rate in rates]),
                 'PCK_5_per_image': np.mean(
                     [dict_results['rate_{}'.format(rate)]['PCK_5_per_image'] for rate in rates]),
                 'pck-1-per-rate': np.mean([dict_results['rate_{}'.format(rate)]['pck-1-per-rate'] for rate in rates]),
                 'pck-3-per-rate': np.mean([dict_results['rate_{}'.format(rate)]['pck-3-per-rate'] for rate in rates]),
                 'pck-5-per-rate': np.mean([dict_results['rate_{}'.format(rate)]['pck-5-per-rate'] for rate in rates]),
                 }
    dict_results['avg'] = avg_rates
    return dict_results


def run_evaluation_semantic(network, test_dataloader, device, estimate_uncertainty=False, flipping_condition=False,
                            path_to_save=None, plot=False, plot_100=False, plot_ind_images=False):
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    mean_epe_list, epe_all_list, pck_0_05_list, pck_0_01_list, pck_0_1_list, pck_0_15_list = [], [], [], [], [], []
    dict_list_uncertainties = {}
    eval_buf = {'cls_pck': dict(), 'vpvar': dict(), 'scvar': dict(), 'trncn': dict(), 'occln': dict()}

    # pck curve per image
    pck_thresholds = [0.01]
    pck_thresholds.extend(np.arange(0.05, 0.4, 0.05).tolist())
    pck_per_image_curve = np.zeros((len(pck_thresholds), len(test_dataloader)), np.float32)

    for i_batch, mini_batch in pbar:
        source_img = mini_batch['source_image']
        target_img = mini_batch['target_image']
        flow_gt = mini_batch['flow_map'].to(device)
        mask_valid = mini_batch['correspondence_mask'].to(device)

        if 'pckthres' in list(mini_batch.keys()):
            L_pck = mini_batch['pckthres'][0].float().item()
        else:
            raise ValueError('No pck threshold in mini_batch')

        if estimate_uncertainty:
            if flipping_condition:
                raise NotImplementedError('No flipping condition for PDC-Net yet')
            flow_est, uncertainty_est = network.estimate_flow_and_confidence_map(source_img, target_img)
        else:
            uncertainty_est = None
            if flipping_condition:
                flow_est = network.estimate_flow_with_flipping_condition(source_img, target_img)
            else:
                flow_est = network.estimate_flow(source_img, target_img)
        if plot_ind_images:
            plot_individual_images(path_to_save, 'image_{}'.format(i_batch), source_img, target_img, flow_est)

        if plot or (plot_100 and i_batch < 100):
            if 'source_kps' in list(mini_batch.keys()):
                # I = estimate_probability_of_confidence_interval_of_mixture_density(log_var_map_padded, R=1.0)
                plot_sparse_keypoints(path_to_save, 'image_{}'.format(i_batch), source_img, target_img, flow_est,
                                      mini_batch['source_kps'][0][:, 0], mini_batch['source_kps'][0][:, 1],
                                      mini_batch['target_kps'][0][:, 0], mini_batch['target_kps'][0][:, 1],
                                      uncertainty_comp_est=uncertainty_est)
            else:
                plot_flow_and_uncertainty(path_to_save, 'image_{}'.format(i_batch), source_img, target_img,
                                          flow_gt, flow_est)

        flow_est = flow_est.permute(0, 2, 3, 1)[mask_valid]
        flow_gt = flow_gt.permute(0, 2, 3, 1)[mask_valid]

        epe = torch.sum((flow_est - flow_gt) ** 2, dim=1).sqrt()

        epe_all_list.append(epe.view(-1).cpu().numpy())
        mean_epe_list.append(epe.mean().item())
        pck_0_05_list.append(epe.le(0.05*L_pck).float().mean().item())
        pck_0_01_list.append(epe.le(0.01*L_pck).float().mean().item())
        pck_0_1_list.append(epe.le(0.1*L_pck).float().mean().item())
        pck_0_15_list.append(epe.le(0.15*L_pck).float().mean().item())
        for t in range(len(pck_thresholds)):
            pck_per_image_curve[t, i_batch] = epe.le(pck_thresholds[t]*L_pck).float().mean().item()

        if 'category' in mini_batch.keys():
            if eval_buf['cls_pck'].get(mini_batch['category'][0]) is None:
                eval_buf['cls_pck'][mini_batch['category'][0]] = []
            eval_buf['cls_pck'][mini_batch['category'][0]].append(epe.le(0.1 * L_pck).float().mean().item())

        if 'vpvar' in mini_batch.keys():
            for name in ['vpvar', 'scvar', 'trncn', 'occln']:
                # different difficulties
                # means it is spair
                if eval_buf[name].get('{}'.format(mini_batch[name][0])) is None:
                    eval_buf[name]['{}'.format(mini_batch[name][0])] = []
                eval_buf[name]['{}'.format(mini_batch[name][0])].append(epe.le(0.1 * L_pck).float().mean().item())

        if estimate_uncertainty:
            dict_list_uncertainties = compute_uncertainty_per_image(uncertainty_est, flow_gt, flow_est, mask_valid,
                                                                    dict_list_uncertainties)

    epe_all = np.concatenate(epe_all_list)
    pck_0_05_dataset = np.mean(epe_all <= 0.05 * L_pck)
    pck_0_01_dataset = np.mean(epe_all <= 0.01 * L_pck)
    pck_0_1_dataset = np.mean(epe_all <= 0.1 * L_pck)
    pck_0_15_dataset = np.mean(epe_all <= 0.15 * L_pck)

    output = {'AEPE': np.mean(mean_epe_list), 'PCK_0_05_per_image': np.mean(pck_0_05_list),
              'PCK_0_01_per_image': np.mean(pck_0_01_list), 'PCK_0_1_per_image': np.mean(pck_0_1_list),
              'PCK_0_15_per_image': np.mean(pck_0_15_list),
              'PCK_0_01_per_dataset': pck_0_01_dataset, 'PCK_0_05_per_dataset': pck_0_05_dataset,
              'PCK_0_1_per_dataset': pck_0_1_dataset, 'PCK_0_15_per_dataset': pck_0_15_dataset,
              'pck_threshold_alpha': pck_thresholds, 'pck_curve_per_image': np.mean(pck_per_image_curve, axis=1).tolist()
              }
    print("Validation EPE: %f, alpha=0_01: %f, alpha=0.05: %f" % (output['AEPE'], output['PCK_0_01_per_image'],
                                                                  output['PCK_0_05_per_image']))

    for name in eval_buf.keys():
        output[name] = {}
        for cls in eval_buf[name]:
            if eval_buf[name] is not None:
                cls_avg = sum(eval_buf[name][cls]) / len(eval_buf[name][cls])
                output[name][cls] = cls_avg

    if estimate_uncertainty:
        for uncertainty_name in dict_list_uncertainties.keys():
            output['uncertainty_dict_{}'.format(uncertainty_name)] = compute_average_of_uncertainty_metrics(
                dict_list_uncertainties[uncertainty_name])
    return output


def run_evaluation_caltech(network, test_dataloader, device, estimate_uncertainty=False, flipping_condition=False,
                           path_to_save=None, plot_ind_images=False,):

    def compute_mean(results):
        good_idx = np.flatnonzero((results != -1) * ~np.isnan(results))
        filtered_results = np.float64(results)[good_idx]
        return np.mean(filtered_results)

    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    list_intersection_over_union, list_label_transfer_accuracy, list_localization_error = [], [], []

    for i_batch, mini_batch in pbar:
        mini_batch['nbr'] = i_batch
        source_img = mini_batch['source_image']
        target_img = mini_batch['target_image']
        h_src, w_src = mini_batch['source_image_size'][0]
        h_tgt, w_tgt = mini_batch['source_image_size'][0]

        target_mask_np, target_mask = poly_str_to_mask(
            mini_batch['target_kps'][0, :mini_batch['n_pts'][0], 0],
            mini_batch['target_kps'][0, :mini_batch['n_pts'][0], 1], h_tgt, w_tgt)

        source_mask_np, source_mask = poly_str_to_mask(
            mini_batch['source_kps'][0, :mini_batch['n_pts'][0], 0],
            mini_batch['source_kps'][0, :mini_batch['n_pts'][0], 1], h_src, w_src)

        if estimate_uncertainty:
            flow_est, uncertainty_est = network.estimate_flow_and_confidence_map(source_img, target_img)
        else:
            if flipping_condition:
                flow_est = network.estimate_flow_with_flipping_condition(source_img, target_img)
            else:
                flow_est = network.estimate_flow(source_img, target_img)

        flow_est = flow_est[:, :, :h_tgt, :w_tgt]  # remove the padding, to original images.

        warped_mask_1 = warp(source_mask, flow_est)

        list_intersection_over_union.append(intersection_over_union(warped_mask_1, target_mask).item())
        list_label_transfer_accuracy.append(label_transfer_accuracy(warped_mask_1, target_mask).item())

        if plot_ind_images:
            mask = None
            plot_individual_images(path_to_save, 'image_{}'.format(i_batch), source_img, target_img, flow_est, mask)

    output = {'intersection_over_union': compute_mean(list_intersection_over_union),
              'label_transfer_accuracy': compute_mean(list_label_transfer_accuracy)
              }
    print("Validation IoU: %f, transfer Acc: %f" % (output['intersection_over_union'],
                                                    output['label_transfer_accuracy']))
    return output
