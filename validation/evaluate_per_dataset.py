import numpy as np
import torch
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from utils_flow.util import pad_to_same_shape
from validation.metrics_uncertainty import compute_aucs, compute_average_of_uncertainty_metrics
from datasets.geometric_matching_datasets.ETH3D_interval import ETHInterval
from .plot import plot_sparse_keypoints, plot_flow_and_uncertainty


def resize_images_to_min_resolution(minSize, I, x, y, strideNet):
    # resize image according to the minsize, at the same time resize the x,y coordinate
    # Is is source image resized
    # Xs contains the keypoint x coordinate in source image
    # Ys contains the keypoints y coordinate in source image
    # valids is bool on wheter the keypoint is contained in the source image
    x = np.array(list(map(float, x.split(';')))).astype(np.float32)  # contains all the x coordinate
    y = np.array(list(map(float, y.split(';')))).astype(np.float32)

    w, h = I.size
    ratio = min(w / float(minSize), h / float(minSize))
    new_w, new_h = round(w / ratio), round(h / ratio)
    new_w, new_h = new_w // strideNet * strideNet, new_h // strideNet * strideNet

    ratioW, ratioH = new_w / float(w), new_h / float(h)
    I = I.resize((new_w, new_h), resample=Image.LANCZOS)

    x, y = x * ratioW, y * ratioH  # put coordinate in proper size after resizing the images
    index_valid = (x > 0) * (x < new_w) * (y > 0) * (y < new_h)

    return I, x, y, index_valid


def compute_pck_sparse_data(x_s, y_s, x_r, y_r, flow, pck_thresholds, uncertainty_est=None):

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
        flow_gt_x = x_s - x_r
        flow_gt_y = y_s - y_r
        flow_est_x = flow_x[np.int32(np.round(y_r)), np.int32(np.round(x_r))]
        flow_est_y = flow_y[np.int32(np.round(y_r)), np.int32(np.round(x_r))]
        EPE = ((flow_gt_x - flow_est_x) ** 2 + (flow_gt_y - flow_est_y) ** 2)**0.5
        EPE = EPE.reshape((-1, 1))
        count_pck = np.sum(EPE <= pck_thresholds, axis = 0)
        # here compares the EPE of the pixels to be inferior to some value pixelGrid
    else:
        count_pck = np.zeros(pck_thresholds.shape[1])

    results = {'count_pck': count_pck, 'nbr_valid_corr': nbr_valid_corr}

    # calculates sparsification plot information
    if uncertainty_est is not None:
        flow_est = torch.from_numpy(np.concatenate((flow_est_x.reshape(-1, 1), flow_est_y.reshape(-1, 1)), axis=1))
        flow_gt = torch.from_numpy(np.concatenate((flow_gt_x.reshape(-1, 1), flow_gt_y.reshape(-1, 1)), axis=1))

        # compute metrics based on uncertainty
        uncert_p_r = (1.0 / (uncertainty_est['p_r'] + 1e-6)).squeeze()[np.int32(np.round(y_r)), np.int32(np.round(x_r))]
        # uncert shape is #number_of_elements
        uncert_variance = uncertainty_est['variance'].squeeze()[np.int32(np.round(y_r)), np.int32(np.round(x_r))]

        results['uncertainty_dict_variance'] = compute_aucs(flow_gt, flow_est, uncert_variance, intervals=50)
        results['uncertainty_dict_p_r'] = compute_aucs(flow_gt, flow_est, uncert_p_r, intervals=50)
    return results


def run_evaluation_megadepth_or_robotcar(network, testDir, testCSV, estimate_uncertainty=False,
                                         min_size=480, stride_net=8, path_to_save=None, plot=False):

    df = pd.read_csv(testCSV, dtype=str)
    nbImg = len(df)

    # pixelGrid = np.around(np.logspace(0, np.log10(36), 8).reshape(-1, 8))
    # looks at different distances for the keypoint
    pixelGrid = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 13.0, 22.0, 36.0]).reshape(-1, 9)
    # for metric calculation
    count_pck = np.zeros_like(pixelGrid)
    nbr_valid_corr = 0
    list_uncertainty_metrics_variance, list_uncertainty_metrics_p_r = [], []

    for i in tqdm(range(nbImg)):
        scene = df['scene'][i]

        # read and open the source and target image
        Is_original = Image.open( os.path.join(os.path.join(testDir, scene), df['source_image'][i])).convert('RGB') \
            if scene != '/' else Image.open(os.path.join(testDir, df['source_image'][i])).convert('RGB')
        It_original = Image.open(os.path.join(os.path.join(testDir, scene), df['target_image'][i])).convert('RGB') \
            if scene != '/' else Image.open(os.path.join(testDir, df['target_image'][i])).convert('RGB')

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

        if estimate_uncertainty:
            flow_estimated, uncertainty_est = network.estimate_flow_and_confidence_map(Is, It)
        else:
            uncertainty_est = None
            flow_estimated = network.estimate_flow(Is, It)

        dict_results = compute_pck_sparse_data(Xs, Ys, Xt, Yt, flow_estimated, pixelGrid, uncertainty_est)
        count_pck = count_pck + dict_results['count_pck']
        nbr_valid_corr += dict_results['nbr_valid_corr']
        if 'uncertainty_dict_variance' in list(dict_results.keys()):
            list_uncertainty_metrics_variance.append(dict_results['uncertainty_dict_variance'])
        if 'uncertainty_dict_p_r' in list(dict_results.keys()):
            list_uncertainty_metrics_p_r.append(dict_results['uncertainty_dict_p_r'])

        if plot:
            # I = estimate_probability_of_confidence_interval_of_mixture_density(log_var_map_padded, R=1.0)
            plot_sparse_keypoints(path_to_save, 'image_{}'.format(i), Is, It, flow_estimated, Xs, Ys, Xt, Yt,
                                  uncertainty_comp_est=uncertainty_est)

    output = {'pixel-threshold': pixelGrid.tolist(), 'PCK': (count_pck / (nbr_valid_corr + 1e-6)).tolist()}
    print("Validation MegaDepth: {}".format(output['PCK']))
    if estimate_uncertainty:
        output['uncertainty_dict_variance'] = compute_average_of_uncertainty_metrics(
            list_uncertainty_metrics_variance)
        output['uncertainty_dict_p_r'] = compute_average_of_uncertainty_metrics(list_uncertainty_metrics_p_r)
    return output

    
def run_evaluation_kitti(network, test_dataloader, device, estimate_uncertainty=False,
                         path_to_save=None, plot=False):
    out_list, epe_list, list_uncertainty_metrics_variance, list_uncertainty_metrics_p_r = [], [], [], []
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

        if plot:
            plot_flow_and_uncertainty(path_to_save, 'image_{}'.format(i_batch), source_img, target_img,
                                      flow_gt, flow_est, compute_rgb_flow=True)

        flow_est = flow_est.permute(0, 2, 3, 1)[mask_valid]
        flow_gt = flow_gt.permute(0, 2, 3, 1)[mask_valid]

        epe = torch.sum((flow_est - flow_gt) ** 2, dim=1).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe.mean().item())
        out_list.append(out.cpu().numpy())

        if estimate_uncertainty:
            # compute metrics based on uncertainty
            uncert_p_r = (1.0 / (uncertainty_est['p_r'] + 1e-6))[mask_valid.unsqueeze(1)]
            # uncert shape is #number_of_elements
            uncert_variance = uncertainty_est['variance'][mask_valid.unsqueeze(1)]

            uncertainty_metric_dict_variance = compute_aucs(flow_gt, flow_est, uncert_variance,
                                                            intervals=50)
            list_uncertainty_metrics_variance.append(uncertainty_metric_dict_variance)
            uncertainty_metric_dict_p_r = compute_aucs(flow_gt, flow_est, uncert_p_r, intervals=50)
            list_uncertainty_metrics_p_r.append(uncertainty_metric_dict_p_r)

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)
    print("Validation KITTI: %f, %f" % (epe, f1))
    output = {'AEPE': epe, 'kitti-f1': f1}
    if estimate_uncertainty:
        output['uncertainty_dict_variance'] = compute_average_of_uncertainty_metrics(list_uncertainty_metrics_variance)
        output['uncertainty_dict_p_r'] = compute_average_of_uncertainty_metrics(list_uncertainty_metrics_p_r)
    return output


def run_evaluation_sintel(network, test_dataloader, device, estimate_uncertainty=False):
    epe_list, uncertainty_metrics, list_uncertainty_metrics_variance, list_uncertainty_metrics_p_r = [], [], [], []
    pck_1_list, pck_3_list, pck_5_list = [], [], []
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
            # compute metrics based on uncertainty
            uncert_p_r = (1.0 / (uncertainty_est['p_r'] + 1e-6))[mask_valid.unsqueeze(1)]
            # uncert shape is #number_of_elements
            uncert_variance = uncertainty_est['variance'][mask_valid.unsqueeze(1)]

            uncertainty_metric_dict_variance = compute_aucs(flow_gt, flow_est, uncert_variance,
                                                            intervals=50)
            list_uncertainty_metrics_variance.append(uncertainty_metric_dict_variance)
            uncertainty_metric_dict_p_r = compute_aucs(flow_gt, flow_est, uncert_p_r, intervals=50)
            list_uncertainty_metrics_p_r.append(uncertainty_metric_dict_p_r)

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
        output['uncertainty_dict_variance'] = compute_average_of_uncertainty_metrics(list_uncertainty_metrics_variance)
        output['uncertainty_dict_p_r'] = compute_average_of_uncertainty_metrics(list_uncertainty_metrics_p_r)
    return output


def run_evaluation_generic(network, test_dataloader, device, estimate_uncertainty=False):
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    mean_epe_list, epe_all_list, pck_1_list, pck_3_list, pck_5_list, list_uncertainty_metrics_variance, \
    list_uncertainty_metrics_p_r = [], [], [], [], [], [], []
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
            # compute metrics based on uncertainty
            uncert_p_r = (1.0 / (uncertainty_est['p_r'] + 1e-6))[mask_valid.unsqueeze(1)]
            # uncert shape is #number_of_elements
            uncert_variance = uncertainty_est['variance'][mask_valid.unsqueeze(1)]

            uncertainty_metric_dict_variance = compute_aucs(flow_gt, flow_est, uncert_variance,
                                                            intervals=50)
            list_uncertainty_metrics_variance.append(uncertainty_metric_dict_variance)
            uncertainty_metric_dict_p_r = compute_aucs(flow_gt, flow_est, uncert_p_r, intervals=50)
            list_uncertainty_metrics_p_r.append(uncertainty_metric_dict_p_r)

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
    print("Validation EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (np.mean(mean_epe_list), pck1_dataset, pck3_dataset, pck5_dataset))
    if estimate_uncertainty:
        output['uncertainty_dict_variance'] = compute_average_of_uncertainty_metrics(list_uncertainty_metrics_variance)
        output['uncertainty_dict_p_r'] = compute_average_of_uncertainty_metrics(list_uncertainty_metrics_p_r)
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


def run_evaluation_semantic(network, test_dataloader, device, estimate_uncertainty=False, flipping_condition=True,
                            path_to_save=None, plot=False):
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    mean_epe_list, epe_all_list, pck_0_05_list, pck_0_01_list, pck_0_1_list, pck_0_15_list,\
    list_uncertainty_metrics_variance, list_uncertainty_metrics_p_r = [], [], [], [], [], [], [], []

    # pck curve per image
    pck_thresholds = [0.01]
    pck_thresholds.extend(np.arange(0.05, 0.4, 0.05).tolist())

    pck_per_image_curve = np.zeros((len(pck_thresholds), len(test_dataloader)), np.float32)
    for i_batch, mini_batch in pbar:
        source_img = mini_batch['source_image']
        target_img = mini_batch['target_image']
        flow_gt = mini_batch['flow_map'].to(device)
        mask_valid = mini_batch['correspondence_mask'].to(device)
        if 'L_bounding_box' in list(mini_batch.keys()):
            L_pck = mini_batch['L_bounding_box'][0].float().item()
        else:
            L_pck = max(mini_batch['source_image_size'][0], mini_batch['source_image_size'][1]).float().item()

        if estimate_uncertainty:
            # not evaluated
            flow_est, uncertainty_est = network.estimate_flow_and_confidence_map(source_img, target_img)
        else:
            uncertainty_est = None
            if flipping_condition:
                flow_est = network.estimate_flow_with_flipping_condition(source_img, target_img)
            else:
                flow_est = network.estimate_flow(source_img, target_img)

        if plot:
            if 'source_coor' in list(mini_batch.keys()):
                # I = estimate_probability_of_confidence_interval_of_mixture_density(log_var_map_padded, R=1.0)
                plot_sparse_keypoints(path_to_save, 'image_{}'.format(i_batch), source_img, target_img, flow_est,
                                      mini_batch['source_coor'][0][:, 0], mini_batch['source_coor'][0][:, 1],
                                      mini_batch['target_coor'][0][:, 0], mini_batch['target_coor'][0][:, 1],
                                      uncertainty_comp_est=uncertainty_est)
            else:
                plot_flow_and_uncertainty(path_to_save, 'image_{}'.format(i_batch), source_img, target_img,
                                          flow_gt, flow_est, uncertainty_comp_est=uncertainty_est)

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

        if estimate_uncertainty:
            # compute metrics based on uncertainty
            uncert_p_r = (1.0 / (uncertainty_est['p_r'] + 1e-6))[mask_valid.unsqueeze(1)]
            # uncert shape is #number_of_elements
            uncert_variance = uncertainty_est['variance'][mask_valid.unsqueeze(1)]

            uncertainty_metric_dict_variance = compute_aucs(flow_gt, flow_est, uncert_variance,
                                                            intervals=50)
            list_uncertainty_metrics_variance.append(uncertainty_metric_dict_variance)
            uncertainty_metric_dict_p_r = compute_aucs(flow_gt, flow_est, uncert_p_r, intervals=50)
            list_uncertainty_metrics_p_r.append(uncertainty_metric_dict_p_r)

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
    if estimate_uncertainty:
        output['uncertainty_dict_variance'] = compute_average_of_uncertainty_metrics(list_uncertainty_metrics_variance)
        output['uncertainty_dict_p_r'] = compute_average_of_uncertainty_metrics(list_uncertainty_metrics_p_r)
    return output