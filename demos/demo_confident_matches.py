"""
This demo is heavily inspired by [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork/).
"""
from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import os
import sys
import numpy as np
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)
from model_selection import select_model
from utils_flow.visualization_utils import overlay_semantic_mask
from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from demos.utils import (AverageTimer, VideoStreamer, make_matching_and_warping_plot_fast, make_matching_plot_fast)
from validation.test_parser import define_model_parser
from utils_flow.visualization_utils import make_and_save_video
torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Matching demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    define_model_parser(parser)  # model parameters
    parser.add_argument('--pre_trained_model', type=str, required=True,
                        help='name of the pre trained model')
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--save_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--visualize_warp', action='store_true',
        help='visualize the warp, as well as the matches.')
    parser.add_argument(
        '--mask_uncertain_regions', action='store_true',
        help='When warping the source to the target image, we zeros out the regions for which the matches are '
             'not predicted as confident.')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--save_video', action='store_true',
        help='Save output (with match visualizations) to a video.')
    parser.add_argument(
        '--save_input', action='store_true',
        help='Save the input images to a video (for gathering repeatable input source).')

    # selection of confident match points
    parser.add_argument('--confident_mask_type', default='proba_interval_1_above_10', type=str,
                        help='mask type for filtering confident matches (when confidence is between 0 and 1)')
    parser.add_argument(
        '--cyclic_consistency_mask_threshold', type=int, default=2,
        help='Threshold used to filter matches based on their cyclic consistency error (in pixels)'
        ' (Must be positive)')

    opt = parser.parse_args()
    print(opt)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    # Configure I/O
    if opt.save_video:
        print('Writing video to {}-matches.mp4...'.format(opt.model))
        writer = cv2.VideoWriter('{}-matches.mp4'.format(opt.model), cv2.VideoWriter_fourcc(*'mp4v'), 15,
                                 (640*2 + 10, 480))
    if opt.save_input:
        print('Writing video to demo-input.mp4...')
        input_writer = cv2.VideoWriter('demo-input.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))

    network, estimate_uncertainty = select_model(
        opt.model, opt.pre_trained_model, opt, opt.optim_iter, opt.local_optim_iter,
        path_to_pre_trained_models=opt.path_to_pre_trained_models)

    confident_mask_type = opt.confident_mask_type
    # name of the model + pre trained model
    name_to_save = opt.model
    if 'PDCNet' in opt.model:
        name_to_save += '_{}'.format(opt.multi_stage_type)
    name_to_save += '_{}'.format(opt.pre_trained_model)
    '''
    alternative would be to consider cyclic consistency error directly instead of inverse cyclic consistency error. 
    confident_mask_type = 'cyclic_consistency_error_below_{}'.format(opt.cyclic_consistency_mask_threshold) if \
    'PDCNet' not in opt.model else opt.confident_mask_type
    '''

    vs = VideoStreamer(opt.input, opt.resize, opt.skip, opt.image_glob, opt.max_length)
    frame, ret = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'

    last_frame = frame
    if len(frame.shape) == 2:
        last_frame = cv2.cvtColor(last_frame, cv2.COLOR_GRAY2RGB)
    last_data = {'image0': last_frame}
    last_image_id = 0

    if opt.save_dir is not None:
        print('==> Will write outputs to {}'.format(opt.save_dir))
        Path(opt.save_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not opt.no_display:
        cv2.namedWindow('{} matches'.format(opt.model), cv2.WINDOW_NORMAL)
        cv2.resizeWindow('{} matches'.format(opt.model), (640*2, 480))
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the anchor\n'
          '\te/r: increase/decrease the keypoint confidence threshold\n'
          '\td/f: increase/decrease the match filtering threshold\n'
          '\tk: toggle the visualization of keypoints\n'
          '\tq: quit')

    timer = AverageTimer()

    while True:
        frame, ret = vs.next_frame()
        if not ret:
            print('Finished demo_superglue.py')
            break
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        timer.update('data')
        stem_t, stem_s = last_image_id, vs.i - 1

        pred = network.get_matches_and_confidence(target_img=torch.from_numpy(last_data['image0']).permute(2, 0, 1)
                                                  .unsqueeze(0), source_img=torch.from_numpy(frame).permute(2, 0, 1)
                                                  .unsqueeze(0), confident_mask_type=confident_mask_type)

        timer.update('forward')

        mkpts_s = pred['kp_source']
        mkpts_t = pred['kp_target']
        confidence = pred['confidence_value']
        text = [
            '{}'.format(name_to_save),
            'Top 1000 Matches out of {}'.format(len(mkpts_s))
        ]
        # select only a subset of matches (the most confident)
        mkpts_s = mkpts_s[:1000]
        mkpts_t = mkpts_t[:1000]
        confidence = confidence[:1000]

        flow_t_to_s = pred['flow'].squeeze().permute(1, 2, 0).cpu().numpy()
        mask_t_to_s = pred['mask'].squeeze().float().cpu().numpy()

        color = cm.jet(confidence)

        small_text = [
            'Image Pair: {:06}:{:06}'.format(stem_s, stem_t),
            'Mask type: {} '.format(confident_mask_type)
        ]

        if opt.visualize_warp:
            warped_frame = remap_using_flow_fields(frame, flow_t_to_s[:, :, 0], flow_t_to_s[:, :, 1]).astype(np.uint8)
            if not opt.mask_uncertain_regions:

                warped_and_overlay_image = overlay_semantic_mask(warped_frame,
                                                                 ann=255 - mask_t_to_s.astype(np.uint8) * 255,
                                                                 color=[51, 102, 255])  # BGR instead of RGB
                small_text.append('(red regions are predicted uncertain in the 3rd image)')
            else:
                warped_and_overlay_image = warped_frame * np.tile(np.expand_dims(mask_t_to_s, axis=2), (1, 1, 3))
                small_text.append('Only confident warped regions are shown in the 3rd image')
            out = make_matching_and_warping_plot_fast(
                frame, last_frame, kpts0=None, kpts1=None, mkpts0=mkpts_s, mkpts1=mkpts_t, color=color, text=text,
                warped_and_overlay_image=warped_and_overlay_image, path=None, show_keypoints=False,
                small_text=small_text)
        else:
            out = make_matching_plot_fast(
                frame, last_frame, kpts0=None, kpts1=None, mkpts0=mkpts_s, mkpts1=mkpts_t, color=color, text=text,
                path=None, show_keypoints=False, small_text=small_text)

        if not opt.no_display:
            if opt.save_video:
                writer.write(out)
            cv2.imshow('{} matches'.format(opt.model), out)
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                vs.cleanup()
                print('Exiting (via q) demo_superglue.py')
                break
            elif key == 'n':  # set the current frame as anchor
                last_data['image0'] = frame
                last_frame = frame
                last_image_id = (vs.i - 1)

        timer.update('viz')
        timer.print()

        if opt.save_dir is not None:
            # stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
            stem = 'matches_{:06}_{:06}'.format(stem_s, stem_t)
            out_file = str(Path(opt.save_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, out)

    if not opt.no_display:
        cv2.destroyAllWindows()

    if opt.save_video:
        print('Saving video...')
        name = os.path.basename(opt.save_dir)
        make_and_save_video(opt.save_dir, os.path.join(opt.save_dir, '{}'.format(name) + '.mp4'), rate=15)
    vs.cleanup()
