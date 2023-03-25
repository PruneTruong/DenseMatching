import torch
from packaging import version
import torch.nn.functional as F

from utils_flow.pixel_wise_mapping import warp
from utils_flow.flow_and_mapping_operations import create_border_mask


class BatchedImageTripletCreation:
    """ Class responsible for creating the image triplet used in the warp-consistency graph, from a pair of source and
    target images.
    Particularly, from the source and target image pair (which are usually not related by a known ground-truth flow),
    it creates an image triplet (source, target and target_prime) where the target prime image is related to the
    target by a known synthetically and randomly generated flow field (the flow generator is specified by user
    with argument 'synthetic_flow_generator').
    The final image triplet is obtained by cropping central patches of the desired dimensions in the three images.
    """

    def __init__(self, settings, synthetic_flow_generator, crop_size, output_size,
                 compute_mask_zero_borders=False, min_percent_valid_corr=0.1, padding_mode='border'):
        """
        Args:
            settings: settings
            synthetic_flow_generator: class responsible for generating a synthetic flow field.
            crop_size: size of the center crop .
            output_size: size of the final outputted images and flow fields (resized after the crop).
            compute_mask_zero_borders: compute the mask of zero borders in target prime image? will be equal to 0
                                       where the target prime image is 0, 1 otherwise.
            min_percent_valid_corr: minimum percentage of matching regions between target prime and target. Otherwise,
                                    use ground-truth correspondence mask.
            padding_mode: for warping the target_image to obtain the target_image_prime. 'border' could be better.
        """
        self.compute_mask_zero_borders = compute_mask_zero_borders

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.synthetic_flow_generator = synthetic_flow_generator
        if not isinstance(crop_size, (tuple, list)):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size

        if not isinstance(output_size, (tuple, list)):
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.min_percent_valid_corr = min_percent_valid_corr
        self.padding_mode = padding_mode

    def __call__(self, mini_batch, net=None, training=True, *args, **kwargs):
        """
        This takes the original mini_batch (with info for target and source), and creates a new one 'output_mini_batch'
        where all the data fits the required format of the triplet.
        In the triplet (output_mini_batch), the flow_gt and correspondence_mask is between target prime
        and target, whereas in the ORIGINAL mini_batch, if provided, flow_map and correspondence_mask were
        between target and source. Therefore, if provided, the 'flow_map' and 'correspomdence_mask' of original
        mini_batch become 'flow_map_target_to_source' and 'correspondence_mask_target_to_source' in
        the output_mini_batch.
        In summary, input mini_batch and output_mini_batch outputted by this function are completely different.


        Args:
            mini_batch: The mini batch input data, should at least contain the fields 'source_image', 'target_image'.
                ATTENTION, if it contains 'flow_map' and 'correspondence_mask', this corresponds to the gt flow
                and valid flow mask between the target and the source. In the outputted batch, this will be renamed
                to 'flow_map_target_to_source' and 'correspondence_mask_target_to_source'. It can also contains the
                gt keypoints in the source and target image.
            training: bool indicating if we are in training or evaluation mode
            net: network
        Returns:
            output_mini_batch: output data block with at least the fields 'source_image', 'target_image',
                        'target_image_prime', 'flow_map', 'correspondence_mask'.
                        ATTENTION: 'flow_map' contains the synthetic flow field, relating the target_image_prime to
                        the target. This is NOT the same flow_map than in the original mini_batch. Similarly,
                        'correspondence_mask' identifies the valid (in-view) flow regions of the synthetic flow_map.

                        If self.compute_mask_zero_borders, will contain field 'mask_zero_borders'.

                        If ground-truth between source and target image is known (was provided), will contain the fields
                        'flow_map_target_to_source', 'correspondence_mask_target_to_source'.

                        If ground-truth keypoints in source and target were provided, will contain the fields
                        'target_kps' and 'source_kps'.
        """

        output_mini_batch = mini_batch.copy()
        # take original images
        source_image = output_mini_batch['source_image'].to(self.device)
        target_image = output_mini_batch['target_image'].to(self.device)
        b, _, h, w = source_image.shape

        if h < self.output_size[0] or w < self.output_size[1]:
            # should be larger, otherwise the warping will create huge black borders
            print('Image or flow is the same size than desired output size in warping dataset ! ')

        # get synthetic homography transformation from the synthetic flow generator
        # flow_gt here is between target prime and target
        flow_gt_trgprime_to_trg = self.synthetic_flow_generator(mini_batch=mini_batch, training=training, net=net)
        flow_gt_trgprime_to_trg.require_grad = False
        bs, _, h_f, w_f = flow_gt_trgprime_to_trg.shape

        if h_f != h or w_f != w:
            # reshape and rescale the flow so it has the size of the original images
            flow_gt_trgprime_to_trg = F.interpolate(flow_gt_trgprime_to_trg, (h, w), mode='bilinear',
                                                    align_corners=False)
            flow_gt_trgprime_to_trg[:, 0] *= float(w) / float(w_f)
            flow_gt_trgprime_to_trg[:, 1] *= float(h) / float(h_f)
        target_image_prime, mask_zero_borders = warp(target_image, flow_gt_trgprime_to_trg,
                                                     padding_mode=self.padding_mode,
                                                     return_mask=True)
        target_image_prime = target_image_prime.byte()
        # because here, i still have my images in [0-255] so for augmentations to work well, the images
        # should be uint8 will be interpolated with 'area'

        # if there exists a ground-truth flow between the source and target image, also modify it so it corresponds
        # to the new source and target images.
        if 'flow_map' in list(mini_batch.keys()):
            if isinstance(mini_batch['flow_map'], list):
                flow_gt_target_to_source = mini_batch['flow_map'][-1].to(self.device)
                mask_gt_target_to_source = mini_batch['correspondence_mask'][-1].to(self.device)
            else:
                flow_gt_target_to_source = mini_batch['flow_map'].to(self.device)
                mask_gt_target_to_source = mini_batch['correspondence_mask'].to(self.device)
        else:
            flow_gt_target_to_source = None
            mask_gt_target_to_source = None

        # crop a center patch from the images and the ground-truth flow field, so that black borders are removed
        x_start = w // 2 - self.crop_size[1] // 2
        y_start = h // 2 - self.crop_size[0] // 2
        source_image_resized = source_image[:, :, y_start: y_start + self.crop_size[0],
                                            x_start: x_start + self.crop_size[1]]
        target_image_resized = target_image[:, :, y_start: y_start + self.crop_size[0],
                                            x_start: x_start + self.crop_size[1]]
        target_image_prime_resized = target_image_prime[:, :, y_start: y_start + self.crop_size[0],
                                                        x_start: x_start + self.crop_size[1]]
        flow_gt_trgprime_to_trg_resized = flow_gt_trgprime_to_trg[:, :, y_start: y_start + self.crop_size[0],
                                                                  x_start: x_start + self.crop_size[1]]
        mask_zero_borders = mask_zero_borders[:, y_start: y_start + self.crop_size[0],
                                              x_start: x_start + self.crop_size[1]]

        if 'flow_map' in list(mini_batch.keys()):
            flow_gt_target_to_source = flow_gt_target_to_source[:, :, y_start: y_start + self.crop_size[0],
                                                                x_start: x_start + self.crop_size[1]]
            if mask_gt_target_to_source is not None:
                mask_gt_target_to_source = mask_gt_target_to_source[:, y_start: y_start + self.crop_size[0],
                                                                    x_start: x_start + self.crop_size[1]]

        if 'target_kps' in mini_batch.keys():
            target_kp = mini_batch['target_kps'].to(self.device).clone()  # b, N, 2
            source_kp = mini_batch['source_kps'].to(self.device).clone()  # b, N, 2
            source_kp[:, :, 0] = source_kp[:, :, 0] - x_start   # will just make the not valid part even smaller
            source_kp[:, :, 1] = source_kp[:, :, 1] - y_start
            target_kp[:, :, 0] = target_kp[:, :, 0] - x_start
            target_kp[:, :, 1] = target_kp[:, :, 1] - y_start

        # resize to final output size, this is to prevent the crop from removing all common areas
        if self.output_size != self.crop_size:
            source_image_resized = F.interpolate(source_image_resized, self.output_size,
                                                 mode='area')
            target_image_resized = F.interpolate(target_image_resized, self.output_size,
                                                 mode='area')
            target_image_prime_resized = F.interpolate(target_image_prime_resized, self.output_size,
                                                       mode='area')
            flow_gt_trgprime_to_trg_resized = F.interpolate(flow_gt_trgprime_to_trg_resized, self.output_size,
                                                            mode='bilinear', align_corners=False)
            flow_gt_trgprime_to_trg_resized[:, 0] *= float(self.output_size[1]) / float(self.crop_size[1])
            flow_gt_trgprime_to_trg_resized[:, 1] *= float(self.output_size[0]) / float(self.crop_size[0])

            mask_zero_borders = F.interpolate(mask_zero_borders.float().unsqueeze(1), self.output_size,
                                              mode='bilinear', align_corners=False).floor().squeeze(1)

            if 'flow_map' in list(mini_batch.keys()):
                flow_gt_target_to_source = F.interpolate(flow_gt_target_to_source, self.output_size,
                                                         mode='bilinear', align_corners=False)
                flow_gt_target_to_source[:, 0] *= float(self.output_size[1]) / float(self.crop_size[1])
                flow_gt_target_to_source[:, 1] *= float(self.output_size[0]) / float(self.crop_size[0])
                if mask_gt_target_to_source is not None:
                    mask_gt_target_to_source = F.interpolate(mask_gt_target_to_source.unsqueeze(1).float(),
                                                             self.output_size,
                                                             mode='bilinear', align_corners=False).floor()
                    mask_gt_target_to_source = mask_gt_target_to_source.bool() if \
                        version.parse(torch.__version__) >= version.parse("1.1") else mask_gt_target_to_source.byte()

            # if target kps, also resize them
            if 'target_kps' in mini_batch.keys():
                source_kp[:, :, 0] *= float(self.output_size[1]) / float(self.crop_size[1])
                source_kp[:, :, 1] *= float(self.output_size[0]) / float(self.crop_size[0])
                target_kp[:, :, 0] *= float(self.output_size[1]) / float(self.crop_size[1])
                target_kp[:, :, 1] *= float(self.output_size[0]) / float(self.crop_size[0])

        # create ground truth correspondence mask for flow between target prime and target
        mask_gt = create_border_mask(flow_gt_trgprime_to_trg_resized)
        mask_gt = mask_gt.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask_gt.byte()

        if self.compute_mask_zero_borders:
            # if mask_gt has too little commun areas, overwrite to use that mask in anycase
            if mask_gt.sum() < mask_gt.shape[-1] * mask_gt.shape[-2] * self.min_percent_valid_corr:
                mask_zero_borders = mask_gt
            '''
            else:
                # if padding mode is zeros, can compute the mask_zero_borders like this, from the intensity of the image
                # mask black borders that might have appeared from the warping, when creating target_image_prime
                mask_zero_borders = define_mask_zero_borders(target_image_prime_resized)
            '''
            output_mini_batch['mask_zero_borders'] = mask_zero_borders.bool() if \
                version.parse(torch.__version__) >= version.parse("1.1") else mask_zero_borders.byte()

        if 'target_kps' in mini_batch.keys():
            output_mini_batch['target_kps'] = target_kp  # b, N, 2
            output_mini_batch['source_kps'] = source_kp  # b, N, 2

        if 'flow_map' in list(mini_batch.keys()):
            # gt between target and source (what we are trying to estimate during training unsupervised)
            # it corresponds to 'flow_map' of the ORIGINAL mini_batch.
            output_mini_batch['flow_map_target_to_source'] = flow_gt_target_to_source
            output_mini_batch['correspondence_mask_target_to_source'] = mask_gt_target_to_source

        # save the new batch information
        output_mini_batch['source_image'] = source_image_resized.byte()
        output_mini_batch['target_image'] = target_image_resized.byte()
        output_mini_batch['target_image_prime'] = target_image_prime_resized.byte()  # if apply transfo after
        output_mini_batch['correspondence_mask'] = mask_gt
        output_mini_batch['flow_map'] = flow_gt_trgprime_to_trg_resized
        # between target_prime and target, replace the old one
        return output_mini_batch


class BatchedImageTripletCreation2Flows:
    """ Class responsible for creating the image triplet used in the warp-consistency graph, from a pair of source and
    target images.
    Particularly, from the source and target image pair (which are usually not related by a known ground-truth flow),
    it creates an image triplet (source, target and target_prime) where the target prime image is related to the
    target by a known synthetically and randomly generated flow field (the flow generator is specified by user
    with argument 'synthetic_flow_generator').
    The final image triplet is obtained by cropping central patches of the desired dimensions in the three images.
    Here, two image triplets are actually created with different synthetic flow fields, resulting in different
    target_prime images, to apply different losses on each. The tensors corresponding to the second image triplet
    have the suffix '_ss' at the end of all fieldnames.
    """

    def __init__(self, settings, synthetic_flow_generator_for_unsupervised,
                 synthetic_flow_generator_for_self_supervised, crop_size, output_size,
                 compute_mask_zero_borders=False,
                 min_percent_valid_corr=0.1, padding_mode='border'):
        """
        Args:
            settings: settings
            synthetic_flow_generator_for_unsupervised: class responsible for generating a synthetic flow field.
            synthetic_flow_generator_for_self_supervised: class responsible for generating a synthetic flow field.
            crop_size: size of the center crop .
            output_size: size of the final outputted images and flow fields (resized after the crop).
            compute_mask_zero_borders: compute the mask of zero borders in target prime image? will be equal to 0
                                       where the target prime image is 0, 1 otherwise.
            min_percent_valid_corr: minimum percentage of matching regions between target prime and target. Otherwise,
                                    use ground-truth correspondence mask.
            padding_mode: for warping the target_image to obtain the target_image_prime. 'border' could be better
        """

        self.synthetic_flow_generator_for_unsupervised = synthetic_flow_generator_for_unsupervised
        self.synthetic_flow_generator_for_self_supervised = synthetic_flow_generator_for_self_supervised

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        if not isinstance(crop_size, (tuple, list)):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size

        if not isinstance(output_size, (tuple, list)):
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.min_percent_valid_corr = min_percent_valid_corr
        self.compute_mask_zero_borders = compute_mask_zero_borders
        self.padding_mode = padding_mode

    def compute_correspondence_mask(self, flow_gt_resized, mask_zero_borders):
        # compute mask gt
        # create ground truth mask (for eval at least)
        mask_gt = create_border_mask(flow_gt_resized)
        mask_gt = mask_gt.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask_gt.byte()

        if self.compute_mask_zero_borders:
            # if mask_gt is all zero (no commun areas), overwrite to use the mask in anycase
            if mask_gt.sum() < mask_gt.shape[-1] * mask_gt.shape[-2] * self.min_percent_valid_corr:
                mask_zero_borders = mask_gt
        mask_zero_borders = mask_zero_borders.bool() if version.parse(torch.__version__) >= version.parse("1.1") else \
            mask_zero_borders.byte()
        return mask_zero_borders, mask_gt

    def __call__(self, mini_batch, grid=None, net=None, training=True, *args, **kwargs):
        """
        This takes the original mini_batch (with info for target and source), and creates a new one 'output_mini_batch'
        where all the data fits the required format of the triplet.
        In the triplet (output_mini_batch), the flow_gt and correspondence_mask is between target prime
        and target, whereas in the ORIGINAL mini_batch, if provided, flow_map and correspondence_mask were
        between target and source. Therefore, if provided, the 'flow_map' and 'correspomdence_mask' of original
        mini_batch become 'flow_map_target_to_source' and 'correspondence_mask_target_to_source' in
        the output_mini_batch.
        In summary, input mini_batch and output_mini_batch outputted by this function are completely different.


        Args:
            mini_batch: The mini batch input data, should at least contain the fields 'source_image', 'target_image'.
                ATTENTION, if it contains 'flow_map' and 'correspondence_mask', this corresponds to the gt flow
                and valid flow mask between the target and the source. In the outputted batch, this will be renamed
                to 'flow_map_target_to_source' and 'correspondence_mask_target_to_source'. It can also contains the
                gt keypoints in the source and target image.
            training: bool indicating if we are in training or evaluation mode
            net: network
        Returns:
            output_mini_batch: output data block with at least the fields 'source_image', 'target_image',
                        'target_image_prime', 'flow_map', 'correspondence_mask',
                        'target_image_prime_ss', 'flow_map_ss', 'correspondence_mask_ss'.

                        ATTENTION: 'flow_map' contains the synthetic flow field, relating the target_image_prime to
                        the target. This is NOT the same flow_map than in the original mini_batch. Similarly,
                        'correspondence_mask' identifies the valid (in-view) flow regions of the synthetic flow_map.

                        If self.compute_mask_zero_borders, will contain field 'mask_zero_borders' and
                        'mask_zero_borders_ss',

                        If ground-truth between source and target image is known (was provided), will contain the fields
                        'flow_map_target_to_source', 'correspondence_mask_target_to_source'.

                        If ground-truth keypoints in source and target were provided, will contain the fields
                        'target_kps' and 'source_kps'.
        """

        output_mini_batch = mini_batch.copy()
        # take original images
        source_image = mini_batch['source_image'].to(self.device)
        target_image = mini_batch['target_image'].to(self.device)
        b, _, h, w = source_image.shape

        if h <= self.output_size[0] or w <= self.output_size[1]:
            print('Image or flow is the same size than desired output size in warping dataset ! ')

        # get synthetic homography transformation from the synthetic flow generator
        # flow_gt_for_unsupervised here is between target prime and target
        flow_gt_for_unsupervised = self.synthetic_flow_generator_for_unsupervised(mini_batch=output_mini_batch,
                                                                                  training=training, net=net).detach()
        bs, _, h_f, w_f = flow_gt_for_unsupervised.shape
        if h_f != h or w_f != w:
            # reshape and rescale the flow so it has the size of the original images
            flow_gt_for_unsupervised = F.interpolate(flow_gt_for_unsupervised, (h, w),
                                                     mode='bilinear', align_corners=False)
            flow_gt_for_unsupervised[:, 0] *= float(w) / float(w_f)
            flow_gt_for_unsupervised[:, 1] *= float(h) / float(h_f)

        target_image_prime_for_unsupervised, mask_zero_borders_for_unsupervised = \
            warp(target_image, flow_gt_for_unsupervised, padding_mode=self.padding_mode, return_mask=True)
        target_image_prime_for_unsupervised = target_image_prime_for_unsupervised.byte()

        # for self-supervised
        flow_gt_for_self_supervised = self.synthetic_flow_generator_for_self_supervised(mini_batch=output_mini_batch,
                                                                                        training=training,
                                                                                        net=net).detach()
        bs, _, h_f, w_f = flow_gt_for_self_supervised.shape
        if h_f != h or w_f != w:
            # reshape and rescale the flow so it has the size of the original images
            flow_gt_for_self_supervised = F.interpolate(flow_gt_for_self_supervised, (h, w), mode='bilinear',
                                                        align_corners=False)
            flow_gt_for_self_supervised[:, 0] *= float(w) / float(w_f)
            flow_gt_for_self_supervised[:, 1] *= float(h) / float(h_f)
        target_image_prime_for_self_supervised, mask_zero_borders_for_self_supervised = \
            warp(target_image, flow_gt_for_self_supervised, padding_mode=self.padding_mode, return_mask=True)
        target_image_prime_for_self_supervised = target_image_prime_for_self_supervised.byte()

        # flow between source and target if it exists
        if 'flow_map' in list(output_mini_batch.keys()):
            flow_gt_target_to_source = mini_batch['flow_map'].to(self.device)
            mask_gt_target_to_source = mini_batch['correspondence_mask'].to(self.device)
        else:
            flow_gt_target_to_source = None
            mask_gt_target_to_source = None

        # crop a center patch from the images and the ground-truth flow field, so that black borders are removed
        x_start = w // 2 - self.crop_size[1] // 2
        y_start = h // 2 - self.crop_size[0] // 2
        source_image_resized = source_image[:, :, y_start: y_start + self.crop_size[0],
                                            x_start: x_start + self.crop_size[1]]
        target_image_resized = target_image[:, :, y_start: y_start + self.crop_size[0],
                                            x_start: x_start + self.crop_size[1]]
        target_image_prime_for_unsupervised_resized = target_image_prime_for_unsupervised \
            [:, :, y_start: y_start + self.crop_size[0], x_start: x_start + self.crop_size[1]]
        flow_gt_for_unsupervised_resized = flow_gt_for_unsupervised[:, :, y_start: y_start + self.crop_size[0],
                                                                    x_start: x_start + self.crop_size[1]]
        mask_zero_borders_for_unsupervised = mask_zero_borders_for_unsupervised[:, y_start: y_start + self.crop_size[0],
                                                                                x_start: x_start + self.crop_size[1]]

        # for self-supervised
        target_image_prime_for_self_supervised_resized = target_image_prime_for_self_supervised \
            [:, :, y_start: y_start + self.crop_size[0], x_start: x_start + self.crop_size[1]]
        flow_gt_for_self_supervised_resized = flow_gt_for_self_supervised[:, :, y_start: y_start + self.crop_size[0],
                                                                          x_start: x_start + self.crop_size[1]]
        mask_zero_borders_for_self_supervised = \
            mask_zero_borders_for_self_supervised[:, y_start: y_start + self.crop_size[0],
                                                  x_start: x_start + self.crop_size[1]]

        if 'flow_map' in list(mini_batch.keys()):
            flow_gt_target_to_source = flow_gt_target_to_source[:, :, y_start: y_start + self.crop_size[0],
                                                                x_start: x_start + self.crop_size[1]]
            if mask_gt_target_to_source is not None:
                mask_gt_target_to_source = mask_gt_target_to_source[:, y_start: y_start + self.crop_size[0],
                                                                    x_start: x_start + self.crop_size[1]]

        if 'target_kps' in mini_batch.keys():
            target_kp = mini_batch['target_kps'].to(self.device).clone()  # b, N, 2
            source_kp = mini_batch['source_kps'].to(self.device).clone()  # b, N, 2
            source_kp[:, :, 0] = source_kp[:, :, 0] - x_start  # will just make the not valid part even smaller
            source_kp[:, :, 1] = source_kp[:, :, 1] - y_start
            target_kp[:, :, 0] = target_kp[:, :, 0] - x_start
            target_kp[:, :, 1] = target_kp[:, :, 1] - y_start

        # resize to final outptu size, this is to prevent the crop from removing all common areas
        if self.output_size != self.crop_size:
            source_image_resized = F.interpolate(source_image_resized, self.output_size, mode='area')
            target_image_resized = F.interpolate(target_image_resized, self.output_size, mode='area')
            target_image_prime_for_unsupervised_resized = F.interpolate(target_image_prime_for_unsupervised_resized,
                                                                        self.output_size, mode='area')
            flow_gt_for_unsupervised_resized = F.interpolate(flow_gt_for_unsupervised_resized, self.output_size,
                                                             mode='bilinear', align_corners=False)
            flow_gt_for_unsupervised_resized[:, 0] *= float(self.output_size[1]) / float(self.crop_size[1])
            flow_gt_for_unsupervised_resized[:, 1] *= float(self.output_size[0]) / float(self.crop_size[0])

            mask_zero_borders_for_unsupervised = \
                F.interpolate(mask_zero_borders_for_unsupervised.float().unsqueeze(1), self.output_size,
                              mode='bilinear', align_corners=False).floor().squeeze(1)

            target_image_prime_for_self_supervised_resized = \
                F.interpolate(target_image_prime_for_self_supervised_resized, self.output_size, mode='area')
            flow_gt_for_self_supervised_resized = F.interpolate(flow_gt_for_self_supervised_resized,
                                                                self.output_size, mode='bilinear', align_corners=False)
            flow_gt_for_self_supervised_resized[:, 0] *= float(self.output_size[1]) / float(self.crop_size[1])
            flow_gt_for_self_supervised_resized[:, 1] *= float(self.output_size[0]) / float(self.crop_size[0])

            mask_zero_borders_for_self_supervised = \
                F.interpolate(mask_zero_borders_for_self_supervised.float().unsqueeze(1), self.output_size,
                              mode='bilinear', align_corners=False).floor().squeeze(1)

            if 'flow_map' in list(mini_batch.keys()):
                flow_gt_target_to_source = F.interpolate(flow_gt_target_to_source, self.output_size,
                                                         mode='bilinear', align_corners=False)
                flow_gt_target_to_source[:, 0] *= float(self.output_size[1]) / float(self.crop_size[1])
                flow_gt_target_to_source[:, 1] *= float(self.output_size[0]) / float(self.crop_size[0])
                if mask_gt_target_to_source is not None:
                    mask_gt_target_to_source = F.interpolate(mask_gt_target_to_source.unsqueeze(1).float(),
                                                             self.output_size, mode='bilinear',
                                                             align_corners=False).floor()
                    mask_gt_target_to_source = mask_gt_target_to_source.bool() if \
                        version.parse(torch.__version__) >= version.parse("1.1") else mask_gt_target_to_source.byte()

            if 'target_kps' in mini_batch.keys():
                source_kp[:, :, 0] *= float(self.output_size[1]) / float(self.crop_size[1])
                source_kp[:, :, 1] *= float(self.output_size[0]) / float(self.crop_size[0])
                target_kp[:, :, 0] *= float(self.output_size[1]) / float(self.crop_size[1])
                target_kp[:, :, 1] *= float(self.output_size[0]) / float(self.crop_size[0])

        mask_zero_borders_for_unsupervised, mask_gt_for_unsupervised = self.compute_correspondence_mask(
            flow_gt_for_unsupervised_resized, mask_zero_borders_for_unsupervised)

        mask_zero_borders_for_self_supervised, mask_gt_for_self_supervised = self.compute_correspondence_mask(
            flow_gt_for_self_supervised_resized, mask_zero_borders_for_self_supervised)

        if 'flow_map' in list(mini_batch.keys()):
            # gt between target and source (what we are trying to estimate during training unsupervised)
            output_mini_batch['flow_map_target_to_source'] = flow_gt_target_to_source
            output_mini_batch['correspondence_mask_target_to_source'] = mask_gt_target_to_source

        if 'target_kps' in mini_batch.keys():
            output_mini_batch['target_kps'] = target_kp  # b, N, 2
            output_mini_batch['source_kps'] = source_kp  # b, N, 2

        # save the new batch information
        output_mini_batch['source_image'] = source_image_resized.byte()
        output_mini_batch['target_image'] = target_image_resized.byte()

        # for unsupervised estimation
        output_mini_batch['target_image_prime'] = target_image_prime_for_unsupervised_resized
        output_mini_batch['correspondence_mask'] = mask_gt_for_unsupervised
        if self.compute_mask_zero_borders:
            output_mini_batch['mask_zero_borders'] = mask_zero_borders_for_unsupervised
            # flow map gt between target_prime and target, replace the old one
        output_mini_batch['flow_map'] = flow_gt_for_unsupervised_resized

        # for self-supervised estimation
        output_mini_batch['target_image_prime_ss'] = target_image_prime_for_self_supervised_resized
        output_mini_batch['correspondence_mask_ss'] = mask_gt_for_self_supervised
        if self.compute_mask_zero_borders:
            output_mini_batch['mask_zero_borders_ss'] = mask_zero_borders_for_self_supervised
        # flow map gt between target_prime and target, replace the old one
        output_mini_batch['flow_map_ss'] = flow_gt_for_self_supervised_resized
        return output_mini_batch
