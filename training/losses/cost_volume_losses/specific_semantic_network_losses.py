r"""Training objectives of DHPF"""

from skimage import draw
import numpy as np
import torch


from .losses_on_matching_and_non_matching_pairs import SupervisionStrategy
from utils_flow.correlation_to_matches_utils import Norm


# ------------------ SPECIFIC TO DHPF -------------------------------------------------------
def where(predicate):
    r"""Predicate must be a condition on nd-tensor"""
    matching_indices = predicate.nonzero()
    if len(matching_indices) != 0:
        matching_indices = matching_indices.t().squeeze(0)
    return matching_indices


class Evaluator:
    r"""Computes evaluation metrics of PCK, LT-ACC, IoU"""
    def __init__(self, benchmark, alpha=0.1):
        if benchmark == 'caltech':
            self.eval_func = self.eval_mask_transfer
        else:
            self.eval_func = self.eval_kps_transfer
        self.alpha = alpha

    def evaluate(self, prd_kps, batch, predicted_target_pts=True):
        r"""Compute evaluation metric"""
        return self.eval_func(prd_kps, batch, predicted_target_pts=predicted_target_pts)

    def eval_kps_transfer(self, prd_kps, batch, predicted_target_pts=True):
        r"""Compute percentage of correct key-points (PCK) based on prediction"""

        easy_match = {'src': [], 'trg': [], 'dist': []}
        hard_match = {'src': [], 'trg': []}

        pck = []
        for idx in range(len(prd_kps)):
            # per batch
            if predicted_target_pts:
                pk = prd_kps[idx]
                tk = torch.t(batch['target_kps'][idx]).cuda()  # 2, N
            else:
                pk = prd_kps[idx]
                tk = torch.t(batch['source_kps'][idx]).cuda()  # we warp the target point to the source instead
            if 'pckthres' not in list(batch.keys()):
                raise ValueError
            thres = batch['pckthres'][idx]
            npt = batch['n_pts'][idx]

            # original image is always 240x240, reso of corre is 60, rescale kp to correlation and find the index in
            # flattened correlation, per image pair
            kp_s = batch['source_kps'][idx].clone()[:npt].cuda()  # Nx2
            kp_s *= 1.0 / 4.0
            kp_s = torch.round(kp_s)
            index_s = kp_s[:, 1] * 60 + kp_s[:, 0]  # N
            index_s = index_s.long()

            kp_t = batch['target_kps'][idx].clone()[:npt].cuda()  # Nx2
            kp_t *= 1.0 / 4.0
            kp_t = torch.round(kp_t)
            index_t = kp_t[:, 1] * 60 + kp_t[:, 0]  # N
            index_t = index_t.long()

            mask_in_corr = (kp_s[:, 0] > 0) & (kp_s[:, 1] > 0) & (kp_t[:, 0] > 0) & (kp_t[:, 1] > 0) & \
                           (kp_s[:, 0] < 60) & (kp_s[:, 1] < 60) & (kp_t[:, 0] < 60) & (kp_t[:, 1] < 60)

            index_t = index_t[mask_in_corr]
            index_s = index_s[mask_in_corr]

            correct_dist, correct_ids, incorrect_ids = self.classify_prd(pk[:, :npt][:, mask_in_corr],
                                                                         tk[:, :npt][:, mask_in_corr], thres)

            # Collect easy and hard match feature index & store pck to buffer

            easy_match['dist'].append(correct_dist)
            easy_match['src'].append(index_s[correct_ids])
            easy_match['trg'].append(index_t[correct_ids])
            hard_match['src'].append(index_s[incorrect_ids])
            hard_match['trg'].append(index_t[incorrect_ids])

            pck.append((len(correct_ids) / npt.item()) * 100)

        eval_result = {'easy_match': easy_match,
                       'hard_match': hard_match,
                       'pck': pck}

        return eval_result

    def eval_mask_transfer(self, prd_kps, batch):
        r"""Compute LT-ACC and IoU based on transferred points"""

        ltacc = []
        iou = []

        for idx, prd in enumerate(prd_kps):
            trg_n_pts = (batch['trg_kps'][idx] > 0)[0].sum()
            prd_kp = prd[:, :batch['n_pts'][idx]]
            trg_kp = batch['trg_kps'][idx][:, :trg_n_pts]

            imsize = list(batch['trg_img'].size())[2:]
            trg_xstr, trg_ystr = self.pts2ptstr(trg_kp)
            trg_mask = self.ptstr2mask(trg_xstr, trg_ystr, imsize[0], imsize[1])
            prd_xstr, pred_ystr = self.pts2ptstr(prd_kp)
            prd_mask = self.ptstr2mask(prd_xstr, pred_ystr, imsize[0], imsize[1])

            ltacc.append(self.label_transfer_accuracy(prd_mask, trg_mask))
            iou.append(self.intersection_over_union(prd_mask, trg_mask))

        eval_result = {'ltacc': ltacc,
                       'iou': iou}

        return eval_result

    def classify_prd(self, prd_kps, trg_kps, pckthres):
        r"""Compute the number of correctly transferred key-points"""
        l2dist = (prd_kps - trg_kps).pow(2).sum(dim=0).pow(0.5)
        thres = pckthres.expand_as(l2dist).float() * self.alpha
        correct_pts = torch.le(l2dist, thres.cuda())

        correct_ids = where(correct_pts == 1)
        incorrect_ids = where(correct_pts == 0)
        correct_dist = l2dist[correct_pts]

        return correct_dist, correct_ids, incorrect_ids

    @staticmethod
    def intersection_over_union(mask1, mask2):
        r"""Computes IoU between two masks"""
        rel_part_weight = torch.sum(torch.sum(mask2.gt(0.5).float(), 2, True), 3, True) / \
                          torch.sum(mask2.gt(0.5).float())
        part_iou = torch.sum(torch.sum((mask1.gt(0.5) & mask2.gt(0.5)).float(), 2, True), 3, True) / \
                   torch.sum(torch.sum((mask1.gt(0.5) | mask2.gt(0.5)).float(), 2, True), 3, True)
        weighted_iou = torch.sum(torch.mul(rel_part_weight, part_iou)).item()

        return weighted_iou

    @staticmethod
    def label_transfer_accuracy(mask1, mask2):
        r"""LT-ACC measures the overlap with emphasis on the background class"""
        return torch.mean((mask1.gt(0.5) == mask2.gt(0.5)).double()).item()

    @staticmethod
    def pts2ptstr(pts):
        r"""Convert tensor of points to string"""
        x_str = str(list(pts[0].cpu().numpy()))
        x_str = x_str[1:len(x_str)-1]
        y_str = str(list(pts[1].cpu().numpy()))
        y_str = y_str[1:len(y_str)-1]

        return x_str, y_str

    @staticmethod
    def pts2mask(x_pts, y_pts, shape):
        r"""Build a binary mask tensor base on given xy-points"""
        x_idx, y_idx = draw.polygon(x_pts, y_pts, shape)
        mask = np.zeros(shape, dtype=np.bool)
        mask[x_idx, y_idx] = True

        return mask

    def ptstr2mask(self, x_str, y_str, out_h, out_w):
        r"""Convert xy-point mask (string) to tensor mask"""
        x_pts = np.fromstring(x_str, sep=',')
        y_pts = np.fromstring(y_str, sep=',')
        mask_np = self.pts2mask(y_pts, x_pts, [out_h, out_w])
        mask = torch.tensor(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).float()

        return mask


class CEOriginalDHPF:
    def __init__(self, target_rate=0.5, alpha=0.1):
        self.target_rate = target_rate  # for layer selection
        self.alpha = alpha
        self.eps = 1e-30
        self.softmax = torch.nn.Softmax(dim=1)

    def cross_entropy(self, correlation_matrix_t_to_s, easy_match, hard_match, batch):
        r"""Computes sum of weighted cross-entropy values between ground-truth and prediction"""
        loss_buf = correlation_matrix_t_to_s.new_zeros(correlation_matrix_t_to_s.size(0))
        correlation_matrix_t_to_s = Norm.unit_gaussian_normalize(correlation_matrix_t_to_s)
        # shape is b, h_s*w_s, h_t*w_t

        for idx, (ct, thres, npt) in enumerate(zip(correlation_matrix_t_to_s, batch['pckthres'], batch['n_pts'])):

            # Hard (incorrect) match
            if len(hard_match['src'][idx]) > 0:
                cross_ent = self.compute_cross_entropy_loss(ct, hard_match['src'][idx], hard_match['trg'][idx])
                loss_buf[idx] += cross_ent.sum()

            # Easy (correct) match
            if len(easy_match['src'][idx]) > 0:
                cross_ent = self.compute_cross_entropy_loss(ct, easy_match['src'][idx], easy_match['trg'][idx])
                loss_buf[idx] += cross_ent.sum()

            loss_buf[idx] /= npt

        return torch.mean(loss_buf)

    def weighted_cross_entropy(self, correlation_matrix_t_to_s, easy_match, hard_match, batch):
        r"""Computes sum of weighted cross-entropy values between ground-truth and prediction"""
        loss_buf = correlation_matrix_t_to_s.new_zeros(correlation_matrix_t_to_s.size(0))
        correlation_matrix_t_to_s = Norm.unit_gaussian_normalize(correlation_matrix_t_to_s)
        # shape is b, h_s*w_s, h_t*w_t

        for idx, (ct, thres, npt) in enumerate(zip(correlation_matrix_t_to_s, batch['pckthres'], batch['n_pts'])):

            # Hard (incorrect) match
            if len(hard_match['src'][idx]) > 0:
                cross_ent = self.compute_cross_entropy_loss(ct, hard_match['src'][idx], hard_match['trg'][idx])
                loss_buf[idx] += cross_ent.sum()

            # Easy (correct) match
            if len(easy_match['src'][idx]) > 0:
                cross_ent = self.compute_cross_entropy_loss(ct, easy_match['src'][idx], easy_match['trg'][idx])
                smooth_weight = (easy_match['dist'][idx] / (thres * self.alpha)).pow(2)
                loss_buf[idx] += (smooth_weight * cross_ent).sum()

            loss_buf[idx] /= npt

        return torch.mean(loss_buf)

    def compute_cross_entropy_loss(self, correlation_matrix_t_to_s, src_match, trg_match):
        r"""Cross-entropy between predicted pdf and ground-truth pdf (one-hot vector)"""
        pdf = self.softmax(correlation_matrix_t_to_s.index_select(0, src_match))
        # correlation_matrix_t_to_s.index_select(0, src_match)  becomes N, h_t*w_t
        # shape is b, h_s*w_s, h_t*w_t
        prob = pdf[range(len(trg_match)), trg_match]
        cross_ent = -torch.log(prob + self.eps)

        return cross_ent


class StrongSupStrategyDHPF(SupervisionStrategy):
    def __init__(self, weighted_cross_entropy=True):
        self.objective = CEOriginalDHPF()
        self.weighted_cross_entropy = weighted_cross_entropy

    def get_image_pair(self, batch, *args):
        r"""Returns (semantically related) pairs for strongly-supervised training"""
        return batch['source_image'], batch['target_image']

    def get_correlation(self, correlation_matrix):
        r"""Returns correlation matrices of 'ALL PAIRS' in a batch"""
        return correlation_matrix.clone().detach()

    def compute_loss(self, correlation_matrix, layer_sel=None, batch=None, result_eval=None, *args):
        r"""Strongly-supervised matching loss (L_{match})"""
        easy_match = result_eval[0]['easy_match']
        hard_match = result_eval[0]['hard_match']

        stats = {}
        if self.weighted_cross_entropy:
            loss_cre = self.objective.weighted_cross_entropy(correlation_matrix, easy_match, hard_match, batch)
        else:
            loss_cre = self.objective.cross_entropy(correlation_matrix, easy_match, hard_match, batch)
        loss_sel = self.objective.layer_selection_loss(layer_sel)
        loss_net = loss_cre + loss_sel

        stats['Loss_cross_entropy'] = loss_cre.item()
        stats['Loss_layer_selection'] = loss_sel.item()
        stats['Loss'] = loss_net.item()
        return loss_net, stats


# -------------------------------- SPECIFC TO DCC-Net ------------------------------------------


class WeakSupStrategyDCCNet(SupervisionStrategy):
    def __init__(self, scaleloss_weight=1.0):
        self.num_negatives = 0
        self.scaleloss_weight = scaleloss_weight

    def get_negative_pairs(self, batch, *args):
        r"""Forms positive/negative image paris for weakly-supervised training"""
        training = args[0]
        self.bsz = len(batch['source_image'])

        if training:
            # is actually only negative
            shifted_idx = np.roll(np.arange(self.bsz), -1)
            trg_img_neg = batch['target_image'][shifted_idx].clone()
            trg_cls_neg = batch['category_id'][shifted_idx].clone()
            neg_subidx = (batch['category_id'] - trg_cls_neg) != 0

            src_img = batch['source_image'][neg_subidx]
            trg_img = trg_img_neg[neg_subidx]
            self.num_negatives = neg_subidx.sum()
        else:
            src_img, trg_img = batch['source_image'], batch['target_image']
            self.num_negatives = 0
        return src_img, trg_img

    def get_image_pair(self, batch, *args):
        r"""Forms positive/negative image paris for weakly-supervised training"""
        self.bsz = len(batch['source_image'])
        src_img, trg_img = batch['source_image'], batch['target_image']
        return src_img, trg_img

    def get_correlation(self, model_output):
        r"""Returns correlation matrices of 'POSITIVE PAIRS' in a batch"""
        return model_output['correlation_from_t_to_s'].clone().detach()

    def compute_loss(self, model_output_positive, model=None, batch=None, training=False, *args):
        r"""Weakly-supervised matching loss (L_{match})"""

        # positive
        score_pos_merge = model_output_positive['score_merge']
        score_pos_overscales = model_output_positive['score_overscales']

        # negative
        src_img_neg, trg_img_neg = self.get_negative_pairs(batch, training)
        if self.num_negatives > 0:
            model_output_negative = model({'source_image': src_img_neg, 'target_image': trg_img_neg})
            score_neg_merge = model_output_negative['score_merge']
            score_neg_overscales = model_output_negative['score_overscales']
        else:
            score_neg_merge = 0.0
            score_neg_overscales = 0.0

        # loss
        stats = {}
        loss_merge = score_neg_merge - score_pos_merge
        stats['Loss_pos'] = - score_pos_merge.item()
        stats['Loss_neg'] = score_neg_merge.item() if isinstance(score_neg_merge, torch.Tensor) else score_neg_merge
        stats['Loss_pos_neg'] = loss_merge.item()

        if self.scaleloss_weight:
            # compute the loss for the intermediate correlations
            loss_scales_pos = torch.sum(torch.cat(score_pos_overscales))
            loss_scales_neg = torch.sum(torch.cat(score_neg_overscales)) if self.num_negatives > 0 else 0.0
            loss_scales = loss_scales_neg - loss_scales_pos
            stats['Loss_pos_inter'] = - loss_scales_pos.item()
            stats['Loss_neg_inter'] = loss_scales_neg.item() if isinstance(loss_scales_neg, torch.Tensor) \
                else loss_scales_neg
            stats['Loss_pos_neg_inter'] = loss_scales.item()

            loss = loss_merge + self.scaleloss_weight*loss_scales

            stats['Loss_pos_neg_total'] = loss.item()
        else:
            loss = loss_merge

        return loss, stats

