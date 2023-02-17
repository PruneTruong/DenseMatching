import math
import numpy as np
import torch


from .losses_on_matching_and_non_matching_pairs import SupervisionStrategy, EntropyObjective, MaxScoreObjective


class NegMinCorrelationEntropyLoss(SupervisionStrategy):
    """
    Maximizes the correlation entropy of non-matching image pairs (negative).
    """
    def __init__(self, apply_layer_selection_loss=True, rescale_factor=4, log2=True, activation='l1norm',
                 temperature=1.0):
        self.num_negatives = 0
        self.objective = EntropyObjective(rescale_factor=rescale_factor, log2=log2, activation=activation,
                                          temperature=temperature)
        self.apply_layer_selection_loss = apply_layer_selection_loss

    def get_image_pair(self, batch, *args):
        """Forms positive/negative image paris for weakly-supervised training"""
        training = args[0]
        self.bsz = len(batch['source_image'])

        if training:
            shifted_idx = np.roll(np.arange(self.bsz), -1)
            trg_img_neg = batch['target_image'][shifted_idx].clone()
            trg_cls_neg = batch['category_id'][shifted_idx].clone()
            neg_subidx = (batch['category_id'] - trg_cls_neg) != 0

            src_img = torch.cat([batch['source_image'], batch['source_image'][neg_subidx]], dim=0)
            trg_img = torch.cat([batch['target_image'], trg_img_neg[neg_subidx]], dim=0)
            self.num_negatives = neg_subidx.sum()
        else:
            src_img, trg_img = batch['source_image'], batch['target_image']
            self.num_negatives = 0

        return src_img, trg_img

    def get_correlation(self, correlation_matrix):
        """Returns correlation matrices of 'POSITIVE PAIRS' in a batch"""
        return correlation_matrix[:self.bsz].clone().detach()

    def compute_loss(self, correlation_matrix, layer_sel=None, *args, **kwargs):
        """Weakly-supervised matching loss on negative image pairs"""

        b = correlation_matrix.shape[0]
        if len(correlation_matrix.shape) == 3:
            h = w = int(math.sqrt(correlation_matrix.shape[-1]))
        else:
            h, w = correlation_matrix.shape[-2:]

        correlation_matrix = correlation_matrix.view(b, -1, h, w)[:, :h * w]  # remove the bin here, if it exists

        stats = {}
        loss_neg = self.objective.information_entropy(correlation_matrix[self.bsz:]) if self.num_negatives > 0 \
            else torch.as_tensor(1.0)  # want to maximize it
        loss_net = - loss_neg

        stats['avg_max_score_pos'] = torch.max(correlation_matrix[:self.bsz].detach(), dim=1)[0].mean()
        if self.num_negatives > 0:
            stats['avg_max_score_neg'] = torch.max(correlation_matrix[self.bsz:].detach(), dim=1)[0].mean()
        stats['Loss_neg'] = - loss_neg.item()
        stats['Loss_pos_neg/total'] = loss_net.item()

        if self.apply_layer_selection_loss:
            if layer_sel is None:
                raise ValueError
            loss_sel = self.objective.layer_selection_loss(layer_sel)
            loss_net += loss_sel
            stats['Loss_layer/total'] = loss_sel.item()
            stats['Loss_ori_dhpf/total'] = loss_net.item()
        return loss_net, stats


class NegMaxCorrelationScoresLoss(SupervisionStrategy):
    """ Applied only on non-matching image pairs (negative). It minimizes the mean matching score of the hard assigned
    matches of the cost volume computed between non-matching image pairs (negative). """
    def __init__(self, apply_layer_selection_loss=False, activation='softmax', temperature=1.0):
        self.objective = MaxScoreObjective(activation=activation, temperature=temperature)
        self.num_negatives = 0
        self.apply_layer_selection_loss = apply_layer_selection_loss

    def get_image_pair(self, batch, *args):
        r"""Forms positive/negative image paris for weakly-supervised training"""
        training = args[0]
        self.bsz = len(batch['source_image'])

        if training:
            shifted_idx = np.roll(np.arange(self.bsz), -1)
            trg_img_neg = batch['target_image'][shifted_idx].clone()
            trg_cls_neg = batch['category_id'][shifted_idx].clone()
            neg_subidx = (batch['category_id'] - trg_cls_neg) != 0

            src_img = torch.cat([batch['source_image'], batch['source_image'][neg_subidx]], dim=0)
            trg_img = torch.cat([batch['target_image'], trg_img_neg[neg_subidx]], dim=0)
            self.num_negatives = neg_subidx.sum()
        else:
            src_img, trg_img = batch['source_image'], batch['target_image']
            self.num_negatives = 0

        return src_img, trg_img

    def get_correlation(self, correlation_matrix):
        r"""Returns correlation matrices of 'POSITIVE PAIRS' in a batch"""
        return correlation_matrix[:self.bsz].clone().detach()

    def compute_loss(self, correlation_matrix, layer_sel=None, *args, **kwargs):
        """Weakly-supervised matching loss on negative image pairs"""

        b = correlation_matrix.shape[0]
        if len(correlation_matrix.shape) == 3:
            h = w = int(math.sqrt(correlation_matrix.shape[-1]))
        else:
            h, w = correlation_matrix.shape[-2:]

        correlation_matrix = correlation_matrix.view(b, -1, h, w)[:, :h * w]  # remove the bin here, if it exists

        stats = {}
        # only looks at the negative image pairs
        loss_neg = self.objective.compute_mean_max_score(correlation_matrix[self.bsz:]) if \
            self.num_negatives > 0 else torch.as_tensor(0.0)

        loss_net = loss_neg

        stats['avg_max_score_pos'] = torch.max(correlation_matrix[:self.bsz].squeeze().detach(), dim=1)[0].mean()
        if self.num_negatives > 0:
            stats['avg_max_score_neg'] = torch.max(correlation_matrix[self.bsz:].squeeze().detach(), dim=1)[0].mean()
        stats['Loss_neg'] = loss_neg.item()
        stats['Loss_pos_neg/total'] = loss_net.item()

        if self.apply_layer_selection_loss:
            if layer_sel is None:
                raise ValueError
            loss_sel = self.objective.layer_selection_loss(layer_sel)
            loss_net += loss_sel
            stats['Loss_layer/total'] = loss_sel.item()
            stats['Loss_ori_dhpf/total'] = loss_net.item()
        return loss_net, stats
