import math
import torch.nn.functional as F
from abc import ABC, abstractmethod
import numpy as np
import torch


from utils_flow.correlation_to_matches_utils import Correlation, Norm


class SupervisionStrategy(ABC):
    """Different strategies for methods:"""

    @abstractmethod
    def get_image_pair(self, batch, *args):
        """Forms positive/negative image paris for weakly-supervised training"""
        pass

    @abstractmethod
    def get_correlation(self, correlation_matrix):
        """Returns correlation matrices of 'POSITIVE PAIRS' in a batch"""
        pass

    @abstractmethod
    def compute_loss(self, correlation_matrix, *args, **kwargs):
        """Compute weakly-supervised matching loss on positive and negative image pairs"""
        pass


class EntropyObjective:
    """ Computes the correlation entropy. (from DHPF)"""
    def __init__(self, target_rate=0.5, alpha=0.1, rescale_factor=4, log2=True, activation='l1norm', temperature=1.0):
        """
        Args:
            target_rate: useful for DHPF only. rate of selection of each layer.
            alpha: alpha for PCK computation.
            rescale_factor:
            log2: use log2 instead of log for cross entropy loss?
            activation: activation to apply to the row cost volume scores to convert them to probabilistic mappings.
            temperature: to apply to the cost volume scores before applying softmax (if activation is softmax)
        """
        self.rescale_factor = rescale_factor
        self.activation = activation
        self.temperature = temperature
        self.log2 = log2
        self.eps = 1e-30

        self.target_rate = target_rate   # for layer selection
        self.alpha = alpha

    def stable_softmax(self, A, dim=1):
        M, _ = A.max(dim=dim, keepdim=True)
        A = A - M  # subtract maximum value for stability
        return F.softmax(A / self.temperature, dim=dim)

    def information_entropy(self, correlation_matrix):
        r"""Computes information entropy of all candidate matches"""
        bsz = correlation_matrix.size(0)

        if 'no_mutual_nn' not in self.activation:
            correlation_matrix = Correlation.mutual_nn_filter(correlation_matrix)

        side = int(math.sqrt(correlation_matrix.size(1)))
        new_side = side // self.rescale_factor

        trg2src_dist = correlation_matrix.view(bsz, -1, side, side)
        src2trg_dist = correlation_matrix.view(bsz, side, side, -1).permute(0, 3, 1, 2)

        # Squeeze distributions for reliable entropy computation
        trg2src_dist = F.interpolate(trg2src_dist, [new_side, new_side], mode='bilinear', align_corners=True)
        src2trg_dist = F.interpolate(src2trg_dist, [new_side, new_side], mode='bilinear', align_corners=True)

        if 'l1norm' in self.activation:
            src_pdf = Norm.l1normalize(trg2src_dist.view(bsz, -1, (new_side * new_side)))
            trg_pdf = Norm.l1normalize(src2trg_dist.view(bsz, -1, (new_side * new_side)))
        elif 'softmax' in self.activation:
            src_pdf = F.softmax(trg2src_dist.view(bsz, -1, (new_side * new_side)).clone() / self.temperature, dim=2)
            trg_pdf = F.softmax(src2trg_dist.view(bsz, -1, (new_side * new_side)).clone() / self.temperature, dim=2)
        elif 'stable_softmax' in self.activation:
            src_pdf = self.stable_softmax(trg2src_dist.view(bsz, -1, (new_side * new_side)).clone(), dim=2)
            trg_pdf = self.stable_softmax(src2trg_dist.view(bsz, -1, (new_side * new_side)).clone(), dim=2)
        elif 'unit_gaussian_softmax' in self.activation:
            src_pdf = F.softmax(Norm.unit_gaussian_normalize(
                trg2src_dist.view(bsz, -1, (new_side * new_side)).clone()) / self.temperature, dim=2)
            trg_pdf = F.softmax(Norm.unit_gaussian_normalize(
                src2trg_dist.view(bsz, -1, (new_side * new_side)).clone()) / self.temperature, dim=2)
        elif 'noactivation' in self.activation:
            src_pdf = trg2src_dist.view(bsz, -1, (new_side * new_side)).clone()
            trg_pdf = src2trg_dist.view(bsz, -1, (new_side * new_side)).clone()
        else:
            raise ValueError

        src_pdf = src_pdf + self.eps
        trg_pdf = trg_pdf + self.eps

        if self.log2:
            src_ent = (-(src_pdf * torch.log2(src_pdf)).sum(dim=2)).view(bsz, -1)
            trg_ent = (-(trg_pdf * torch.log2(trg_pdf)).sum(dim=2)).view(bsz, -1)
        else:
            src_ent = (-(src_pdf * torch.log(src_pdf)).sum(dim=2)).view(bsz, -1)
            trg_ent = (-(trg_pdf * torch.log(trg_pdf)).sum(dim=2)).view(bsz, -1)
        score_net = (src_ent + trg_ent).mean(dim=1) / 2

        return score_net.mean()

    def layer_selection_loss(self, layer_sel):
        r"""Encourages model to select each layer at a certain rate"""
        return (layer_sel.mean(dim=0) - self.target_rate).pow(2).sum()


class PosNegMinCorrelationEntropyLoss(SupervisionStrategy):
    """Minimizes the correlation entropy of matching image pairs (positive), while maximizing it for non-matching
    image pairs (negative). """
    def __init__(self, apply_layer_selection_loss=True, final_loss='ratio', rescale_factor=4,
                 log2=True, activation='softmax_no_mutual_nn', temperature=1.0):
        """
        Args:
            apply_layer_selection_loss:
            final_loss:
            rescale_factor:
            log2:
            activation:
            temperature:
        """
        self.objective = EntropyObjective(rescale_factor=rescale_factor, log2=log2, activation=activation,
                                          temperature=temperature)
        self.num_negatives = 0
        self.apply_layer_selection_loss = apply_layer_selection_loss
        self.final_loss = final_loss
        self.eps = 1e-30

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
        """Weakly-supervised matching loss on positive and negative image pairs"""

        b = correlation_matrix.shape[0]
        if len(correlation_matrix.shape) == 3:
            h = w = int(math.sqrt(correlation_matrix.shape[-1]))
        else:
            h, w = correlation_matrix.shape[-2:]

        correlation_matrix = correlation_matrix.view(b, -1, h, w)[:, :h*w]  # remove the bin here, if it exists

        stats = {}
        loss_pos = self.objective.information_entropy(correlation_matrix[:self.bsz])
        loss_neg = self.objective.information_entropy(correlation_matrix[self.bsz:]) if self.num_negatives > 0 \
            else torch.as_tensor(1.0)

        if self.final_loss == 'ratio':
            loss_net = (loss_pos / loss_neg)
        elif self.final_loss == 'sum':
            loss_net = loss_pos - loss_neg
        else:
            raise ValueError

        stats['avg_max_score_pos'] = torch.max(correlation_matrix[:self.bsz].detach(), dim=1)[0].mean()
        if self.num_negatives > 0:
            stats['avg_max_score_neg'] = torch.max(correlation_matrix[self.bsz:].detach(), dim=1)[0].mean()
        stats['Loss_pos'] = loss_pos.item()
        stats['Loss_neg'] = loss_neg.item() if isinstance(loss_neg, torch.Tensor) else loss_neg
        stats['Loss_pos_neg/total'] = loss_net.item()

        if self.apply_layer_selection_loss:
            if layer_sel is None:
                raise ValueError
            loss_sel = self.objective.layer_selection_loss(layer_sel)
            loss_net += loss_sel
            stats['Loss_layer/total'] = loss_sel.item()
            stats['Loss_ori_dhpf/total'] = loss_net.item()
        return loss_net, stats


class MaxScoreObjective:
    """
    Computes the mean maximum scores of the hard assigned matches of a cost volume. (from NCNet)
    """
    def __init__(self, activation='softmax', temperature=1.0, target_rate=0.5):
        self.activation = activation
        self.temperature = temperature
        self.target_rate = target_rate   # for layer selection for dhpf

    def stable_softmax(self, A, dim=1):
        M, _ = A.max(dim=dim, keepdim=True)
        A = A - M  # subtract maximum value for stability
        return F.softmax(A / self.temperature, dim=dim)

    def compute_mean_max_score(self, corr4d):
        if self.activation is None:
            normalize = lambda x: x
        elif "softmax" in self.activation:
            normalize = lambda x: torch.nn.functional.softmax(x / self.temperature, 1)
        elif 'stable_softmax' in self.activation:
            normalize = lambda x: self.stable_softmax(x, dim=1)
        elif "l1norm" in self.activation:
            normalize = lambda x: x / (torch.sum(x, dim=1, keepdim=True) + 0.0001)
        elif "noactivation" in self.activation:
            normalize = lambda x: x
        else:
            raise ValueError

        if len(corr4d.shape) == 3:
            # b, h*w, h*w
            batch_size = corr4d.size(0)
            feature_size = int(math.sqrt(corr4d.shape[-1]))
        else:
            batch_size = corr4d.size(0)
            feature_size = corr4d.size(2)

        nc_B_Avec = corr4d.reshape(
            batch_size, feature_size * feature_size, feature_size, feature_size)  # [batch_idx,k_A,i_B,j_B]
        nc_A_Bvec = corr4d.reshape(
            batch_size, feature_size, feature_size, feature_size * feature_size).permute(0, 3, 1, 2)  #

        if 'with_mutual_nn' in self.activation:
            nc_A_Bvec = Correlation.mutual_nn_filter(nc_A_Bvec.view(batch_size, -1, feature_size * feature_size)) \
                .view(batch_size, -1, feature_size, feature_size)
            nc_B_Avec = Correlation.mutual_nn_filter(nc_B_Avec.view(batch_size, -1, feature_size * feature_size)) \
                .view(batch_size, -1, feature_size, feature_size)
        nc_B_Avec = normalize(nc_B_Avec)
        nc_A_Bvec = normalize(nc_A_Bvec)

        # compute matching scores
        scores_B, _ = torch.max(nc_B_Avec, dim=1)
        scores_A, _ = torch.max(nc_A_Bvec, dim=1)
        score = torch.mean(scores_A + scores_B) / 2
        return score

    def layer_selection_loss(self, layer_sel):
        r"""Encourages model to select each layer at a certain rate"""
        return (layer_sel.mean(dim=0) - self.target_rate).pow(2).sum()


class PosNegMaxCorrelationScoresLoss(SupervisionStrategy):
    """Maximizes the mean matching score of the hard assigned matches of the cost volume computed between matching
    images (positive), while minimizing this same quantity for non-matching image pairs (negative).
    Originally introduced by NC-Net. """

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
        """Weakly-supervised matching loss on positive and negative image pairs"""

        b = correlation_matrix.shape[0]
        if len(correlation_matrix.shape) == 3:
            h = w = int(math.sqrt(correlation_matrix.shape[-1]))
        else:
            h, w = correlation_matrix.shape[-2:]

        correlation_matrix = correlation_matrix.view(b, -1, h, w)[:, :h * w]  # remove the bin here, if it exists

        stats = {}
        loss_pos = self.objective.compute_mean_max_score(correlation_matrix[:self.bsz])
        loss_neg = self.objective.compute_mean_max_score(correlation_matrix[self.bsz:]) if \
            self.num_negatives > 0 else torch.as_tensor(0.0)

        loss_net = loss_neg - loss_pos

        stats['avg_max_score_pos'] = torch.max(correlation_matrix[:self.bsz].squeeze().detach(), dim=1)[0].mean()
        if self.num_negatives > 0:
            stats['avg_max_score_neg'] = torch.max(correlation_matrix[self.bsz:].squeeze().detach(), dim=1)[0].mean()
        stats['Loss_pos'] = - loss_pos.item()
        stats['Loss_neg'] = loss_neg.item() if isinstance(loss_neg, torch.Tensor) else loss_neg
        stats['Loss_pos_neg/total'] = loss_net.item()

        if self.apply_layer_selection_loss:
            if layer_sel is None:
                raise ValueError
            loss_sel = self.objective.layer_selection_loss(layer_sel)
            loss_net += loss_sel
            stats['Loss_layer/total'] = loss_sel.item()
            stats['Loss_ori_dhpf/total'] = loss_net.item()

        return loss_net, stats

