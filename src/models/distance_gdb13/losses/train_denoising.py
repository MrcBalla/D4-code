
##########################################################################################################
#
# FROM https://github.com/cvignac/DiGress/blob/main/dgd/metrics/train_metrics.py
#
##########################################################################################################

from typing import List

import torch
from torch import Tensor
import torch.nn as nn

import src.models.distance_gdb13.labels as labels

import torch.nn.functional as F

def generate_weights(mask: Tensor):
    num_elems = mask.flatten(start_dim=1).sum(dim=-1)
    alive_batches = (num_elems > 0).sum()
    weights_per_batch_elem = 1 / (num_elems * alive_batches)
    weights = torch.repeat_interleave(weights_per_batch_elem, num_elems)

    return weights
    

class TrainLoss_distance(nn.Module):
    """ Train with Cross entropy"""
    def __init__(
            self,
            lambda_train_E: float = 1.,
            lambda_train_ext_E: float = 1.,
            concat_edges: bool = False,
            weighted: bool = False,
            class_weighted: bool = False,
            **kwargs
        ):
        super().__init__()
        self.lambda_train_E = lambda_train_E
        self.lambda_train_ext_E = lambda_train_ext_E
        self.concat_edges = concat_edges
        self.weighted = weighted
        self.class_weighted = class_weighted

    def forward(
            self,
            pred_values: List[Tensor],
            true_values: List[Tensor],
            reduce: bool=True,
            ret_log: bool=False
        ):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """

        assert not self.weighted or (self.weighted and len(pred_values) == 8), "If weighted, pred_values must contain masks"

        if len(pred_values) == 5:
            pred_x, pred_e, pred_ext_e, pred_dist, pred_c = pred_values
        elif len(pred_values) == 8:
            pred_x, pred_e, pred_ext_e, nodes_mask, edges_mask, ext_edges_mask, pred_dist, pred_c = pred_values

        true_x, true_e, true_ext_e, true_pos, true_c = true_values

        # compute cross entropy loss
        reduction = 'mean' if reduce else 'none'

        reduction_to_do = reduction if not self.weighted else 'none'

        if self.class_weighted:
            edge_class_weights = torch.full((pred_e.shape[-1],), fill_value=5., device=pred_e.device)
            edge_class_weights[0] = 1.
        else:
            edge_class_weights = None


        loss_x = F.cross_entropy(pred_x, true_x, reduction=reduction_to_do) if true_x.numel() > 0 else torch.zeros(1, device=pred_x.device)
        loss_e = F.cross_entropy(pred_e, true_e, reduction=reduction_to_do, weight=edge_class_weights) if true_e.numel() > 0 else torch.zeros(1, device=pred_x.device)
        loss_c = F.cross_entropy(pred_c, true_c, reduction='mean') if true_c.numel() > 0 else torch.zeros(1, device=pred_c.device)
        loss_dist = F.mse_loss(pred_dist, true_pos, reduction='mean')
        
        if torch.isnan(loss_dist).item()==True:
            pass
        
        x_i_0=torch.nn.functional.sigmoid(pred_e[:,0])
        
        loss_first = F.binary_cross_entropy((x_i_0), (true_e==0).float(), reduction='none').mean()
        loss_second = (F.cross_entropy(pred_e[true_e!=0,1:], true_e[true_e!=0]-1, reduction='none')).sum()/pred_e.shape[0]
        
        total_loss: Tensor = loss_x.mean() + self.lambda_train_E * (loss_first + loss_second) + loss_c + loss_dist
        
        if ret_log:
            to_log = {
                labels.DENOISE_CE_X: loss_x.detach(),
                labels.DENOISE_CE_E: (loss_e).detach(),
                labels.DENOISE_CE_TOTAL: total_loss.detach(),
                labels.DENOISE_MSE_DIST: loss_dist.detach(),
                labels.DENOISE_CE_C: loss_c.detach(),
                labels.DENOISE_TOTAL: total_loss.detach()
            }
            return total_loss, to_log
        else:
            return total_loss