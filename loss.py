import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch import Tensor
from typing import Union

import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch import Tensor
from typing import Union


class FocalLoss(nn.Module):
    """Computes the focal loss between input and target
    as described here https://arxiv.org/abs/1708.02002v2

    Args:
        gamma (float):  The focal loss focusing parameter.
        weights (Union[None, Tensor]): Rescaling weight given to each class.
        If given, has to be a Tensor of size C. optional.
        reduction (str): Specifies the reduction to apply to the output.
        it should be one of the following 'none', 'mean', or 'sum'.
        default 'mean'.
        ignore_index (int): Specifies a target value that is ignored and
        does not contribute to the input gradient. optional.
        eps (float): smoothing to prevent log from returning inf.
    """
    def __init__(
            self,
            gamma = 2,
            weights: Union[None, Tensor] = None,
            reduction: str = 'mean',
            ignore_index=-100,
            eps=1e-16
            ) -> None:
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(
                'Reduction {} not implemented.'.format(reduction)
                )
        assert weights is None or isinstance(weights, Tensor), \
            'weights should be of type Tensor or None, but {} given'.format(
                type(weights))
        self.reduction = reduction
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps
        self.weights = weights

    def _get_weights(self, target: Tensor) -> Tensor:
        if self.weights is None:
            return torch.ones(target.shape[0])
        weights = target * self.weights
        return weights.sum(dim=-1)

    def _process_target(
            self, target: Tensor, num_classes: int, mask: Tensor
            ) -> Tensor:
        
        #convert all ignore_index elements to zero to avoid error in one_hot
        #note - the choice of value 0 is arbitrary, but it should not matter as these elements will be ignored in the loss calculation
        target = target * (target!=self.ignore_index) 
        target = target.view(-1)
        return one_hot(target, num_classes=num_classes)

    def _process_preds(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = torch.vstack([1 - x, x])
            x = x.permute(1, 0)
            return x
        return x.view(-1, x.shape[-1])

    def _calc_pt(
            self, target: Tensor, x: Tensor, mask: Tensor
            ) -> Tensor:
        p = target * x
        p = p.sum(dim=-1)
        p = p * ~mask
        return p

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        assert torch.all((x >= 0.0) & (x <= 1.0)), ValueError(
            'The predictions values should be between 0 and 1, \
                make sure to pass the values to sigmoid for binary \
                classification or softmax for multi-class classification'
        )
        mask = target == self.ignore_index
        mask = mask.view(-1)
        x = self._process_preds(x)
        num_classes = x.shape[-1]
        target = self._process_target(target, num_classes, mask)
        weights = self._get_weights(target).to(x.device)
        # weights = self._get_weights(target)
        pt = self._calc_pt(target, x, mask)
        focal = 1 - pt
        nll = -torch.log(self.eps + pt)
        nll = nll.masked_fill(mask, 0)
        loss = weights * (focal ** self.gamma) * nll
        return self._reduce(loss, mask, weights)

    def _reduce(self, x: Tensor, mask: Tensor, weights: Tensor) -> Tensor:
        if self.reduction == 'mean':
            return x.sum() / (~mask * weights).sum()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x
# class FocalLoss(nn.Module):
#     """Computes the focal loss between input and target
#     as described here https://arxiv.org/abs/1708.02002v2

#     Args:
#         gamma (float):  The focal loss focusing parameter.
#         weights (Union[None, Tensor]): Rescaling weight given to each class.
#         If given, has to be a Tensor of size C. optional.
#         reduction (str): Specifies the reduction to apply to the output.
#         it should be one of the following 'none', 'mean', or 'sum'.
#         default 'mean'.
#         ignore_index (int): Specifies a target value that is ignored and
#         does not contribute to the input gradient. optional.
#         eps (float): smoothing to prevent log from returning inf.
#     """
#     def __init__(
#             self,
#             gamma = 2,
#             weights: Union[None, Tensor] = None,
#             reduction: str = 'mean',
#             ignore_index=-100,
#             eps=1e-16
#             ) -> None:
#         super().__init__()
#         if reduction not in ['mean', 'none', 'sum']:
#             raise NotImplementedError(
#                 'Reduction {} not implemented.'.format(reduction)
#                 )
#         assert weights is None or isinstance(weights, Tensor), \
#             'weights should be of type Tensor or None, but {} given'.format(
#                 type(weights))
#         self.reduction = reduction
#         self.gamma = gamma
#         self.ignore_index = ignore_index
#         self.eps = eps
#         self.weights = weights

#     def _get_weights(self, target: Tensor) -> Tensor:
#         if self.weights is None:
#             return torch.ones(target.shape[0])
#         weights = target * self.weights
#         return weights.sum(dim=-1)

#     def _process_target(
#             self, target: Tensor, num_classes: int, mask: Tensor
#             ) -> Tensor:
        
#         #convert all ignore_index elements to zero to avoid error in one_hot
#         #note - the choice of value 0 is arbitrary, but it should not matter as these elements will be ignored in the loss calculation
#         target = target * (target!=self.ignore_index) 
#         target = target.view(-1)
#         return one_hot(target, num_classes=num_classes)

#     def _process_preds(self, x: Tensor) -> Tensor:
#         if x.dim() == 1:
#             x = torch.vstack([1 - x, x])
#             x = x.permute(1, 0)
#             return x
#         return x.view(-1, x.shape[-1])

#     def _calc_pt(
#             self, target: Tensor, x: Tensor, mask: Tensor
#             ) -> Tensor:
#         p = target * x
#         p = p.sum(dim=-1)
#         p = p * ~mask
#         return p

#     def forward(self, x: Tensor, target: Tensor) -> Tensor:
#         assert torch.all((x >= 0.0) & (x <= 1.0)), ValueError(
#             'The predictions values should be between 0 and 1, \
#                 make sure to pass the values to sigmoid for binary \
#                 classification or softmax for multi-class classification'
#         )
#         mask = target == self.ignore_index
#         mask = mask.view(-1)
#         x = self._process_preds(x)
#         num_classes = x.shape[-1]
#         target = self._process_target(target, num_classes, mask)
#         weights = self._get_weights(target).to(x.device)
#         pt = self._calc_pt(target, x, mask)
#         focal = 1 - pt
#         nll = -torch.log(self.eps + pt)
#         nll = nll.masked_fill(mask, 0)
#         loss = weights * (focal ** self.gamma) * nll
#         return self._reduce(loss, mask, weights)

#     def _reduce(self, x: Tensor, mask: Tensor, weights: Tensor) -> Tensor:
#         if self.reduction == 'mean':
#             return x.sum() / (~mask * weights).sum()
#         elif self.reduction == 'sum':
#             return x.sum()
#         else:
#             return x

# def sigmoid_focal_loss(
#     inputs: torch.Tensor,
#     targets: torch.Tensor,
#     alpha: float = 0.25,
#     gamma: float = 2,
#     reduction: str = "none",
# ) -> torch.Tensor:
#     """
#     Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

#     Args:
#         inputs (Tensor): A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
#                 classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#         alpha (float): Weighting factor in range (0,1) to balance
#                 positive vs negative examples or -1 for ignore. Default: ``0.25``.
#         gamma (float): Exponent of the modulating factor (1 - p_t) to
#                 balance easy vs hard examples. Default: ``2``.
#         reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
#                 ``'none'``: No reduction will be applied to the output.
#                 ``'mean'``: The output will be averaged.
#                 ``'sum'``: The output will be summed. Default: ``'none'``.
#     Returns:
#         Loss tensor with the reduction option applied.
#     """
#     # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

#     if not torch.jit.is_scripting() and not torch.jit.is_tracing():
#         _log_api_usage_once(sigmoid_focal_loss)
#     p = torch.sigmoid(inputs)
#     ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
#     p_t = p * targets + (1 - p) * (1 - targets)
#     loss = ce_loss * ((1 - p_t) ** gamma)

#     if alpha >= 0:
#         alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
#         loss = alpha_t * loss

#     # Check reduction option and return loss accordingly
#     if reduction == "none":
#         pass
#     elif reduction == "mean":
#         loss = loss.mean()
#     elif reduction == "sum":
#         loss = loss.sum()
#     else:
#         raise ValueError(
#             f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
#         )
#     return loss
