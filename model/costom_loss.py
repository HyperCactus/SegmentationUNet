"""
Coppied from https://www.kaggle.com/code/aniketkolte04/sennet-hoa-seg-pytorch-attention-gated-unet
"""
import torch
from torch import Tensor, einsum
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, cast
import numpy as np
from utils import one_hot2hd_dist, probs2one_hot, simplex, one_hot
from global_params import PREDICTION_THRESHOLD


# Intersection over Union (IoU) loss
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, smooth=1):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(prediction)       
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()                            
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + self.smooth)/(union + self.smooth)  # smooth is added to avoid divide by zero error
        
        return 1 - IoU


# a loss function based on the ratio of black pixels to white pixels in the mask\
# this is used to ensure that the model does not predict a completely black image
class BlackToWhiteRatioLoss(nn.Module):
    """Loss based on the ratio of black pixels to white pixels in the mask"""
    
    def __init__(self, weight=None, avtivation_threshold=PREDICTION_THRESHOLD,
                 size_average=True, eps=1e-7):
        """Initialize the loss function"""
        
        super(BlackToWhiteRatioLoss, self).__init__()
        
        self.threshold = avtivation_threshold

    def forward(self, predictions, targets, smooth=1):
        """Forward pass of the loss function"""
        assert predictions.shape[0] == targets.shape[0], "predict & target batch size don't match"
        
        predictions = F.sigmoid(predictions)
        # intersection = (predictions * targets).sum()
        # loss = 1 - intersection
        th_preds = predictions > self.threshold
        sum_ratio = predictions.sum() / targets.sum()
        loss = torch.abs(sum_ratio - 1)
        
        return loss


# The below class is from https://github.com/hubutui/DiceLoss-PyTorch
# It is a loss function based on the Dice score.
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception(f'Unexpected reduction {self.reduction}')

###############################################################################
# following from: https://github.com/sunfan-bvb/BoundaryDoULoss/blob/main/TransUNet/utils.py

class BoundaryDoULoss(nn.Module):
    def __init__(self, n_classes):
        super(BoundaryDoULoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0,1,0], [1,1,1], [0,1,0]])
        padding_out = torch.zeros((target.shape[0], target.shape[-2]+2, target.shape[-1]+2))
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1)).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(), padding=1)
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)  ## We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
        # inputs = torch.softmax(inputs, dim=1)
        inputs = F.sigmoid(inputs)
        target = target.squeeze(1)
        target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes
    
    

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes



###############################################################################




###########################################################################
# Starting here is from: https://github.com/LIVIAETS/boundary-loss
class DiceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

        divided: Tensor = torch.ones_like(intersection) - (2 * intersection + 1e-10) / (union + 1e-10)

        loss = divided.mean()

        return loss


class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        # self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        probs = F.sigmoid(probs)
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, ...].type(torch.float32)
        dc = dist_maps[:, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss

class HausdorffLoss():
    """
    Implementation heavily inspired from https://github.com/JunMa11/SegWithDistMap
    """
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        # self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, preds: Tensor, target: Tensor) -> Tensor:
        probs = F.softmax(preds, dim=1)
        # probs = F.sigmoid(preds)
        print(f'preds shape = {preds.shape}, target shape = {target.shape}')
        print(f'simplex axis 0: {simplex(target, axis=0)}, simplex axis 1: {simplex(target, axis=1)}, simplex axis 2: {simplex(target, axis=2)}')
        print(f'max in target = {torch.max(target)}')
        assert simplex(probs, axis=1)
        assert simplex(target, axis=1)
        assert probs.shape == target.shape

        B, K, *xyz = probs.shape  # type: ignore

        pc = cast(Tensor, probs[:, ...].type(torch.float32))
        tc = cast(Tensor, target[:, ...].type(torch.float32))
        assert pc.shape == tc.shape == (B, 1, *xyz)

        target_dm_npy: np.ndarray = np.stack([one_hot2hd_dist(tc[b].cpu().detach().numpy())
                                              for b in range(B)], axis=0)
        assert target_dm_npy.shape == tc.shape == pc.shape
        tdm: Tensor = torch.tensor(target_dm_npy, device=probs.device, dtype=torch.float32)

        pred_segmentation: Tensor = probs2one_hot(probs).cpu().detach()
        pred_dm_npy: np.nparray = np.stack([one_hot2hd_dist(pred_segmentation[b, ...].numpy())
                                            for b in range(B)], axis=0)
        assert pred_dm_npy.shape == tc.shape == pc.shape
        pdm: Tensor = torch.tensor(pred_dm_npy, device=probs.device, dtype=torch.float32)

        delta = (pc - tc)**2
        dtm = tdm**2 + pdm**2

        multipled = einsum("bkwh,bkwh->bkwh", delta, dtm)

        loss = multipled.mean()

        return loss


class FocalLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        # self.idc: List[int] = kwargs["idc"]
        self.gamma: float = kwargs["gamma"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        probs = F.sigmoid(probs)
        assert simplex(probs) and simplex(target)

        masked_probs: Tensor = probs[:, ...]
        log_p: Tensor = (masked_probs + 1e-10).log()
        mask: Tensor = cast(Tensor, target[:, ...].type(torch.float32))

        w: Tensor = (1 - masked_probs)**self.gamma
        loss = - einsum("bkwh,bkwh,bkwh->", w, mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss

# Ending here
###############################################################################


class EpicLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, eps=1e-7, 
                iou_weight=0.3, dice_weight=0.2, boundary_weight=0.4,
                focal_weight=0.0, cross_entropy_weight=0.1):
        """Initialize the loss function"""
        
        super(EpicLoss, self).__init__()
        
        self.iou_weight = iou_weight
        # self.focal_weight = focal_weight
        self.cross_entropy_weight = cross_entropy_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        # self.hausdorff_weight = hausdorff_weight
        # self.surface_weight = surface_weight
        
        self.iou_loss = IoULoss()
        # self.focal_loss = FocalLoss(gamma=2)
        self.cross_entropy_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = BinaryDiceLoss()
        self.boundary_loss = BoundaryDoULoss(1)
        # self.hausdorff_loss = HausdorffLoss()
        # self.surface_loss = SurfaceLoss()
    
    def forward(self, prediction, targets, smooth=1):
        """Forward pass of the loss function"""
        # comment out if your model contains a sigmoid or equivalent activation layer

        # hausdorff_loss = self.hausdorff_loss(prediction, targets)
        # surface_loss = self.surface_loss(prediction, targets)
        # focal_loss = self.focal_loss(prediction, targets)
        iou_loss = self.iou_loss(prediction, targets)
        cross_entropy_loss = self.cross_entropy_loss(prediction, targets)
        dice_loss = self.dice_loss(prediction, targets)
        boundary_loss = self.boundary_loss(prediction, targets)
        
        loss = (self.iou_weight * iou_loss) + \
                (self.cross_entropy_weight * cross_entropy_loss) + \
                (self.dice_weight * dice_loss) + \
                (self.boundary_weight * boundary_loss)
                # (self.focal_weight * focal_loss) + \
            #    (self.hausdorff_weight * hausdorff_loss) + \
            #    (self.surface_weight * surface_loss)
        
        return loss
class ReduceLROnThreshold(_LRScheduler):
    """Writtern by GPT 4.
    Reduces learning rate when a metric has passed a threshold.
    """
    def __init__(self, optimizer, threshold, factor=0.1, mode='above', verbose=False, **kwargs):
        self.threshold = threshold
        self.factor = factor
        self.verbose = verbose
        self.mode = mode
        super().__init__(optimizer, **kwargs)
        
    def step(self, metric=None):
        # need to do this because the scheduler is called before the metric is calculated
        # in super().__init__(optimizer, **kwargs) in __init__
        if metric is None:
            # If no metric is provided, just pass as the base class does nothing in its step method.
            return
        
        if self.mode == 'above':
            condition = metric > self.threshold
        elif self.mode == 'below':
            condition = metric < self.threshold
        else:
            raise ValueError(f"mode must be 'above' or 'below', got {self.mode}")
        
        if condition:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.factor
            if self.verbose:
                print(f"Reducing learning rate to {param_group['lr']} for metric {metric}")
    
    # def get_lr(self) -> float:
    #     """This method needs to be redevined because the original method in _LRScheduler is not implemented."""
    #     return [group['lr'] for group in self.optimizer.param_groups]

