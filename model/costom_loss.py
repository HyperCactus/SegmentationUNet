"""
Coppied from https://www.kaggle.com/code/aniketkolte04/sennet-hoa-seg-pytorch-attention-gated-unet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler



class FocalLoss(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=0, size_average=None, ignore_index=-100,
                 reduce=None, balance_param=1.0):
        super(FocalLoss, self).__init__(size_average)
        self.gamma = gamma
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)

        logpt = - F.binary_cross_entropy_with_logits(input, target)
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss

class IslandLoss(nn.Module):
    """
    A loss function that penalizes the model for predicting more or fewer 
    islands of pixels then are present in the mask.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, prediction, targets, smooth=1):
        
        # an island is a group of white pixels, (pixels with value of 1) that are connected
        # islands are separated by black pixels (pixels with value of 0)
        # we will need to use a clever algorithm to find the number of islands in the mask
        pass
        

def calc_islands(mask):
    """
    Calculates the number of islands in the mask
    """
    pass


# Intersection over Union (IoU) loss
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, eps=1e-7):
        super(IoULoss, self).__init__()

    def forward(self, prediction, targets, smooth=1):
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
        
        IoU = (intersection + smooth)/(union + smooth)  # smooth is added to avoid divide by zero error
        
        return 1 - IoU


# a loss function based on the ratio of black pixels to white pixels in the mask\
# this is used to ensure that the model does not predict a completely black image
class BlackToWhiteRatioLoss(nn.Module):
    """Loss based on the ratio of black pixels to white pixels in the mask"""
    
    def __init__(self, weight=None, avtivation_threshold=0.5,
                 size_average=True, eps=1e-7):
        """Initialize the loss function"""
        
        super(BlackToWhiteRatioLoss, self).__init__()
        
        self.threshold = avtivation_threshold

    def forward(self, predictions, targets, smooth=1):
        """Forward pass of the loss function"""
        assert predictions.shape[0] == targets.shape[0], "predict & target batch size don't match"
        inputs = predictions.contiguous().view(predictions.shape[0], -1)
        targets = targets.contiguous().view(targets.shape[0], -1)
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(predictions)       
        
        # flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        
        # calculate the ratio of black pixels to white pixels
        # black_pixels_pred = (inputs < self.threshold).sum()
        # white_pixels_pred = (inputs >= self.threshold).sum()
        # ratio_pred = black_pixels_pred / white_pixels_pred
        
        # black_pixels_mask = (targets == 0).sum()
        # white_pixels_mask = (targets == 1).sum()
        # ratio_mask = black_pixels_mask / white_pixels_mask
        
        # loss = torch.abs(ratio_pred - ratio_mask)\
        
        # print('Pass 1')
        # predicted = torch.Tensor((inputs < self.threshold).sum().sum())
        # print('Pass 2')
        # target = torch.Tensor(targets.sum().sum())
        # print('Pass 3')
        # # loss = nn.MSELoss(predicted, target)
        # loss = F.mse_loss(predicted, target)
        # print(f'loss: {loss}, type: {type(loss)}')
        
        # calculate the number of ones (white pixels) in the mask
        ones_in_mask = torch.sum(targets)
        
        # calculate the number of elements in the prediction that are greater than the threshold
        # ones_in_pred = (inputs > self.threshold).sum()
        mask = inputs > self.threshold
        ones = torch.ones_like(inputs)
        ones_in_pred = torch.sum(ones[mask])
        
        loss = torch.abs(1 - (ones_in_pred / ones_in_mask))
        
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
            raise Exception('Unexpected reduction {}'.format(self.reduction))


# class GPUMemoryLoss(nn.Module):
#     def __init__(self, weight=1, size_average=True, eps=1e-7):
#         super(GPUMemoryLoss, self).__init__()
#         self.weight = weight

#     def forward(self, prediction=None, targets=None):
#         t = torch.cuda.get_device_properties(0).total_memory / 1024**3
#         a = torch.cuda.memory_allocated(0) / 1024**3
#         loss = (a / t) * self.weight
        



class EpicLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, eps=1e-7, 
                 ratio_weight=0.0, iou_weight=0.5, 
                 focal_weight=0.0, cross_entropy_weight=0.5):
        """Initialize the loss function"""
        
        super(EpicLoss, self).__init__()
        
        self.ratio_weight = ratio_weight
        self.iou_weight = iou_weight
        self.focal_weight = focal_weight
        self.cross_entropy_weight = cross_entropy_weight
        
        self.pixel_ratio_loss = BlackToWhiteRatioLoss()
        self.iou_loss = IoULoss()
        self.focal_loss = FocalLoss()
        self.cross_entropy_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = BinaryDiceLoss()
    
    def forward(self, prediction, targets, smooth=1):
        """Forward pass of the loss function"""
        # comment out if your model contains a sigmoid or equivalent activation layer

        # print(f'PREDS SHAPE: {prediction.shape}, TARGETS SHAPE: {targets.shape}')
                
        pixel_ratio_loss = torch.abs(prediction - targets)
        
        iou_loss = self.iou_loss(prediction, targets)
        
        focal_loss = self.focal_loss(prediction, targets)
        
        cross_entropy_loss = self.cross_entropy_loss(prediction, targets)
        
        loss = (self.iou_weight * iou_loss) + \
               (self.focal_weight * focal_loss) + \
               (self.cross_entropy_weight * cross_entropy_loss)# + \
            #    (self.ratio_weight * pixel_ratio_loss)
        
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

