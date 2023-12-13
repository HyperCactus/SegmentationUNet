"""
Coppied from https://www.kaggle.com/code/aniketkolte04/sennet-hoa-seg-pytorch-attention-gated-unet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F



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


class EpicLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, eps=1e-7, 
                 ratio_weight=0.5, iou_weight=0.7, 
                 focal_weight=0.0, cross_entropy_weight=0.3):
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
                
        pixel_ratio_loss = torch.abs(prediction - targets)
        
        iou_loss = self.iou_loss(prediction, targets)
        
        focal_loss = self.focal_loss(prediction, targets)
        
        cross_entropy_loss = self.cross_entropy_loss(prediction, targets)
        
        loss = (self.iou_weight * iou_loss) + \
               (self.focal_weight * focal_loss) + \
               (self.cross_entropy_weight * cross_entropy_loss)# + \
            #    (self.ratio_weight * pixel_ratio_loss)
        
        return loss