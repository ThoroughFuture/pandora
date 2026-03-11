import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
class DiceLossV1(nn.Module):
    """
    different from normal dice loss
    for image with no positive mask, only calculate dice for background
    for positive image, only calculate positive pixel
    """
    def __init__(self, smooth=1e-7, weight=None):
        """
        Diceloss for segmentation
        :param smooth: smooth value for fraction
        :param weight: class weight
        """
        super(DiceLossV1, self).__init__()
        self.smooth = smooth
        if weight is not None:
            weight = torch.tensor(weight).float()
        self.weight = weight

    def forward(self, gt, logits, reduction='mean'):
        """
        code from https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
        Note that PyTorch optimizers minimize a loss. In this case,
        we would like to maximize the dice loss so we return the negated dice loss.
        :param gt: a tensor of shape [B, 1 , H, W]
        :param logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logit of the model.
        :param reduction:
        :return:
            dice_loss: dice loss
        """
        num_classes = logits.shape[1]
        gt = (gt.long())
        if num_classes == 1:
            gt_1_hot = torch.eye(num_classes + 1).cuda()[gt.squeeze(1)]  # B,H,W,2
            gt_1_hot = gt_1_hot.permute(0, 3, 1, 2).float()  # B,2,H,W
            gt_1_hot = torch.cat([gt_1_hot[:, 0:1, :, :],
                                  gt_1_hot[:, 1:2, :, :]],
                                 dim=1)
            pos_prob = torch.sigmoid(logits)  # B,1,H,W
            neg_prob = 1 - pos_prob  # B,1,H,W
            probas = torch.cat([neg_prob, pos_prob], dim=1)  # B,2,H,W
        else:
            gt_1_hot = torch.eye(num_classes)[gt.squeeze(1)]  # B,H,W,cls
            gt_1_hot = gt_1_hot.permute(0, 3, 1, 2).float()  # B,cls,H,W
            probas = F.softmax(logits, dim=1)  # B,cls,H,W

        gt_1_hot = gt_1_hot.type(logits.type())

        # whether gt have pos pixel
        is_pos = gt.gt(0).sum(dim=(1, 2, 3)).gt(0)  # B

        dims = tuple(range(2, logits.ndimension()))
        intersection = torch.sum(probas * gt_1_hot, dims)  # B,cls
        cardinality = torch.sum(probas + gt_1_hot, dims)  # B,cls
        dice_coff = 2 * intersection / (cardinality + self.smooth)  # B,cls
        if reduction == 'none':
            dice_coff = torch.where(is_pos, dice_coff[:, 1:].mean(1), dice_coff[:, 0])
            return dice_coff
        dice_loss = 1 - dice_coff
        if self.weight is not None:
            weight = self.weight.cuda()  # cls
            dice_loss = weight * dice_loss  # B, cls

        dice_loss = (torch.where(is_pos, dice_loss[:, 1:].mean(1), dice_loss[:, 0])).mean()

        return dice_loss

class BceLoss(nn.Module):
    def __init__(self, weight=None):
        """
        Computes the weighted binary cross-entropy loss.
        :param weight: a scalar representing the weight attributed
                        to the positive class. This is especially useful for
                        an imbalanced dataset.
        """
        super(BceLoss, self).__init__()
        if weight is not None:
            if isinstance(weight, (list, tuple)) and len(weight) == 2:
                weight = weight[1]
            weight = torch.tensor(weight).float()
        self.weight = weight

    def forward(self, gt, logits, reduction='mean'):
        """
        Computes the weighted binary cross-entropy loss.
        :param gt: a tensor of shape [B, 1, H, W].
        :param logits: a tensor of shape [B, 1, H, W]. Corresponds to
        :param reduction: same as F.binary_cross_entropy_with_logits
        :return: bce_loss: the weighted binary cross-entropy loss.
        """
        num_class = logits.shape[1]
        assert num_class <= 2, "For class num larger than 2, use CrossEntropy instead"
        if self.weight is not None:
            weight = self.weight.cuda()
        else:
            weight = None
        if num_class == 2:
            gt = torch.eye(2)[gt.squeeze(1).long()].cuda()  # B, H, W, 2
            gt = gt.permute(0, 3, 1, 2).contiguous()  # B, 2, H, W

        bce_loss = F.binary_cross_entropy_with_logits(
            logits.float(),
            gt.float(),
            reduction=reduction,
            pos_weight=weight
        )

        return bce_loss


class BceWithLogDiceLoss(nn.Module):
    """
    bce loss - ln(dice) for segmentation
    :param smooth: smooth value for fraction
    :param class_weight: class weight for both bce loss and dice loss
    :param bce_weight: result is bce_loss * bce_weight + dice_loss
    """
    def __init__(self, smooth=1e-7, class_weight=None, bce_weight=2.):
        super(BceWithLogDiceLoss, self).__init__()

        self.bce_weight = (torch.tensor([bce_weight]) / (bce_weight + 1)).float()
        self.dice_weight = (torch.tensor([1.]) / (bce_weight + 1)).float()
        self.dice_loss = DiceLossV1(smooth=smooth, weight=class_weight)
        self.bce_loss = BceLoss(weight=class_weight)

    def forward(self, gt, logits):
        bce_loss = self.bce_loss(gt, logits, reduction='none').cuda()  # b, c, w, h
        dice_coff = self.dice_loss(gt, logits,  reduction='none').cuda()  # b

        bce_loss = bce_loss.mean(dim=(1, 2, 3))  # b
        dice_loss = - torch.log(dice_coff)  # ln dice loss = - ln(dice)

        loss = bce_loss* 3  + dice_loss * 1
     
        loss = loss.mean()

        return loss
    

    





import torch.nn as nn
import torch


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)

        #y_pred = (y_pred > 0.6).type(torch.float32)  # 转化为整形分类  # 阈值

        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)

        #iou = (intersection + self.smooth) / ((y_pred + y_true).sum() - intersection + self.smooth)

        return 1. - dsc #- iou


import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU




class DiceLoss_with_classification(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=None):
        super(DiceLoss_with_classification, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # input: (B, C, H, W), target: (B, H, W)
        # input needs to be reshaped to (B, C, H * W) and then transposed to (B, H * W, C)
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, input.size(1))
        target = target.view(-1)

        # Create a mask to ignore the specified index
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            input = input[valid_mask]
            target = target[valid_mask]

        # Convert target to one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=input.size(1)).float()

        # Compute Dice coefficient
        intersection = (input * target_one_hot).sum(dim=0)
        union = (input + target_one_hot).sum(dim=0)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # Average Dice score across all classes, excluding ignored class
        dice = dice[1:] if self.ignore_index == 0 else dice  # Skip the ignored class index
        dice_loss = 1 - dice.mean()

        return dice_loss




class Ce_with_Dice_loss(nn.Module):
    def __init__(self, ce_weight=3.0, dice_weight=1.0, ignore_index=0):
        super(Ce_with_Dice_loss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index)
        self.dice_loss = DiceLoss_with_classification(ignore_index=ignore_index)

    def forward(self, output, target):
        # output: (B, C, H, W), target: (B, H, W)
        ce_loss = self.ce_loss(output, target)

        # Compute Dice Loss
        dice_loss = self.dice_loss(output, target)

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
