import torch    
import numpy as np
import pandas as pd
import torch.nn.functional as f
import torch.nn.functional as F


def channel_pooling(input_tensor, output_channels=12, pooling_mode='max'):


    B, C, W, H = input_tensor.size()

  
    num_groups = (C + output_channels - 1) // output_channels 

  
    padding_size = num_groups * output_channels - C
    padded_input = torch.nn.functional.pad(input_tensor, (0, 0, 0, 0, 0, padding_size), "constant", 0)

    reshaped_input = padded_input.view(B, num_groups, output_channels, W, H)


    if pooling_mode == 'max':
        pooled_output, _ = torch.max(reshaped_input, dim=1)  
    elif pooling_mode == 'mean':
        pooled_output = torch.mean(reshaped_input, dim=1)  
    return pooled_output

def slice_image_step(img, patch_size=256,step=128):


    B, C, W, H = img.shape
    imglist = []
    for b in range(B):
        for x in range(0,W-patch_size+1,step):
            for y in range(0,H-patch_size+1,step):
               imglist.append(img[b, :, x:x+patch_size, y:y+patch_size])
    img = torch.stack(imglist, dim=0)
    return img


def slice_image(img,patch_size=256):


    img = torch.split(img, split_size_or_sections=patch_size, dim=2)
    img = torch.cat(img, dim=3)
    img = torch.split(img, split_size_or_sections=patch_size, dim=3)
    img = torch.stack(img, dim=1)
    B, N, C, W, H = img.shape
    img = img.contiguous().view(-1, C, W, H)
    return img


def merge_image_step(img, heatmap,patch_size=256,step=128):

    B, C, W, H = heatmap.shape

    num = 0
    for b in range(B):
        for x in range(0, W - patch_size + 1, step):
            for y in range(0, H - patch_size + 1, step):
                heatmap[b, :, x:x + patch_size, y:y + patch_size] += img[num] / 4
               
                num += 1
    heatmap[:, :, 0:step, :] = heatmap[:, :, 0:step, :] * 2
    heatmap[:, :, -step:, :] = heatmap[:, :, -step:, :] * 2
    heatmap[:, :, :, 0:step] = heatmap[:, :, :, 0:step] * 2
    heatmap[:, :, :, -step:] = heatmap[:, :, :, -step:] * 2
    return heatmap


def merge_image(image, input_size=256, orginal_size=2048):

    B = int(image.shape[0] / (orginal_size / input_size) ** 2)
    out = torch.zeros(size=(B, image.shape[1], orginal_size, orginal_size), dtype=image.dtype).to(device=image.device)
    for i in range(0, B):
        img = image[i*int((orginal_size / input_size) ** 2 ): (i+1) * int((orginal_size / input_size) ** 2), :]
        img = torch.split(img, split_size_or_sections=1, dim=0)
        img = torch.cat(img, dim=-1)
        img = torch.split(img, split_size_or_sections=orginal_size, dim=-1)
        img = torch.cat(img, dim=-2)
        img = img.view(1, -1, orginal_size, orginal_size)
        out[i, :] = img
    return out





def Dice(y_pred, y_true):
    
    smooth = 1e-9
    y_pred = y_pred.contiguous().view(-1)
    y_true = y_true.contiguous().view(-1)

    intersection = (y_pred * y_true).sum()
    dsc = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

    iou = (intersection + smooth) / ((y_pred + y_true).sum() - intersection + smooth)

    return dsc ,iou 



def multiclassification_dice_with_iou(y_true, y_pred, num_classes=6, ignore_index=0):


    y_true = torch.tensor(y_true, dtype=torch.long)
    y_pred = torch.tensor(y_pred, dtype=torch.long)

    dice_scores = {i: 0 for i in range(1, num_classes)}
    iou_scores = {i: 0 for i in range(1, num_classes)}
    total_correct = 0
    total_elements = 0


    valid_mask = (y_true != ignore_index)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]


    for cls in range(1, num_classes):
       
        tp = (y_true == cls) & (y_pred == cls)
        fp = (y_true != cls) & (y_pred == cls)
        fn = (y_true == cls) & (y_pred != cls)

     
        if tp.sum() > 0:
            dice_scores[cls] = 2 * tp.float().sum() / (2 * tp.float().sum() + fp.float().sum() + fn.float().sum())
            iou_scores[cls] = tp.float().sum() / (tp.float().sum() + fp.float().sum() + fn.float().sum())
          
        else:
            dice_scores[cls] = 0.0
            iou_scores[cls] = 0.0

        correct = tp.sum().item()
        total_correct += correct
        total_elements += (y_true == cls).sum().item()

       
    mean_dice = sum(dice_scores.values()) / len(dice_scores) if dice_scores else 0.0
    mean_iou = sum(iou_scores.values()) / len(iou_scores) if iou_scores else 0.0

    if total_elements > 0:
        accuracy = total_correct / total_elements
    else:
        accuracy = 0.0

    return dice_scores, iou_scores, torch.tensor(accuracy), torch.tensor(mean_dice), torch.tensor(mean_iou)





import torch
from sklearn.metrics import jaccard_score


def dice_coefficient(pred, target, num_classes, ignore_index=0):
    smooth = 1e-5

    assert pred.dim() == 3 and target.dim() == 3, "pred and target should both be 3D tensors."
    
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    pred_one_hot = F.one_hot(pred.long(), num_classes).view(-1, num_classes)
    target_one_hot = F.one_hot(target.long(), num_classes).view(-1, num_classes)

  
    dice_scores = []

  
    for class_idx in range(num_classes):
        if class_idx == ignore_index:
            continue  

        pred_class = pred_one_hot[:, class_idx]
        target_class = target_one_hot[:, class_idx]

  
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()

     
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())

    return dice_scores

def iou_coefficient(pred, target, num_classes, ignore_index=0):
    smooth = 1e-5

  
    assert pred.dim() == 3 and target.dim() == 3, "pred and target should both be 3D tensors."
    
  
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    pred_one_hot = F.one_hot(pred.long(), num_classes).view(-1, num_classes)
    target_one_hot = F.one_hot(target.long(), num_classes).view(-1, num_classes)

    iou_scores = []

   
    for class_idx in range(num_classes):
        if class_idx == ignore_index:
            continue  

        pred_class = pred_one_hot[:, class_idx]
        target_class = target_one_hot[:, class_idx]

   
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum() - intersection

   
        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou.item())

    return iou_scores