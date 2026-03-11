import albumentations as A
import numpy as np
from PIL import Image
from torchvision import transforms
import warnings
import torch
import torchvision.transforms.functional as F
import random
warnings.simplefilter("ignore")
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def image_rotate_seg(input_image, label_image, label_image2, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5):
    expend1 = False
    expend2 = False
    # 检查并匹配标签图像的维度
    if len(label_image.shape) == 2:  
        label_image = label_image.unsqueeze(0)  
        expend1 = True
    if len(label_image2.shape) == 2:  
        label_image2 = label_image2.unsqueeze(0)  
        expend2 = True


    assert label_image.shape[0] == 1, "Label image should have a single channel"
    assert label_image2.shape[0] == 1, "Label image2 should have a single channel"

    # 随机水平翻转
    if torch.rand(1) < hflip_prob:
        input_image = TF.hflip(input_image)
        label_image = TF.hflip(label_image)
        label_image2 = TF.hflip(label_image2)

    # 随机垂直翻转
    if torch.rand(1) < vflip_prob:
        input_image = TF.vflip(input_image)
        label_image = TF.vflip(label_image)
        label_image2 = TF.vflip(label_image2)

    # 随机旋转
    if torch.rand(1) < rotate_prob:
        angle = torch.randint(1, 4, (1,)).item() * 90 

        input_image = TF.rotate(input_image, angle)
        label_image = TF.rotate(label_image, angle, interpolation=TF.InterpolationMode.NEAREST)  # 最近邻插值
        label_image2 = TF.rotate(label_image2, angle, interpolation=TF.InterpolationMode.NEAREST)  # 最近邻插值

    if expend1:
      
        label_image = label_image.squeeze(0)  
    if expend2:
        label_image2 = label_image2.squeeze(0)  
    return input_image, label_image, label_image2




def image_rotate_camel(input_image, input_image2, label_image, hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5):
    expend1 = False
    expend2 = False
   
    if len(label_image.shape) == 2: 
        label_image = label_image.unsqueeze(0)  
        expend1 = True
    

    
    assert label_image.shape[0] == 1, "Label image should have a single channel"
    
    # 随机水平翻转
    if torch.rand(1) < hflip_prob:
        input_image = TF.hflip(input_image)
        input_image2 = TF.hflip(input_image2)
        label_image = TF.hflip(label_image)


    # 随机垂直翻转
    if torch.rand(1) < vflip_prob:
        input_image = TF.vflip(input_image)
        input_image2 = TF.vflip(input_image2)
        label_image = TF.vflip(label_image)
      
    # 随机旋转
    if torch.rand(1) < rotate_prob:
        angle = torch.randint(1, 4, (1,)).item() * 90  
        input_image = TF.rotate(input_image, angle)
        input_image2 = TF.rotate(input_image2, angle)
        label_image = TF.rotate(label_image, angle, interpolation=TF.InterpolationMode.NEAREST)  # 最近邻插值
       
    if expend1:
   
        label_image = label_image.squeeze(0)  #
   
    return input_image, input_image2,label_image

def albumentations_transformer(x):
      
    transforms = A.Compose([
        
        A.CLAHE(p=0.1), 
        A.RandomGamma(gamma_limit=(80, 120), p=0.1), 
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2),  #随机更改图像的颜色，饱和度和值。
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),  
        A.AdvancedBlur(blur_limit=(3, 7), p=0.1),
        A.ColorJitter(brightness=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.1),
        A.MotionBlur(blur_limit=7, p=0.1),
        A.MedianBlur(blur_limit=7, p=0.1),
        A.ChannelShuffle(p=0.5)  
    ])

    out = transforms(image=x.copy())["image"]

    return out





def A_transformer(x):

    #img = np.asarray(x)
    img = np.uint8(x)
    img =  albumentations_transformer(img)


    return img

if __name__ == '__main__':

    t = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Resize((input_size, input_size)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
           
    while 1:
        cancer_path ='/traindata/yangmoxuan3/traindata/cervix_train_2048_20_slide/pos_image/B1206636-4_32768_11264.png'
        #img = np.asarray(Image.open(cancer_path).convert('RGB'))
        img = np.uint8(Image.open(cancer_path).convert('RGB'))
        img =  t(A_transformer(img))


        print(img.dtype,img.shape)