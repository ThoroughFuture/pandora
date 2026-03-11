from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import random
from camel.dataload.Augment import A_transformer,image_rotate_seg


def random_crop_and_resize_v2(img, label, output_size=1024, crop_mode="random", scale_size=0.5):

    assert img.size == label.size, f"Image and label size mismatch: {img.size} vs {label.size}"
    width, height = img.size

    if crop_mode == "center":
        
        if width < output_size or height < output_size:
            
            raise ValueError(
                f"Center crop requires image size >= output_size ({output_size}), "
                f"but got image size {img.size}."
            )
        
        left = (width - output_size) // 2
        top = (height - output_size) // 2
        right = left + output_size
        bottom = top + output_size
        
        img_crop = img.crop((left, top, right, bottom))
        label_crop = label.crop((left, top, right, bottom))
        
     
        return img_crop, label_crop
    
    elif crop_mode == "random":
        scale = random.uniform(scale_size, 1.0)
        
     
        crop_size = int(round(min(height, width) * scale)) 
        crop_size = min(crop_size, height, width) 
        if height == crop_size:
             top = 0
        else:
             top = random.randint(0, height - crop_size)
             
        if width == crop_size:
             left = 0
        else:
             left = random.randint(0, width - crop_size)

        right = left + crop_size
        bottom = top + crop_size
        

        img_crop = img.crop((left, top, right, bottom))
        label_crop = label.crop((left, top, right, bottom))

        img_resized = img_crop.resize(
            (output_size, output_size), 
            resample=Image.BILINEAR
        )
        
     
        label_resized = label_crop.resize(
            (output_size, output_size), 
            resample=Image.NEAREST 
        )

        return img_resized, label_resized

    else:
        raise ValueError(f"Unsupported crop_mode: {crop_mode}. Use 'random' or 'center'.")
    



class data_load(Dataset):

    def __init__(self, pos_dir, label_dir):
        self.t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.t2 = transforms.Compose([
            transforms.ToTensor(),
            ])

        

        self.image_path = np.array(pd.read_table(pos_dir, sep='\t', encoding='utf_8_sig', header=None)).tolist()
        self.label_path = np.array(pd.read_table(label_dir, sep='\t', encoding='utf_8_sig', header=None)).tolist()
        self.image_path.pop(-1)
        self.label_path.pop(-1)
       
        cc = list(zip(self.image_path, self.label_path))
        random.shuffle(cc)
        self.image_path[:], self.label_path[:] = zip(*cc)


    def __getitem__(self, item):
        img = self.image_path[item][0]
        label_path = self.label_path[item][0]
        img = Image.open(img).convert('RGB')
        label = Image.open(label_path).convert('RGB')

        img, label = random_crop_and_resize_v2(img, label, output_size=2048, crop_mode="random", scale_size=1.0)
        if random.random() >= 0.2:
            img = self.t(A_transformer(img))
        else:
            img = self.t(img)
        
        
        train_label = self.t2(label)[0, :].unsqueeze(0) 
        real_label = self.t2(label)[2, :].unsqueeze(0)  
        

        train_label = (train_label == 1).to(torch.long)  


        return img, train_label, real_label 
        

    def __len__(self):
        return len(self.image_path)
    

class data_load_old(Dataset):

    def __init__(self, pos_dir, label_dir):
        self.t = transforms.Compose([
            transforms.ToTensor(),
           
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.t2 = transforms.Compose([
            transforms.ToTensor(),
           
            ])

        

        self.pos_path = np.array(pd.read_table(pos_dir, sep='\t', encoding='utf_8_sig', header=None)).tolist()
        self.neg_path = np.array(pd.read_table(label_dir, sep='\t', encoding='utf_8_sig', header=None)).tolist()
        self.pos_path.pop(-1)
        self.neg_path.pop(-1)
        len_neg = len(self.neg_path)
        len_pos = len(self.pos_path)

        assert len_neg==len_pos

        self.image_path = []
        self.label_path = []
        
        for i in range(0, len_pos):
            self.image_path.append(self.pos_path[i])
            self.label_path.append(self.neg_path[i])
     
            if 'pos_image' in self.pos_path[i][0]:
                self.image_path.append(self.pos_path[i])
                self.label_path.append(self.neg_path[i])

        assert len(self.image_path) == len(self.label_path)
     
        cc = list(zip(self.image_path, self.label_path))
        random.shuffle(cc)
        self.image_path[:], self.label_path[:] = zip(*cc)


    def __getitem__(self, item):
        img = self.image_path[item][0]
        label_path = self.label_path[item][0]

        
        if random.random() >= 0.4:
            img = self.t(A_transformer(Image.open(img).convert('RGB')))
        else:
            if random.random()>=0.5:
                img = self.t(Image.open(img).convert('RGB'))
            else:
                img = Image.open(img).convert('RGB')
                img = np.array(img)[:, :, ::-1]           
                img = self.t(Image.fromarray(np.uint8(img)))

        label = Image.open(label_path).convert('RGB')
        train_label = self.t2(label)[0, :].unsqueeze(0) 
        real_label = self.t2(label)[2, :].unsqueeze(0) 
        

        train_label = (train_label == 1).to(torch.long)  


        return img, train_label, real_label 
        

    def __len__(self):
        return len(self.image_path)
    


class data_load_cancer_classification(Dataset):

    def __init__(self, pos_dir, label_dir):
        self.t = transforms.Compose([
            transforms.ToTensor(),

            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])


        self.pos_path = np.array(pd.read_table(pos_dir, sep='\t', encoding='utf_8_sig', header=None)).tolist()
        self.neg_path = np.array(pd.read_table(label_dir, sep='\t', encoding='utf_8_sig', header=None)).tolist()
        self.pos_path.pop(-1)
        self.neg_path.pop(-1)
        len_neg = len(self.neg_path)
        len_pos = len(self.pos_path)

        self.image_path = []
        self.label_path = []
        for i in range(0, len_pos):
            self.image_path.append(self.pos_path[i])

        for j in range(0, len_neg):
            self.label_path.append(self.neg_path[j])


        cc = list(zip(self.image_path, self.label_path))
        random.shuffle(cc)
        self.image_path[:], self.label_path[:] = zip(*cc)


    def __getitem__(self, item):
        img = self.image_path[item][0]
        label_path = self.label_path[item][0]

        if random.random() >= 0.8:
            img = self.t(A_transformer(Image.open(img).convert('RGB')))
        else:
            if random.random()>=0.5:
                img = self.t(Image.open(img).convert('RGB'))
            else:
                img = Image.open(img).convert('RGB')
                img = np.array(img)[:, :, ::-1]
                img = self.t(Image.fromarray(np.uint8(img)))

        label = Image.open(label_path).convert('RGB')
    
        label = torch.from_numpy(np.array(label)[:, :, -1]).long()
        
        train_label = torch.abs(label-1)
        real_label = label 
        
        img,train_label,real_label = image_rotate_seg(img,train_label,real_label)
        
        return img, train_label, real_label 
        

    def __len__(self):
        return len(self.image_path)
    

class data_load_val(Dataset):

    def __init__(self, pos_dir, label_dir):
        self.t = transforms.Compose([
            transforms.ToTensor(),

            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.t2 = transforms.Compose([
            transforms.ToTensor(),

            ])

        self.RGB = True

        self.pos_path = np.array(pd.read_table(pos_dir, sep='\t', encoding='utf_8_sig', header=None)).tolist()
        self.neg_path = np.array(pd.read_table(label_dir, sep='\t', encoding='utf_8_sig', header=None)).tolist()
        self.pos_path.pop(-1)
        self.neg_path.pop(-1)
        len_neg = len(self.neg_path)
        len_pos = len(self.pos_path)

        self.image_path = []
        self.label_path = []
        for i in range(0, len_pos):
            self.image_path.append(self.pos_path[i])

        for j in range(0, len_neg):
            self.label_path.append(self.neg_path[j])

        cc = list(zip(self.image_path, self.label_path))
        random.shuffle(cc)
        self.image_path[:], self.label_path[:] = zip(*cc)


    def __getitem__(self, item):
        img = self.image_path[item][0]
        label_path = self.label_path[item][0]

        if self.RGB:
            img = self.t(Image.open(img).convert('RGB'))
        else:
            img = Image.open(img).convert('RGB')
            img = np.array(img)[:, :, ::-1]          
            img = self.t(Image.fromarray(np.uint8(img)))
        
        label = self.t2(Image.open(label_path).convert('RGB'))[2, :].unsqueeze(0)
        
        return img, label
        

    def __len__(self):
        return len(self.image_path)







class data_load_cancer_classification_val(Dataset):

    def __init__(self, pos_dir, label_dir):
        self.t = transforms.Compose([
            transforms.ToTensor(),

            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])


        self.pos_path = np.array(pd.read_table(pos_dir, sep='\t', encoding='utf_8_sig', header=None)).tolist()
        self.neg_path = np.array(pd.read_table(label_dir, sep='\t', encoding='utf_8_sig', header=None)).tolist()
        self.pos_path.pop(-1)
        self.neg_path.pop(-1)
        len_neg = len(self.neg_path)
        len_pos = len(self.pos_path)

        self.image_path = []
        self.label_path = []
        for i in range(0, len_pos):
            self.image_path.append(self.pos_path[i])

        for j in range(0, len_neg):
            self.label_path.append(self.neg_path[j])


        cc = list(zip(self.image_path, self.label_path))
        random.shuffle(cc)
        self.image_path[:], self.label_path[:] = zip(*cc)


    def __getitem__(self, item):
        img = self.image_path[item][0]
        label_path = self.label_path[item][0]

        img = self.t(Image.open(img).convert('RGB'))

        label = Image.open(label_path).convert('RGB')
    
        label = torch.from_numpy(np.array(label)[:, :, -1]).long()
        
        train_label = torch.abs(label-1)
        real_label = label  
        
        img,train_label,real_label = image_rotate_seg(img,train_label,real_label)
        
        return img, train_label, real_label 
        

    def __len__(self):
        return len(self.image_path)