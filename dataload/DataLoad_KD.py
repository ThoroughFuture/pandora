
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import random
from camel.dataload.Augment import A_transformer



class data_load_kd(Dataset):
    def __init__(self, no_label_dir):
        super(data_load_kd, self).__init__()
        self.t = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop((256,256)),
            
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

       
        self.no_label_path = np.array(pd.read_table(no_label_dir, sep='\t', encoding='utf_8_sig', header=None)).tolist()
        self.no_label_path.pop(-1)
        random.shuffle(self.no_label_path)

        len_nolabel = len(self.no_label_path)

        self.kd_image_path = []

        for i in range(0, len_nolabel):
            self.kd_image_path.append(self.no_label_path[i])

    def __getitem__(self, item):

        kd_img = self.kd_image_path[item][0]

        if random.random()>=0.8:

            kd_image = self.t(A_transformer(Image.open(kd_img).convert('RGB')))
        else:

            if random.random()>=0.5:
                
                kd_image = self.t(Image.open(kd_img).convert('RGB'))
            else:
              
                kd_image = Image.open(kd_img).convert('RGB')
                kd_image = np.array(kd_image)[:, :, ::-1]            
                kd_image = self.t(Image.fromarray(np.uint8(kd_image)))

        return kd_image

        
    def __len__(self):
        return len(self.kd_image_path)
    

class data_load_kd_multi_teacher(Dataset):
    def __init__(self, no_label_dir):
        super(data_load_kd_multi_teacher, self).__init__()
        
        self.no_label_path = np.array(pd.read_table(no_label_dir, sep='\t', encoding='utf_8_sig', header=None)).tolist()
        self.no_label_path.pop(-1)
        random.shuffle(self.no_label_path)

        len_nolabel = len(self.no_label_path)

        self.kd_image_path = []

        for i in range(0, len_nolabel):
            self.kd_image_path.append(self.no_label_path[i])

    def __getitem__(self, item):

        kd_img = self.kd_image_path[item][0]

        try:
            kd_img = torch.load(kd_img)
        except:
            print(kd_img)

        image = kd_img['image']
        f1 = kd_img['f1']
        f2 = kd_img['f2']
        f3 = kd_img['f3']
        f4 = kd_img['f4']
        
        return image,f1,f2,f3,f4

        
    def __len__(self):
        return len(self.kd_image_path)
    