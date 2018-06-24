import os
import random
import collections
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from torch.utils import data
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage

p_class = [0.43588215, 0.03066133, 0.01785927, 0.01534676, 0.01778424,
           0.00643387, 0.06047723, 0.02926917, 0.04262554, 0.00232084, 0.00310897,
           0.00905617, 0.00493505, 0.03667296, 0.01675903, 0.01267699, 0.02157302,
           0.03188205, 0.00725101, 0.00947719, 0.02288264, 0.00499989, 0.01217205,
           0.03357802, 0.003332, 0.0019589, 0.01257349, 0.05899915, 0.0282565, 0.00054127, 0.00865327]

class SunRGBLoader(data.DataLoader):
    def __init__(self, root, split='train', is_transform=False, augment=False, label_scale=False):
        self.img_size = [512, 512]
        self.is_transform = is_transform
        self.augment = augment
        self.n_classes = 31
        self.label_scale = label_scale
        self.files = collections.defaultdict(list)
        weight = 1 / (0.02 + np.array(p_class))
        weight[30] = 0
        self.weight = torch.from_numpy(weight).float()
        self.files = []
        dirs = os.listdir(root)
        for dir in dirs:
            img_path = os.path.join(root, dir, split)
            file_list = [file for file in os.listdir(os.path.join(img_path)) if file.endswith('color.png')]
            for file in file_list:
                item = (os.path.join(img_path, file), os.path.join(img_path, file.split('color.png')[0] + 'id_image.png'))
                self.files.append(item)
            break
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img_path, lbl_path = self.files[index]
        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(lbl_path).convert('RGB')
        
        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        
        return img, lbl
    
    def transform(self, img, lbl):
        # img = Resize(self.img_size, Image.BILINEAR)(img)
        # lbl = Resize(self.img_size, Image.NEAREST)(lbl)
        if (self.augment):
            hflip = random.random()
            if (hflip < 0.5):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)
            img = ImageOps.expand(img, border=(transX, transY, 0, 0), fill=0)
            lbl = ImageOps.expand(lbl, border=(transX, transY, 0, 0), fill=0)
            img = img.crop((0, 0, img.size[0]-transX, img.size[1]-transY))
            lbl = lbl.crop((0, 0, lbl.size[0]-transX, lbl.size[1]-transY))

        if (self.label_scale):
            lbl = Resize([self.img_size[0] // 8, self.img_size[1] // 8], Image.NEAREST)(lbl)
        img = ToTensor()(img)
        lbl = np.array(lbl)[:, :, 2]
        lbl_copy = lbl.copy()
        for i in range(0, 31):
            lbl[lbl_copy == i] = (i - 1 + 31) % 31
        lbl = torch.from_numpy(lbl).long()
        return img, lbl
    
    def decode_segmap(self, temp, plot=False):
        b = np.zeros_like(temp)
        for l in range(0, self.n_classes):
            b[temp == l] = (l + 1) % self.n_classes

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 2] = b
        rgb = np.array(rgb, dtype=np.uint8)
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

if __name__ == '__main__':
    local_path = '/home/vietdoan/Downloads/image_minos'
    dst = SunRGBLoader(local_path, is_transform=True, label_scale=False)
    print(dst.weight)
    trainloader = data.DataLoader(dst, batch_size=1)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        img = torchvision.utils.make_grid(imgs).numpy()
        img = np.transpose(img, (1, 2, 0))
        img *= 255
        img = img.astype(np.uint8)
        plt.imshow(img)
        plt.show()
        dst.decode_segmap(labels.numpy()[0], plot=True)
            


        
