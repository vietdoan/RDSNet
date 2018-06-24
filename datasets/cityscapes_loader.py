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

ignore_label = 19
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32, 0, 0, 0]


class CityscapesLoader(data.Dataset):
    def __init__(self, root, split='train', is_transform=False, img_size=None, augment=False, label_scale=False):
        self.img_size = [512, 1024]
        self.is_transform = is_transform
        self.augment = augment
        self.n_classes = 20
        self.label_scale = label_scale
        self.files = collections.defaultdict(list)
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        weight = torch.ones(self.n_classes)
        if (label_scale):
            weight[0] = 2.3653597831726
            weight[1] = 4.4237880706787
            weight[2] = 2.9691488742828
            weight[3] = 5.3442072868347
            weight[4] = 5.2983593940735
            weight[5] = 5.2275490760803
            weight[6] = 5.4394111633301
            weight[7] = 5.3659925460815
            weight[8] = 3.4170460700989
            weight[9] = 5.2414722442627
            weight[10] = 4.7376127243042
            weight[11] = 5.2286224365234
            weight[12] = 5.455126285553
            weight[13] = 4.3019247055054
            weight[14] = 5.4264230728149
            weight[15] = 5.4331531524658
            weight[16] = 5.433765411377
            weight[17] = 5.4631009101868
            weight[18] = 5.3947434425354
        else:
            weight[0] = 2.8149201869965
            weight[1] = 6.9850029945374
            weight[2] = 3.7890393733978
            weight[3] = 9.9428062438965
            weight[4] = 9.7702074050903
            weight[5] = 9.5110931396484
            weight[6] = 10.311357498169
            weight[7] = 10.026463508606
            weight[8] = 4.6323022842407
            weight[9] = 9.5608062744141
            weight[10] = 7.8698215484619
            weight[11] = 9.5168733596802
            weight[12] = 10.373730659485
            weight[13] = 6.6616044044495
            weight[14] = 10.260489463806
            weight[15] = 10.287888526917
            weight[16] = 10.289801597595
            weight[17] = 10.405355453491
            weight[18] = 10.138095855713
        weight[19] = 0
        self.weight = weight
        self.files = []
        img_path = os.path.join(root, 'leftImg8bit', split)
        mask_path = os.path.join(root, 'gtFine', split)
        categories = os.listdir(img_path)
        for c in categories:
            c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
            for it in c_items:
                item = (os.path.join(img_path, c, it + '_leftImg8bit.png'), os.path.join(mask_path, c, it + '_gtFine_labelIds.png'))
                self.files.append(item)
        if ('train' in split):
            random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path, lbl_path = self.files[index]
        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(lbl_path).convert('P')

        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        
        return img, lbl

    def transform(self, img, lbl):
        img = Resize(self.img_size, Image.BILINEAR)(img)
        lbl = Resize(self.img_size, Image.NEAREST)(lbl)
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
        # img = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        lbl = np.array(lbl)
        lbl_copy = lbl.copy()
        for k, v in self.id_to_trainid.items():
            lbl[lbl_copy == k] = v
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def decode_segmap(self, temp, plot=False):
        label_colours = np.array(palette).reshape(-1, 3)
        r = np.zeros_like(temp)
        g = np.zeros_like(temp)
        b = np.zeros_like(temp)
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        rgb = np.array(rgb, dtype=np.uint8)
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

if __name__ == '__main__':
    local_path = '/home/vietdoan/cityscapes'
    dst = CityscapesLoader(local_path, is_transform=True, label_scale=True)
    trainloader = data.DataLoader(dst, batch_size=1)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img *= np.array([0.229, 0.224, 0.225])
            img += np.array([0.485, 0.456, 0.406])
            img *= 255
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.show()
            dst.decode_segmap(labels.numpy()[0], plot=True)
