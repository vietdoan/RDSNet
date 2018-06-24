import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import time

from torch.autograd import Variable
from torch.utils import data
from datasets.loader import get_loader
from utils import get_data_path, scores
import scipy.misc as m
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage
from models.our_net import Net
def validate(args):

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, split=args.split, is_transform=True)
    n_classes = loader.n_classes

    # Setup Model
    model = Net(n_classes)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    if torch.cuda.is_available():
        model.cuda(0)
    img_path = os.path.join(data_path, 'leftImg8bit', 'val')
    categories = os.listdir(img_path)
    trainid_to_id = {19: 0, 0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 17, 6: 19, 7: 20, 8: 21, 9: 22, 10: 23, 11: 24,
                     12: 25, 13: 26, 14: 27, 15: 28, 16: 31, 17: 32, 18: 33}
    for c in categories:
        c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            item = (os.path.join(img_path, c, it + '_leftImg8bit.png'), it + '_gtFine_labelIds')
            img = Image.open(item[0]).convert('RGB')
            img = Resize([512, 1024], Image.BILINEAR)(img)
            img = ToTensor()(img)
            # img = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            img = img.view(1, img.size(0), img.size(1), img.size(2))
            if torch.cuda.is_available():
                img = Variable(img.cuda(0))
            else:
                img = Variable(img)
            t1 = time.time()
            outputs = model(img)
            t2 = time.time()
            print(t2 - t1)
            #continue
            pred = outputs.data.max(1)[1].cpu().numpy()
            pred = pred.reshape(512, 1024).astype(np.uint8)
            pred_copy = pred.copy()
            for k, v in trainid_to_id.items():
                pred[pred_copy == k] = v
            pred = Image.fromarray(pred.astype(np.uint8), 'P')
            pred = Resize([1024, 2048], Image.NEAREST)(pred)
            pred.save('./images_val/{}.png'.format(item[1]))
            # m.imsave('./images/{}.png'.format(item[1]), pred)    
            # break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_path', nargs='?', type=str, default='cityscapes.pkl', 
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='cityscapes', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='val', 
                        help='Split of dataset to test on')
    args = parser.parse_args()
    validate(args)
