import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from utils import CrossEntropyLoss2d, scores, get_data_path, lovasz_softmax
from datasets.pascal_voc_loader import VOC2011ClassSeg, SBDClassSeg
from datasets.loader import get_loader
from torch.optim import lr_scheduler
from models.our_net import Net

def train(args):
    if (args.dataset == 'pascal'):
        another_loader = VOC2011ClassSeg(root='/home/vietdv', transform=True)
        loader = SBDClassSeg(root='/home/vietdv', transform=True, augment=True)
    else:
        data_path = get_data_path(args.dataset)
        label_scale = False
        if (args.model == 'encoder'):
            label_scale = True
        data_loader = get_loader(args.dataset)
        loader = data_loader(data_path, is_transform=True, augment=True, label_scale=label_scale)
        another_loader = data_loader(data_path, split='val', is_transform=True, label_scale=label_scale)
    
    n_classes = loader.n_classes
    trainloader = data.DataLoader(
        loader, batch_size=args.batch_size)
    
    valloader = data.DataLoader(
        another_loader, batch_size=1)
    # get weight for cross_entropy2d
    weight = loader.weight
    model = Net(n_classes)
    if torch.cuda.is_available():
        model.cuda(0)
        weight = weight.cuda(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate, weight_decay=args.w_decay)
    criterion = CrossEntropyLoss2d(weight, False)
    # alpha = 0.5
    lambda1 = lambda epoch: pow((1-(epoch/args.epochs)),0.9)  ## scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    for epoch in range(args.epochs):
        model.train()
        loss_list = []
        file = open(args.folder + '/{}_{}.txt'.format('hnet', epoch), 'w')
        scheduler.step(epoch)
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda(0))
                labels = Variable(labels.cuda(0))
            else:
                images = Variable(images)
                labels = Variable(labels)
            optimizer.zero_grad()
            outputs = model(images)
            # loss = alpha * criterion(outputs, labels) / len(images) + (1 - alpha) * lovasz_softmax(outputs, labels, ignore=n_classes-1)
            loss = criterion(outputs, labels) / len(images)
            print(loss.data[0])
            loss_list.append(loss.data[0])
            loss.backward()
            optimizer.step()
        
        file.write(str(np.average(loss_list)) + '\n')
        model.eval()
        gts, preds = [], []
        for i, (images, labels) in enumerate(valloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda(0))
                labels = Variable(labels.cuda(0))
            else:
                images = Variable(images)
                labels = Variable(labels)
            outputs = model(images)
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels.data.cpu().numpy()
            for gt_, pred_ in zip(gt, pred):
                gts.append(gt_)
                preds.append(pred_)
        score, class_iou = scores(gts, preds, n_class=n_classes)
        # scheduler.step(score['Mean IoU : \t'])
        for k, v in score.items():
            file.write('{} {}\n'.format(k, v))

        for i in range(n_classes - 1):
            file.write('{} {}\n'.format(i, class_iou[i]))
        torch.save(model.state_dict(), args.folder + "/{}_{}_{}.pkl".format(
            'hnet', args.dataset, epoch))
        file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperprams')
    parser.add_argument('--model_path', nargs='?', type=str, default='cityscapes.pkl', 
                        help='Path to the saved model')
    parser.add_argument('--epochs', nargs='?', type=int, default=150,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--lr_rate', nargs='?', type=float, default=5e-4,
                        help='Learning Rate')
    parser.add_argument('--w_decay', nargs='?', type=float, default=2e-4,
                        help='Weight Decay')
    parser.add_argument('--momentum', nargs='?', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--lr_decay', nargs='?', type=float, default=5e-1,
                        help='Learning Rate Decay')                    
    parser.add_argument('--dataset', nargs='?', type=str, default='cityscapes', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--model', nargs='?', type=str, default='decoder',
                        help='Model to train [\'encoder, decoder\']')
    parser.add_argument('--folder', nargs='?', type=str, default="logs",
                        help='folder to store the model')
    args = parser.parse_args()
    train(args)
