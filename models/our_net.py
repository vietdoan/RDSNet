# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# shuffle operation from https://github.com/jaxony/ShuffleNet/blob/master/model.py
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)

    return x


class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        # self.conv = nn.Conv2d(ninput, noutput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        # output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)
    

class RDSBlock (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(chann, chann, 3, 1, 1, groups=chann, bias=False),
            nn.BatchNorm2d(chann),
            nn.ReLU(),
            nn.Conv2d(chann, chann, 3, 1, dilated, dilated, bias=False),
            nn.BatchNorm2d(chann),
            nn.Dropout2d(dropprob)
        )

    def forward(self, input):
        input = channel_shuffle(input, 4)
        output = self.model(input)
        return F.relu(output+input)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 32)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(32, 64))

        for i in range(0, 3):
           self.layers.append(RDSBlock(64, 0.3, 1)) 

        self.layers.append(DownsamplerBlock(64,128))

        self.layers.append(RDSBlock(128, 0.3, 1))
        self.layers.append(RDSBlock(128, 0.3, 2))
        self.layers.append(RDSBlock(128, 0.3, 4))
        self.layers.append(RDSBlock(128, 0.3, 8))
        self.layers.append(RDSBlock(128, 0.3, 16))
        #Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 3, stride=1, padding=1, bias=True)

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)
        return output


class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()


    def forward(self, x):
        h, w = 8 * x.size(2), 8 * x.size(3)
        output = F.upsample(input=x, size=(h, w), mode='bilinear')
        return output

class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = Encoder(num_classes)
        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False):
        output = self.encoder(input)
        return self.decoder(output)
