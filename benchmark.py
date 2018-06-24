import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
import scipy.misc as m
from torch.autograd import Variable

from datasets.loader import get_loader
from utils import get_data_path, scores
from models.our_net import Net
from models.Enet import Enet
def speed(model):
    model.eval()
    t0 = time.time()
    input = torch.rand(1,3,512, 1024).cuda()
    input = Variable(input, volatile = True)
    t2 = time.time()

    model(input)
    torch.cuda.synchronize()
    t3 = time.time()
    
    return (t3 - t2)

if __name__ == '__main__':
    enet = Enet(20)
    our = Net(20)
    speed(enet)
    enet_time = 0.0
    our_time = 0.0
    for i in range(100):
        enet_time += speed(enet)
        our_time += speed(our)

    print('%10s : %f' % ('enet', enet_time / 100))
    print('%10s : %f' % ('our', our_time / 100))
