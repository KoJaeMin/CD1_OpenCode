# -*- coding: utf-8 -*-

### hyper parameter 저장

############### Pytorch CIFAR configuration file ###############
from datetime import datetime
import torch
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

EPOCHS = 150
BATCHSIZE = 256
SEED = 128
WEIGHTDECAY = 1e-2
LEARNINGRATE = 1e-3
PRINTFREQ = 20
MODELNAME = 'RESNET18'
# MODELNUMBER = str(datetime.now()).replace(" ","_").replace("-",".")

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
NUMCLASSES=len(classes)

LAMBDA1 = 0.9
LAMBDA2 = 0.7
LAMBDA3 = 0.7

def fix_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')