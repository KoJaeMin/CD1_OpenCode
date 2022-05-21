# -*- coding: utf-8 -*-

###
### dataset은 CIFAR 10으로 진행
###

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from config import *
from augments import *

### define parameter

batch_size = BATCHSIZE
mean = MEAN
std = STD

### gpu 사용여부
device = device()

### seed 고정
fix_seed()

### transformer

transform = transforms.Compose([
    ### cifar 10 size는 (32,32) 이므로 resize해줄 필요는 없음.
    transforms.ToTensor(),
    # transforms.Resize((256,256)),
    # transforms.Normalize(config.mean, config.std)
])

trainset = torchvision.datasets.CIFAR10(root='./train/', train=True,
                                        download=True, transform=transform)

train_size = int(len(trainset)*0.8)
validation_size = int(len(trainset)*0.2)

### train validation set 분리
trainset, validationset = torch.utils.data.random_split(trainset,[train_size,validation_size])

testset = torchvision.datasets.CIFAR10(root='./test/', train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

validationloader = torch.utils.data.DataLoader(validationset,batch_size=batch_size,
                                          shuffle=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

# for i, data in enumerate(trainloader):
#     inputs, labels = data
#     if i == 1:
#         print(labels)
#         break

### 이미지를 보기
def print_info():
    print(f"your device is {device}")
    print(f"train image : {train_size}\nvalidation image : {validation_size}")
    print(f"(Channel,Height,Width) : {trainset[0][0].shape}")### (Channel,Height,Width) <= tensor화 된 이미지
    print(f"label : {trainset[0][1]}") ### label
    print("\n--------\n")

def show_img():
    img = trainset[0][0] # 0번째 사진
    trans = transforms.ToPILImage()
    img_data = trans(img)
    print(img_data)
    img_data.show()


###
if __name__ == "__main__":
    print_info()
    show_img()