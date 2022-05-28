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
IsGPU = device==torch.device('cuda')
print(IsGPU)

### seed 고정
fix_seed()

### transformer
if(IsGPU):
    transform = transforms.Compose([
            ### cifar 10 size는 (32,32) 이므로 resize해줄 필요는 없음.
            transforms.ToTensor(),
            # transforms.Resize((256,256)),### 사진을 보기 위해서 임의로 resize했습니다.
            transforms.Normalize(mean, std)
])
else:
    transform = transforms.Compose([
            ### cifar 10 size는 (32,32) 이므로 resize해줄 필요는 없음.
            transforms.Resize((256,256)),### 사진을 보기 위해서 임의로 resize했습니다.
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
])


trainset = torchvision.datasets.CIFAR10(root='../train/', train=True,
                                        download=True, transform=transform)

train_size = int(len(trainset)*0.8)
validation_size = int(len(trainset)*0.2)

### train validation set 분리
trainset, validationset = torch.utils.data.random_split(trainset,[train_size,validation_size])

testset = torchvision.datasets.CIFAR10(root='../test/', train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

validationloader = torch.utils.data.DataLoader(validationset,batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False,  num_workers=2)

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
    print(f"label : {classes[trainset[0][1]]}") ### label
    print("\n--------\n")



def save_img():
    X,y = next(iter(trainloader))
    temp_X_1 = X.clone().detach().cuda() if IsGPU else X.clone().detach()
    temp_X_2 = X.clone().detach().cuda() if IsGPU else X.clone().detach()
    X = X.cuda() if IsGPU else X
    y = y.cuda() if IsGPU else y

    lam1 = LAMBDA1
    lam2 = LAMBDA2
    lam3 = LAMBDA3
    rand_index = torch.randperm(X.size()[0].cuda() if IsGPU else X.size()[0]) # batch_size 내의 인덱스가 랜덤하게 셔플됩니다.
    shuffled_y = y[rand_index] # 타겟 레이블을 랜덤하게 셔플합니다.

    bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), lam1)
    X[:,:,bbx1:bbx2, bby1:bby2] = X[shuffled_y,:,bbx1:bbx2, bby1:bby2]### X는 cutmix된 이미지입니다.
    temp_X_2[:,:,:,:]=temp_X_2[shuffled_y,:,:,:]
    lam1 = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (X.size()[-1] * X.size()[-2]))
    print(f"lam1 : {lam1}, lam2 : {lam2}, lam3 : {lam3}")
    mixed_1 = lam2 * X + (1 - lam2) * temp_X_1
    mixed_2 = lam3 * mixed_1 + (1 - lam3) * temp_X_2

    # total_X = torch.cat([temp_X,X], dim=0)
    trans = transforms.ToPILImage()
    trans(mixed_2[0].cpu()).save('../result/img/cutmix_original_image1_original_image2.png','png')# mixup + 원본2 (cutmix하고 mixup된 이미지 0.7만큼, 원본2 0.3만큼)
    trans(mixed_1[0].cpu()).save('../result/img/cutmix_original_image1.png','png')# cutmix + 원본1 (cutmix된 이미지 0.7만큼, 원본1 0.3만큼)
    trans(X[0].cpu()).save('../result/img/cutmix.png','png')# cutmix된 이미지(원본1 lam1만큼 원본2 1-lam1만큼)
    trans(temp_X_1[0].cpu()).save('../result/img/original_image1.png','png')# 원본 1
    trans(temp_X_2[0].cpu()).save('../result/img/original_image2.png','png')# 원본 2


###
if __name__ == "__main__":
    print_info()
    save_img()