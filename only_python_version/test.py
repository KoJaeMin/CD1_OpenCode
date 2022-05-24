# -*- coding: utf-8 -*-

# import argparse
from random import sample
from dataset import *
from densenet import *
from utils import *
import pandas as pd


def eval():
    model1 = densenet169(num_classes=NUMCLASSES)
    model2 = densenet169(num_classes=NUMCLASSES)
    model3 = densenet169(num_classes=NUMCLASSES)
    model1 = model1.cuda() if IsGPU else model1
    model2 = model2.cuda() if IsGPU else model2
    model3 = model3.cuda() if IsGPU else model3
    model1.load_state_dict(torch.load(f'../result/model/model_weight_{MODELNAME}.pth'))
    model2.load_state_dict(torch.load(f'../result/model/model_weight_{MODELNAME}_cutmix.pth'))
    model3.load_state_dict(torch.load(f'../result/model/model_weight_{MODELNAME}_mixup.pth'))
    
    model1.eval()
    model2.eval()
    model3.eval()

    total = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    # 학습 중이 아니므로, 출력에 대한 변화도를 계산할 필요가 없습니다
    print('Evaluate...')
    with torch.no_grad():
        for i, (input, target) in enumerate(testloader):
            input = input.cuda() if IsGPU else input
            target = target.cuda() if IsGPU else target
            # 신경망에 이미지를 통과시켜 출력을 계산합니다
            output1 = model1(input)
            output2 = model2(input)
            output3 = model3(input)
            _, predictions1 = torch.max(output1, 1)
            _, predictions2 = torch.max(output2, 1)
            _, predictions3 = torch.max(output3, 1)
            # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택하겠습니다
            total += target.size(0)
            correct1 += (predictions1 == target).sum().item()
            correct2 += (predictions2 == target).sum().item()
            correct3 += (predictions3 == target).sum().item()
        

    print(f'Accuracy of cutmix and mixup model on the 10000 test images: {100 * correct1 // total} %')
    print(f'Accuracy of cutmix model on the 10000 test images: {100 * correct2 // total} %')
    print(f'Accuracy of mixup model on the 10000 test images: {100 * correct3 // total} %')


if __name__ == "__main__":
    eval()