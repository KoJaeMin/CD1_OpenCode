# -*- coding: utf-8 -*-

# import argparse
from random import sample
from dataset import *
from densenet import *
from utils import *
import pandas as pd


def eval():
    model = densenet169(num_classes=NUMCLASSES)
    model = model.cuda() if IsGPU else model
    model.load_state_dict(torch.load(f'../result/model/model_weight_{MODELNAME}.pth'))
    model.eval()
    correct = 0
    total = 0
    # 학습 중이 아니므로, 출력에 대한 변화도를 계산할 필요가 없습니다
    print('Evaluate...')
    with torch.no_grad():
        for i, (input, target) in enumerate(testloader):
            input = input.cuda() if IsGPU else input
            target = target.cuda() if IsGPU else target
            # 신경망에 이미지를 통과시켜 출력을 계산합니다
            output = model(input)
            _, predictions = torch.max(output, 1)
            # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택하겠습니다
            total += target.size(0)
            correct += (predictions == target).sum().item()

            if(i%PRINTFREQ==0):
                print(correct/total)
        

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


if __name__ == "__main__":
    eval()