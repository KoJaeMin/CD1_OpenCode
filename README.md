# CD1_OpenCode
2022년 1학기 컴퓨터공학과 캡스톤디자인1 오픈코드팀

## Multiple Data Augmentation(CutMix+Mixup)
<img src="./result/img/cutmix_cat_dog.png">

## Experimental results
|Model + Method|Train Top1 Accuracy|Validation Top1 Accuracy|Test Top1 Accuracy|
|---|---|---|---|
|ResNet18|100|65.41|65.46|
|ResNet18+CutMix|98.62|73.57|73.15|
|ResNet18+Mixup|97.38|73.53|73.32|
|ResNet18+CutMixup|99.98|67.11|67.45|
|ResNet18+Multiple Data Augmentation(CutMix+Mixup)|92.80|75.12|74.95|
|ResNet34|100|66.30|65.50|
|ResNet34+CutMix|98.14|73.03|74.54|
|ResNet34+Mixup|97.15|72.92|72.96|
|ResNet34+CutMixup|99.95|68.04|67.78|
|ResNet34+Multiple Data Augmentation(CutMix+Mixup)|92.39|73.65|73.95|
|DenseNet169|100|71.59|71.45|
|DenseNet169+CutMix|98.33|77.57|77.72|
|DenseNet169+Mixup|76.19|76.45|77.07|
|DenseNet169+CutMixup|98.89|76.33|76.87|
|DenseNet169+Multiple Data Augmentation(CutMix+Mixup)|73.74|77.14|77.95|

## Environment
- Python 3.7.13
- Pytorch 1.11.0 
- CUDA 11.3
- Ubuntu 18.04.5 LTS
- Tesla P100