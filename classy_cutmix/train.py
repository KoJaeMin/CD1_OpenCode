# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from classy.models.my_model import MyClassy
from dataset import *
from utils import *


def main():
    model = MyClassy(num_classes=NUMCLASSES)### 모델명만 바꿔주세요!!

    ##### optimizer / learning rate scheduler / criterion #####
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNINGRATE,
                                weight_decay=WEIGHTDECAY)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-7, 
                                            step_size_up=5, max_lr=LEARNINGRATE, 
                                            gamma=0.8, mode='triangular2',cycle_momentum=False)    
    criterion = torch.nn.CrossEntropyLoss()
    ###########################################################

    model = model.cuda() if IsGPU else model
    criterion = criterion.cuda() if IsGPU else criterion

    # Check number of parameters your model
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {pytorch_total_params}")

    # model.load_state_dict(torch.load('model_weight.pth'))

    train_last_top1_acc = 0
    val_last_top1_acc = 0
    train_loss_history = []
    train_correct_history = []
    validation_loss_history = []
    validation_correct_history = []

    for epoch in range(EPOCHS):
        print("\n----- epoch: {}, lr: {} -----".format(
            epoch, optimizer.param_groups[0]["lr"]))
        
        # train for one epoch
        start_time = time.time()
        train_last_top1_acc, train_loss = training(trainloader, epoch, model, optimizer, criterion)
        elapsed_time = time.time() - start_time
        print('==> {:.2f} seconds to train this epoch\n'.format(
            elapsed_time))
        
        # validate for one epoch
        start_time = time.time()
        val_last_top1_acc, val_loss = validating(validationloader,epoch, model, optimizer, criterion)
        elapsed_time = time.time() - start_time
        print('==> {:.2f} seconds to validate this epoch\n'.format(
            elapsed_time))

        # learning rate scheduling
        scheduler.step()

        train_loss_history.append(train_loss)
        train_correct_history.append(train_last_top1_acc)
        validation_loss_history.append(val_loss)
        validation_correct_history.append(val_last_top1_acc)

        # Save model each epoch
        torch.save(model.state_dict(), f'../result/model/model_weight_{MODELNAME}.pth')

    print(f"Train Last Top-1 Accuracy: {train_last_top1_acc}")
    print(f"Validation Last Top-1 Accuracy: {val_last_top1_acc}")
    print(f"Number of parameters: {pytorch_total_params}")
    

    X1 = np.arange(len(train_correct_history))
    train_y1 = np.array(train_correct_history)
    val_y1 = np.array(validation_correct_history)
    X2 = np.arange(len(train_loss_history))
    train_y2 = np.array(train_loss_history)
    val_y2 = np.array(validation_loss_history)


    plt.figure(1,figsize=(12, 8))
    plt.plot(X1,train_y1,label="Train Accuracy")
    plt.plot(X1,val_y1,label="Validation Accuracy")
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Compare Accuracy')
    plt.savefig(f"../result/img/{MODELNAME}_Compare_Accuracy.png")

    plt.figure(2,figsize=(12, 8))
    plt.plot(X2,train_y2,label="Train Loss")
    plt.plot(X2,val_y2,label="Validation Loss")
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Compare Loss')
    plt.savefig(f"../result/img/{MODELNAME}_Compare_Loss.png")
    
    


def training(train_loader, epoch, model, optimizer, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses,
                             top1, top5, prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if i % 3 == 0:
            temp_X_1 = input.clone().detach().cuda() if IsGPU else input.clone().detach()
            temp_X_2 = input.clone().detach().cuda() if IsGPU else input.clone().detach()
            input = input.cuda() if IsGPU else input
            target = target.cuda() if IsGPU else target

            lam1 = LAMBDA1
            lam2 = LAMBDA2
            lam3 = LAMBDA3
            rand_index = torch.randperm(input.size()[0]).cuda() if IsGPU else torch.randperm(input.size()[0]) # batch_size 내의 인덱스가 랜덤하게 셔플됩니다.
            shuffled_y = target[rand_index] # 타겟 레이블을 랜덤하게 셔플합니다.

            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam1)
            input[:,:,bbx1:bbx2, bby1:bby2] = input[shuffled_y,:,bbx1:bbx2, bby1:bby2]### X는 cutmix된 이미지입니다.
            temp_X_2[:,:,:,:]=temp_X_2[shuffled_y,:,:,:]
            lam1 = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            mixed_1 = lam2 * input + (1 - lam2) * temp_X_1
            mixed_1 = mixed_1.cuda() if IsGPU else mixed_1
            mixed_2 = lam3 * mixed_1 + (1 - lam3) * temp_X_2
            mixed_2 = mixed_2.cuda() if IsGPU else mixed_2

            # compute output
            output = model(mixed_2)
            loss = mixup_criterion(criterion,output,target,shuffled_y,lam1*lam2*lam3+(1-lam2)*lam3)

        else:
            input = input.cuda() if IsGPU else input
            target = target.cuda() if IsGPU else target

            output = model(input)
            loss = criterion(output, target)


        # measure accuracy and record loss, accuracy 
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0].item(), input.size(0))
        top5.update(acc5[0].item(), input.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINTFREQ == 0:
            progress.print(i)

    print('=> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    return top1.avg, losses.avg

def validating(val_loader, epoch, model, optimizer, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, data_time, losses,
                             top1, top5, prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda() if IsGPU else input
            target = target.cuda() if IsGPU else target

            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss, accuracy 
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0].item(), input.size(0))
            top5.update(acc5[0].item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % PRINTFREQ == 0:
                progress.print(i)

        print('=> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))
        return top1.avg, losses.avg


if __name__ == "__main__":
    main()