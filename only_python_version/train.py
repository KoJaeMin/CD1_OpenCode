from dataset import *
from densenet import *
from utils import *


def main():
    model = densenet169()

    ##### optimizer / learning rate scheduler / criterion #####
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNINGRATE,
                                momentum=MOMENTUM, weight_decay=WEIGHTDECAY,
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20],
                                                     gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    ###########################################################

    model = model
    criterion = criterion
    # model = model.cuda()
    # criterion = criterion.cuda()

    # Check number of parameters your model
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {pytorch_total_params}")

    #model.load_state_dict(torch.load(SAVEPATH+'model_weight.pth'))

    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation((-8,8)),
        # transforms.RandomCrop(64),
        transforms.ToTensor()
    ])

    last_top1_acc = 0
    for epoch in range(EPOCHS):
        print("\n----- epoch: {}, lr: {} -----".format(
            epoch, optimizer.param_groups[0]["lr"]))

        # train for one epoch
        start_time = time.time()
        last_top1_acc = train(trainloader, epoch, model, optimizer, criterion)
        elapsed_time = time.time() - start_time
        print('==> {:.2f} seconds to train this epoch\n'.format(
            elapsed_time))

        # learning rate scheduling
        scheduler.step()

        # Save model each epoch
        torch.save(model.state_dict(), f'model_weight{MODELNUMBER}.pth')

    print(f"Last Top-1 Accuracy: {last_top1_acc}")
    print(f"Number of parameters: {pytorch_total_params}")



def train(train_loader, epoch, model, optimizer, criterion):
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

        input = input
        target = target
        # input = input.cuda()
        # target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss, accuracy 
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0].item(), input.size(0))
        top5.update(acc5[0].item(), input.size(0))

        # compute gradient and do SGD step
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
    return top1.avg


if __name__ == "__main__":
    main()