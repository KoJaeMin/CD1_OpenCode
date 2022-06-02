import train
import train_original as original
import train_mixup as mixup
import train_cutmix as cutmix
import test
from config import *

if __name__ == '__main__':
    print("--main--")
    main_train_last_top1_acc,main_val_last_top1_acc,main_pytorch_total_params,main_train_loss_history,main_train_correct_history,main_validation_loss_history,main_validation_correct_history=train.main()
    print("--ori--")
    original_train_last_top1_acc,original_val_last_top1_acc,original_pytorch_total_params,original_train_loss_history,original_train_correct_history,original_validation_loss_history,original_validation_correct_history=original.main()
    print("--mixup--")
    mixup_train_last_top1_acc,mixup_val_last_top1_acc,mixup_pytorch_total_params,mixup_train_loss_history,mixup_train_correct_history,mixup_validation_loss_history,mixup_validation_correct_history=mixup.main()
    print("--cutmix--")
    cutmix_train_last_top1_acc,cutmix_val_last_top1_acc,cutmix_pytorch_total_params,cutmix_train_loss_history,cutmix_train_correct_history,cutmix_validation_loss_history,cutmix_validation_correct_history=cutmix.main()

    print ("{:<30} {:<30} {:<30} {:<30}".format('Model','Train Last Top1 Accuracy','Validation Last Top1 Accuracy','Total Parameters'))
    print ("{:<30} {:<30} {:<30} {:<30}".format(f'{MODELNAME}',original_train_last_top1_acc,original_val_last_top1_acc,original_pytorch_total_params))
    print ("{:<30} {:<30} {:<30} {:<30}".format(f'{MODELNAME}+cutmix',cutmix_train_last_top1_acc,cutmix_val_last_top1_acc,cutmix_pytorch_total_params))
    print ("{:<30} {:<30} {:<30} {:<30}".format(f'{MODELNAME}+mixup',mixup_train_last_top1_acc,mixup_val_last_top1_acc,mixup_pytorch_total_params))
    print ("{:<30} {:<30} {:<30} {:<30}".format(f'{MODELNAME}+cutmix+mixup',main_train_last_top1_acc,main_val_last_top1_acc,main_pytorch_total_params))

    test.eval()

    with open("../result/densenet.txt", "w") as f:
        f.write("{:<30} {:<30} {:<30} {:<30}".format("main train correct","main train loss","main validation correct","main validation loss"))
        for i in range(EPOCHS):
            f.write("{:<30} {:<30} {:<30} {:<30}".format(main_train_correct_history[i],main_train_loss_history[i],main_validation_correct_history[i],main_validation_loss_history[i]))
        
        f.write("{:<30} {:<30} {:<30} {:<30}".format("original train correct","original train loss","original validation correct","original validation loss"))
        for i in range(EPOCHS):
            f.write("{:<30} {:<30} {:<30} {:<30}".format(original_train_correct_history[i],original_train_loss_history[i],original_validation_correct_history[i],original_validation_loss_history[i]))


        f.write("{:<30} {:<30} {:<30} {:<30}".format('mixup train correct','mixup train loss','mixup validation correct','mixup validation loss'))
        for i in range(EPOCHS):
            f.write("{:<30} {:<30} {:<30} {:<30}".format(mixup_train_correct_history[i],mixup_train_loss_history[i],mixup_validation_correct_history[i],mixup_validation_loss_history[i]))

        f.write("{:<30} {:<30} {:<30} {:<30}".format('cutmix train correct','cutmix train loss','cutmix validation correct','cutmix validation loss'))
        for i in range(EPOCHS):
            f.write("{:<30} {:<30} {:<30} {:<30}".format(cutmix_train_correct_history[i],cutmix_train_loss_history[i],cutmix_validation_correct_history[i],cutmix_validation_loss_history[i]))