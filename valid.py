import torch
import genotypes as gt
from scipy.stats import kendalltau
import torch.nn as nn
import numpy as np
from models.augment_cnn import AugmentCNN
from config import AugmentConfig
import utils
config = AugmentConfig()

device = torch.device("cuda")
def validate(valid_loader, model, criterion, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    len_val_loader = len(valid_loader)
    top = [i for i in range(10)]

    def val_iter(X, y):
        N = X.size(0)

        logits, _ = model(X)
        loss = criterion(logits, y)

        prec1 = utils.super_accuracy(logits, y)
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)

        if step % config.print_freq == 0 or step == len_val_loader - 1:
            print(
                "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%})".format(
                    epoch + 1, config.epochs, step, len_val_loader - 1, losses=losses,
                    top1=top1))

    model.eval()

    with torch.no_grad():
        if config.data_loader_type == 'DALI':
            for step, data in enumerate(valid_loader):
                X = data[0]["data"].cuda(async=True)
                y = data[0]["label"].squeeze().long().cuda(async=True)
                val_iter(X, y)
            valid_loader.reset()
        else:
            for step, (X, y) in enumerate(valid_loader):
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                if config.fp16:
                    X = X.type(torch.float16)
                val_iter(X, y)
    print("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))
    return top1.avg


def main():
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Important for stablizing training
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data, valid_data = utils.get_data(
        config.dataset, config.data_path, config.image_size, config.cutout_length, validation=True)

    # print(input_size, input_channels, n_classes)
    criterion = nn.CrossEntropyLoss().to(device)
    use_aux = config.aux_weight > 0.

    # change size of input image
    input_size = config.image_size

    model = AugmentCNN(input_size, input_channels, config.init_channels, n_classes, config.layers,
                       use_aux, (config.genotype))
    # model size
    mb_params = utils.param_size(model)
    print("Model size = {:.3f} MB".format(mb_params))
    ckpt = "experiment/darts_10/{}/checkpoint10.pth.tar"
    top1s = []
    file_ = open(config.file)
    lines = file_.readlines()
    for i, line in enumerate(lines):
    # for i in range(24):
        model = AugmentCNN(input_size, input_channels, config.init_channels, n_classes, config.layers,
                           use_aux, eval(line))
        model = nn.DataParallel(model, device_ids=config.gpus).to(device)
        model =torch.load(ckpt.format(i))
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        valid_loader = torch.utils.data.DataLoader(valid_data,
                                                   batch_size=config.batch_size,
                                                   shuffle=False,
                                                   num_workers=config.workers,
                                                   pin_memory=True)
        top1 = validate(valid_loader, model, criterion, 10)
        top1s.append(top1)
    top1s = np.array(top1s)
    np.save("binar_acc.npy", top1s)
    gt = [ 96.82, 97.28, 96.98, 97.19, 96.87, 96.76, 97.09, 97.24, 97.26, 96.92, 96.82, 97.09, 96.62, 97.02, 97.18, 97.15, 97.32, 97.05, 97.02, 97.16, 97.34, 97.00, 97.19, 96.51, ]
    kend , _=  kendalltau(gt, top1s)
    print('kend', kend)





if __name__ == '__main__':
    main()