import torch
from scipy.stats import kendalltau, spearmanr
import torch.nn as nn
import numpy as np
from models.augment_cnn import AugmentCNN
from config import AugmentConfig
import utils
import genotypes as geno
dict = None
config = AugmentConfig(dict)

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

    ckpt = "experiment/bpe_darts_10/{}/checkpoint{}.pth.tar"
    epochs = 10
    file_ = open(config.file)
    lines = file_.readlines()
    kends = []
    spearms = []
    for epoch in range(epochs):
        top1s = []
        for i, line in enumerate(lines):
            # if i>23:
            #     break
        # for i in range(24):
            model = AugmentCNN(input_size, input_channels, config.init_channels, n_classes, config.layers,
                                   use_aux, geno.from_str(line))
            # model size
            mb_params = utils.param_size(model)
            print("Model size = {:.3f} MB".format(mb_params))
            model = nn.DataParallel(model, device_ids=config.gpus).to(device)
            model =torch.load(ckpt.format(i, epoch))
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
        np.save("res/binary_bpe1_epoch{}_acc.npy".format(epoch), top1s)
        # gt = [ 96.82, 97.28, 96.98, 97.19, 96.87, 96.76, 97.09, 97.24, 97.26, 96.92, 96.82, 97.09, 96.62, 97.02, 97.18, 97.15, 97.32, 97.05, 97.02, 97.16, 97.34, 97.00, 97.19, 96.51, ]
        gt = [96.82, 97.28, 96.98, 97.19, 96.87, 96.76, 97.09, 97.24, 97.26, 96.92, 96.82, 97.09, 96.62, 97.02, 97.18,
              97.15, 97.32, 97.05, 97.02, 97.16, 97.34, 97.00, 97.19, 96.51, 96.85, 96.56, 96.89, 97.01, 97.15, 97.10,
              96.87, 96.59, 96.93, 97.24, 97.19, 97.14, 96.55, 97.04, 97.08, 96.95, 97.40, 96.97, 97.16, 96.90, 96.78,
              96.94, 97.12, 97.13, 97.05, 97.04, 96.55, 96.92, 96.80, 97.08, 97.28, 96.94, 97.08, 97.11, 97.35, 97.15,
              97.07, 96.93, 96.40, 97.02, 96.85, 96.79, 96.18, 96.75, 97.02, 97.45, 97.08, 97.41, 96.96, 96.52, 96.87,
              96.47, 97.20, 96.31, 96.99, 97.13, 97.16, 97.08, 97.07, 96.62, 97.18, 97.20, 97.11, 96.81, 96.90, 97.21,
              96.94, 96.96, 96.81, 96.93, 97.36, 97.01, 97.12, 97.05, 96.85, 97.38, ]
        kend , _=  kendalltau(gt, top1s)
        spearm , _=  spearmanr(gt, top1s)
        print('kend', kend)
        print('spearm', spearm)
        kends.append(kend)
        spearms.append(spearm)
    from correlation import draw_linear
    epochs = [i for i in range(epochs)]
    draw_linear(epochs, kends, 'bpe1_binary_kend.pdf')
    draw_linear(epochs, spearms, 'bpe1_binary_spearm.pdf')





if __name__ == '__main__':
    main()