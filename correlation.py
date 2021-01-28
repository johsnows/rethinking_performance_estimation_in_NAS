import numpy as np
import  matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr

def draw_linears_100(x, y, name): #cifar100 top100
    '''
    :param x:  different_epochs []
    :param y:  kend
    :param name:   pdf
    :return:
    '''
    n = len(y)
    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(30, 10))
    for i in range(n):
        j = i //10
        k = i % 10
        axes[j][k].plot(x, y[i])
    plt.savefig(name)

def draw_linears_y(x, y, name):
    '''
    :param x:  different_epochs []
    :param y:  kend
    :param name:   pdf
    :return:
    '''
    n = len(y)
    fig, axes = plt.subplots(nrows=2, ncols=n//2+1, figsize=(30, 10))
    for i in range(n):
        if i//5==0:
            axes[0][i].plot(x, y[i])
        else:
            axes[1][i%5].plot(x, y[i])
    plt.savefig(name)

def draw_linear(x, y, name):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
    axes[0].plot(x, y)
    plt.savefig(name)


def main():
    # str="65.01 66.09 62.26 66.10 65.03 68.02 65.53 64.44 65.19 67.68 66.34 66.86 65.33 66.58 65.94 64.19 67.09 65.42 68.00 68.17 63.29 66.28 67.18 65.83 67.06 66.30 64.86 68.90 60.19 68.48 67.49 64.31 66.80 62.71 67.40 66.41 65.74 63.75 65.01 67.41 63.84 64.89 64.48 65.85 65.32 67.27 66.41 66.88 66.01 64.99 65.40 65.27 68.29 62.04 66.52 66.24 67.21 62.62 63.85 63.48 64.94 68.29 65.83 63.39 67.15 64.80 66.99 66.42 65.28 64.00 66.46 66.07 68.02 65.15 64.35 67.15"
    # print(str.replace(" ", ","))
    # a = [1,2,3]
    # b = [113,12312,12312321]
    # print(kendalltau(a,b)[0])
    gt = [ 96.82, 97.28, 96.98, 97.19, 96.87, 96.76, 97.09, 97.24, 97.26, 96.92, 96.82, 97.09, 96.62, 97.02, 97.18, 97.15, 97.32, 97.05, 97.02, 97.16, 97.34, 97.00, 97.19, 96.51, 96.85, 96.56, 96.89, 97.01, 97.15, 97.10, 96.87, 96.59, 96.93, 97.24, 97.19, 97.14, 96.55, 97.04, 97.08, 96.95, 97.40, 96.97, 97.16, 96.90, 96.78, 96.94, 97.12, 97.13, 97.05, 97.04, 96.55, 96.92, 96.80, 97.08, 97.28, 96.94, 97.08, 97.11, 97.35, 97.15, 97.07, 96.93, 96.40, 97.02, 96.85, 96.79, 96.18, 96.75, 97.02, 97.45, 97.08, 97.41, 96.96, 96.52, 96.87, 96.47, 97.20, 96.31, 96.99, 97.13, 97.16, 97.08, 97.07, 96.62, 97.18, 97.20, 97.11, 96.81, 96.90, 97.21, 96.94, 96.96, 96.81, 96.93, 97.36, 97.01, 97.12, 97.05, 96.85, 97.38, ]
    x = [ 66.88, 67.85, 65.98, 67.81, 66.01, 63.81, 65.71, 64.93, 68.53, 65.10, 63.53, 65.16, 63.33, 66.81, 66.29, 63.09, 64.38, 65.61, 64.93, 65.90, 65.60, 66.12, 68.10, 66.77 \
    ,65.01,66.09,62.26,66.10,65.03,68.02,65.53,64.44,65.19,67.68,66.34,66.86,65.33,66.58,65.94,64.19,67.09,65.42,68.00,68.17,63.29,66.28,67.18,65.83,67.06,66.30,64.86,68.90,60.19,68.48,67.49,64.31,66.80,62.71,67.40,66.41,65.74,63.75,65.01,67.41,63.84,64.89,64.48,65.85,65.32,67.27,66.41,66.88,66.01,64.99,65.40,65.27,68.29,62.04,66.52,66.24,67.21,62.62,63.85,63.48,64.94,68.29,65.83,63.39,67.15,64.80,66.99,66.42,65.28,64.00,66.46,66.07,68.02,65.15,64.35,67.15]
    print(kendalltau(x, gt)[0])
    # return
    model_numbers =100
    epoch_numbers = 200
    top_metric = 100
    epochs =[i for i in range(epoch_numbers)]
    all_kends = []
    all_spearms = []
    best_top=[[0 for i in range(model_numbers)]for i in range(top_metric)]
    for i in range(1, top_metric):  # top 1-9
        kends = []
        spearms = []
        for epoch in range(epoch_numbers):  # epoch
            accs = []
            for j in range(model_numbers):  # model
                acc=np.load("res/cifar100_darts{}epoch{}acc.npy".format(j, epoch))
                # acc=np.load("experiment/test_bpe0_binary/{}/epoch{}acc.npy".format(j, epoch))
                best_top[i][j] = max(best_top[i][j], acc[i])  # get the best top util this epoch for model j using top i
                accs.append(best_top[i][j])
            if epoch==99:
                print(accs)
        # print(gt)
            kend, _ = kendalltau(accs, gt)  #kend on acc and gt at special epoch i and using differnet top x
            spearm, _=spearmanr(accs, gt)
            print('top{} epoch{} kend{}'.format(i, epoch, kend))
            print('top{} epoch{} spearmnar{}'.format(i, epoch, spearm))
            kends.append(kend)
            spearms.append(spearm)
        # for id in range(len(kends)):
        #     if id and id < len(kends) - 1:
        #         kends[id] = (kends[id] + kends[id - 1] + kends[id + 1]) / 3
        all_kends.append(kends)
        draw_linear(epochs, kends, "pdf/cifar100_epoch100_top{}kend_100.pdf".format(i))
        all_spearms.append(spearms)
        draw_linear(epochs, spearms, "pdf/cifar100_epoch100_top{}spearms_100.pdf".format(i))
    draw_linears_y(epochs, all_kends, "pdf/cifar100_epoch100_top1_99kend_100.pdf")
    draw_linears_y(epochs, all_spearms, "pdf/cifar100_epoch100_top1_99spearms_100.pdf")

    # print('kend all', all_spearms)

if __name__ == '__main__':
    main()