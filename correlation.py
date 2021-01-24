import numpy as np
import  matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
from scipy.stats import kendalltau

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
    # a = [1,2,3]
    # b = [113,12312,12312321]
    # print(kendalltau(a,b)[0])
    gt = [ 96.82, 97.28, 96.98, 97.19, 96.87, 96.76, 97.09, 97.24, 97.26, 96.92, 96.82, 97.09, 96.62, 97.02, 97.18, 97.15, 97.32, 97.05, 97.02, 97.16, 97.34, 97.00, 97.19, 96.51, 96.85, 96.56, 96.89, 97.01, 97.15, 97.10, 96.87, 96.59, 96.93, 97.24, 97.19, 97.14, 96.55, 97.04, 97.08, 96.95, 97.40, 96.97, 97.16, 96.90, 96.78, 96.94, 97.12, 97.13, 97.05, 97.04, 96.55, 96.92, 96.80, 97.08, 97.28, 96.94, 97.08, 97.11, 97.35, 97.15, 97.07, 96.93, 96.40, 97.02, 96.85, 96.79, 96.18, 96.75, 97.02, 97.45, 97.08, 97.41, 96.96, 96.52, 96.87, 96.47, 97.20, 96.31, 96.99, 97.13, 97.16, 97.08, 97.07, 96.62, 97.18, 97.20, 97.11, 96.81, 96.90, 97.21, 96.94, 96.96, 96.81, 96.93, 97.36, 97.01, 97.12, 97.05, 96.85, 97.38, ]
    # x = [ 66.88, 67.85, 65.98, 67.81, 66.01, 63.81, 65.71, 64.93, 68.53, 65.10, 63.53, 65.16, 63.33, 66.81, 66.29, 63.09, 64.38, 65.61, 64.93, 65.90, 65.60, 66.12, 68.10, 66.77, ]
    # print(kendalltau(x, gt)[0])
    # return
    model_numbers =100
    epoch_numbers =10
    epochs =[i for i in range(epoch_numbers)]
    all_kends = []
    best_top=[[0 for i in range(model_numbers)]for i in range(10)]
    for i in range(1,10):  # top 1-9
        kends = []
        for epoch in range(epoch_numbers):  # epoch
            accs = []
            for j in range(model_numbers):  # model
                acc=np.load("res/bpe1_darts{}epoch{}acc.npy".format(j, epoch))
                best_top[i][j] = max(best_top[i][j], acc[i])  # get the best top util this epoch for model j using top i
                accs.append(best_top[i][j])
            if epoch==9:
                print(accs)
        # print(gt)
            kend, _ = kendalltau(accs, gt)  #kend on acc and gt at special epoch i and using differnet top x
            print('top{} epoch{} kend{}'.format(i, epoch, kend))
            kends.append(kend)
        # for id in range(len(kends)):
        #     if id and id < len(kends) - 1:
        #         kends[id] = (kends[id] + kends[id - 1] + kends[id + 1]) / 3
        all_kends.append(kends)
        draw_linear(epochs, kends, "bpe1_top{}kend_100.pdf".format(i))
    draw_linears_y(epochs, all_kends, "bpe1_top1_9kend_100.pdf")

    print('kend all', kend)

if __name__ == '__main__':
    main()