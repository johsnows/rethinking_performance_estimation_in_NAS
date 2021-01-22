import numpy as np
import  matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
from scipy.stats import kendalltau

def draw_linears_y(x, y, name):
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
    gt = [ 96.82, 97.28, 96.98, 97.19, 96.87, 96.76, 97.09, 97.24, 97.26, 96.92, 96.82, 97.09, 96.62, 97.02, 97.18, 97.15, 97.32, 97.05, 97.02, 97.16, 97.34, 97.00, 97.19, 96.51, ]
    epochs =[i for i in range(580)]
    all_kends = []
    for i in range(1,10):
        kends = []
        for epoch in range(580):
            accs = []
            for j in range(24):
                acc=np.load("res/darts{}epoch{}acc.npy".format(j, epoch))
                accs.append(acc[i])
        # print(accs)
        # print(gt)
            kend, _ = kendalltau(accs, gt)
            print('top{} epoch{} kend{}'.format(i, epoch, kend))
            kends.append(kend)
        # for id in range(len(kends)):
        #     if id and id < len(kends) - 1:
        #         kends[id] = (kends[id] + kends[id - 1] + kends[id + 1]) / 3
        all_kends.append(kends)
        draw_linear(epochs, kends, "top{}kend_all.pdf".format(i))
    draw_linears_y(epochs, all_kends, "top1_9kend_all.pdf")

    print('kend all', kend)

if __name__ == '__main__':
    main()