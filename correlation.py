import numpy as np
from scipy.stats import kendalltau

def main():
    # a = [1,2,3]
    # b = [113,12312,12312321]
    # print(kendalltau(a,b)[0])
    gt = [ 96.82, 97.28, 96.98, 97.19, 96.87, 96.76, 97.09, 97.24, 97.26, 96.92, 96.82, 97.09, 96.62, 97.02, 97.18, 97.15, 97.32, 97.05, 97.02, 97.16, 97.34, 97.00, 97.19, 96.51, ]
    kends = []
    for i in range(1,10):
        accs = []
        for j in range(24):
            acc=np.load("res/darts{}epoch20acc.npy".format(j))
            accs.append(acc[i])
        # print(accs)
        # print(gt)
        kend, _ = kendalltau(gt, accs)
        print('top{} kend{}'.format(i, kend))
        kends.append(kend)
    print('kend all', kend)

if __name__ == '__main__':
    main()