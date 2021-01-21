import numpy as np
from scipy.stats import kendalltau

def main():
    # a = [1,2,3]
    # b = [113,12312,12312321]
    # print(kendalltau(a,b)[0])
    gt = []
    kends = []
    for i in range(1,10):
        accs = []
        for j in range(24):
            acc=np.load("darts{}epoch10acc.npy".format(j))
            accs.append(acc)
        kend, _ = kendalltau(gt, accs)
        print('top{} kend{}'.format(i+1, kend))
        kends.append(kend)
    print('kend all', kend)

if __name__ == '__main__':
    main()