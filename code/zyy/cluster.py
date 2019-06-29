import pandas as pd
import os
from sklearn.cluster import AgglomerativeClustering
from scipy import cluster
import matplotlib.pyplot as plt
import numpy as np
import random


class Cl:

    def __init__(self, input_np, x, s):
        self.name = s
        self.output: np.array = None
        self.sum = np.zeros(x)
        self.cluster_n = x

        value = input_np
        self.value = value
        self.input_n = value.shape[0]

        model = AgglomerativeClustering(n_clusters=x, affinity='euclidean', linkage='ward')
        model.fit(value)
        labels = model.labels_
        self.label = labels

        for i in range(0, self.input_n):
            self.sum[labels[i]] += 1

        self.col = ['red', 'blue', 'gold', 'green', 'pink', 'orange', 'darkblue', 'darkred', 'purple', 'yellow',
                    'darkorange', 'cyan', 'lightgreen', 'yellowgreen', 'tomato', 'deepskyblue']
        for i in range(0, self.cluster_n):
            plt.scatter(value[labels == i, 0], value[labels == i, 1], s=5, marker='o', color=self.col[i])

    def center(self, x):
        xx, yy = 0.0, 0.0
        for i in range(0, self.input_n):
            if (self.label[i] == x):
                xx += self.value[i][0]
                yy += self.value[i][1]
        return xx / self.sum[x], yy / self.sum[x]

    def run(self):
        temp = []
        for i in range(0, self.cluster_n):
            x, y = self.center(i)
            temp.append([x, y])
            plt.scatter(x, y, s=20, marker='o', color='black')
        self.output = np.array(temp)
        #print(self.output)
        plt.savefig('./result_' + self.name + '.jpg')
        plt.close()

    def test(self, x):
        cost = np.zeros(self.cluster_n, dtype=np.float)
        for i in range(0, self.input_n):
            label = self.label[i]
            dist = pow(pow(abs(self.output[label] - self.value[i]), x).sum(), 1. / x)
            cost[label] += dist
        cost /= self.sum
        print(cost)


def main():
    list = os.listdir("./")
    list = [f for f in list if f.split('.')[-1] == "npy"]
    for f in list:
        a = np.load("./" + f)
        a = a.transpose(0, 2, 3, 1)
        N = a.shape[3]
        a1 = a[:, :, :, 0:N // 2 - 1]
        a2 = a[:, :, :, N // 2:-1]
        a1 = a1.reshape(-1)
        a2 = a2.reshape(-1)
        a = [a1, a2]
        b = np.array(a)
        b = b.transpose(1, 0)

        plt.scatter(b[:,0], b[:,1], s=1, marker='o', color='black')
        print(f + 'origin plant end!')
        plt.savefig('./' + f.split('.')[0] + '.jpg')
        plt.close()
        random.shuffle(b)

        test = Cl(b[0:10000][:], 9, f.split('.')[0])
        test.run()


if __name__ == '__main__':
    print("start")
    main()
