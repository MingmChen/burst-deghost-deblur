import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy import cluster
import matplotlib.pyplot as plt
import numpy as np


class Cl:

	def __init__(self, input_np, x):
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

		self.col = ['red', 'blue', 'gold', 'green', 'pink', 'orange', 'darkblue', 'darkred', 'purple', 'yellow','darkorange', 'cyan', 'lightgreen', 'yellowgreen', 'tomato', 'deepskyblue']
		for i in range(0, self.cluster_n):
			plt.scatter(value[labels == i, 0], value[labels == i, 1], s=10, marker='o', color=self.col[i])

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
		print(self.output)
		plt.show()

	def test(self, x):
		cost = np.zeros(self.cluster_n, dtype=np.float)
		for i in range(0, self.input_n):
			label = self.label[i]
			dist = pow(pow(abs(self.output[label] - self.value[i]), x).sum(), 1. / x)
			cost[label] += dist
		cost /= self.sum
		print(cost)


def main():
	a = np.random.rand(1000, 2)
	test = Cl(a, 16)
	test.run()
	test.test(2)


if __name__ == '__main__':
	print("start")
	main()
