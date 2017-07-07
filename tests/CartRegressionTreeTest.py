import numpy as np
import MyStatisticsLib.CartRegressionTreeUtil as cdtu
import MyStatisticsLib.CartRegressionTree as cdt

data_set = np.array([4.5, 4.75, 4.91, 5.34, 5.8, 7.05, 7.9, 8.23, 8.7, 9.0])

eigens = dict()
eigens["x"] = np.array([1,2,3,4,5,6,7,8,9,10])

cdt.CartRegressionTree.data_set_size_threshold = 3
cdt.CartRegressionTree.deviation_sum_threshold = 0.1
tree = cdt.CartRegressionTree(root=True)
tree.build(data_set, eigens)

tree.draw()