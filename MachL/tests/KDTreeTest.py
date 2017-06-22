import numpy as np
from MyStatisticsLib import KDTree

#data = np.array([[2,3], [5,4], [9,6], [4,7], [8,1], [7,2], [7,4], [3.2,5]])
data = np.array([[9,6,3], [4,7,4], [8,1,5], [7,2,2], [7,4,3], [3.2,5,1]])

tree = KDTree.KDTree()
tree.build(data)

target = np.array([3,1,4])
neibour = tree.search(target, k=1)

print(neibour)


def distance(target, node_data):
    data = node_data  # remove the original index column
    squared_difference = np.square(target - data)
    squared_difference_sum = np.sum(squared_difference, axis=1)
    distance = np.sqrt(squared_difference_sum)
    return distance

print(distance(target, data))