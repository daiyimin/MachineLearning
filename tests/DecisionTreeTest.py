import MyStatisticsLib.DecisionTree as dt
import numpy as np

# category of data set, 0 = no, 1 = yes
data_set = np.array([0,0,1,1,0,0,0,1,1,1,1,1,1,1,0])

eigens = dict()
# for age, 1 = young, 2 = middle aged, 3 = old
eigens["age"] = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3])
# for employment, 0 = no, 1 = yes
eigens["employment"] = np.array([0,0,1,1,0,0,0,1,0,0,0,0,1,1,0])
# for house, 0 = no, 1 = yes
eigens["house"] = np.array([0,0,0,1,0,0,0,1,1,1,1,1,0,0,0])
# for loan, 1 = normal, 2 = good, 3 = very good
eigens["loan"] = np.array([1,2,2,1,1,1,2,2,3,3,3,2,2,3,1])

data_eigens = dict()
data_eigens["age"] = 3
data_eigens["employment"] = 0
data_eigens["house"] = 1
data_eigens["loan"] = 3

dt.DecisionTree.alpha = 1
dt.DecisionTree.info_gain_ratio_threshold = 0.1

tree = dt.DecisionTree(root=True)
tree.build(data_set, eigens)
tree.post_prune()

decision = tree.decide(data_eigens)
print(decision)