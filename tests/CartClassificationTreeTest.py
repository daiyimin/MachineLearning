import MyStatisticsLib.CartClassificationTree as cct
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

cct.CartClassificationTree.data_set_size_threshold = 1
cct.CartClassificationTree.cond_gini_threshold = 0.001

tree = cct.CartClassificationTree(root=True)
tree.build(data_set, eigens)

tree.draw()

test_data_set = np.array([0,0,1,1])
test_data_eigens = dict()
test_data_eigens["age"] = np.array([1,2,3,2])
test_data_eigens["employment"] = np.array([0,1,1,0])
test_data_eigens["house"] = np.array([0,0,1,1])
test_data_eigens["loan"] = np.array([1,2,2,1])

best_tree = tree.post_prune(test_data_set, test_data_eigens)
best_tree.draw()