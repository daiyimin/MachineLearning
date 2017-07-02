import MyStatisticsLib.DecisionTreeUtil as dtu
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

# entropy = dtu.empirical_entropy(data_set)
#
# print(entropy)
#
# empirical_cond_entropy = dtu.empirical_cond_entropy_of_eigens(data_set, eigens)
# print(empirical_cond_entropy)
#
# empirical_info_gain = dtu.empirical_info_gain_of_eigens(data_set, eigens)
# print(empirical_info_gain)
#
empirical_info_gain_ratio = dtu.best_eigen_for_info_gain_ration(data_set, eigens)
print(empirical_info_gain_ratio)

mode = dtu.mode(eigens["loan"] )
print(mode)

splits = dtu.split_all(data_set, eigens, eigens["loan"])
for split in splits:
    print(split)