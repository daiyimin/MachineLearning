import numpy as np
import MyStatisticsLib.CartDecisionTreeUtil as cdtu

data_set = np.array([4.5, 4.75, 4.91, 5.34, 5.8, 7.05, 7.9, 8.23, 8.7, 9.0])

eigens = dict()
eigens["x"] = np.array([1,2,3,4,5,6,7,8,9,10])

split_eigen, split_value, min_sqr_sum = cdtu.choose_best_split(data_set, eigens)

splits = cdtu.split_all(data_set, eigens, split_eigen, split_value)

for split in splits:
    print(split)