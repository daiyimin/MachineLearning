import numpy as np
import math

# date_set: data set for calculation. The value of data set is category of each data.
# return: empirical entropy of data set
def empirical_entropy(date_set):
    def calc_partial_entropy(category):
        cnt = sum(date_set==category)
        prob = cnt/float(total)
        return -1 * prob * math.log(prob, 2)
    total = len(date_set)
    categories = set(date_set.flat)
    entropy_values = map(calc_partial_entropy, categories)
    return sum(entropy_values)

def split_data_set(data_set, eigen_values):
    distinct_values = set(eigen_values)
    for value in distinct_values:
        yield data_set[eigen_values == value]


def split_all(data_set, eigens, eigen_values):
    distinct_values = set(eigen_values)
    for value in distinct_values:
        index = (eigen_values == value)
        data_set_split = data_set[index]
        eigens_split = dict()
        for eigen in eigens.keys():
            orig_eigen_values = eigens[eigen]
            eigens_split[eigen] = orig_eigen_values[index]
        yield (value, data_set_split, eigens_split)

# date_set: data set belonging to one node of decision tree
# eigen_values: values of one eigen (call it eigen A) of this data set.
#      note: The real name of eigen A is not passed in, because it's not related to the calculation.
# return: empirical conditional entropy of data set for eigen A
def empirical_cond_entropy(data_set, eigen_values):
    def calc_partial_cond_entropy(split):
        prob = len(split)/float(total)
        entropy = empirical_entropy(split)
        return prob*entropy

    total = len(data_set)
    splits = split_data_set(data_set, eigen_values)
    entropy_values = map(calc_partial_cond_entropy, splits)
    return sum(entropy_values)

# date_set: data set belonging to one node of decision tree
# eigens: map of eigen name and eigen values.
# return: list of tuple. Each tuple is like (eigen A, conditional entropy of eigen A)
def empirical_cond_entropy_of_eigens(data_set, eigens):
    def empirical_cond_entropy_closure(eigen_values):
        return empirical_cond_entropy(data_set, eigen_values)
    entropy_per_eigen = map(empirical_cond_entropy_closure, eigens.values())
    return zip(eigens.keys(), entropy_per_eigen)

# date_set: data set belonging to one node of decision tree
# eigens: map of eigen name and eigen values.
# return: list of tuple. Each tuple is like (eigen A, info gain of eigen A)
def empirical_info_gain_of_eigens(data_set, eigens):
    def empirical_info_gain_closure(eigen_values):
        cond_entropy = empirical_cond_entropy(data_set, eigen_values)
        info_gain = entropy - cond_entropy
        return info_gain
    entropy = empirical_entropy(data_set)
    info_gain_per_eigen = map(empirical_info_gain_closure, eigens.values())
    return zip(eigens.keys(), info_gain_per_eigen)

# date_set: data set belonging to one node of decision tree
# eigens: map of eigen name and eigen values.
# return: list of tuple. Each tuple is like (eigen A, info gain ratio of eigen A)
def empirical_info_gain_ratio_of_eigens(data_set, eigens):
    def empirical_info_gain_ratio_closure(eigen_values):
        cond_entropy = empirical_cond_entropy(data_set, eigen_values)
        info_gain = entropy - cond_entropy
        return info_gain/entropy
    entropy = empirical_entropy(data_set)
    info_gain_ratio_per_eigen = map(empirical_info_gain_ratio_closure, eigens.values())
    return zip(eigens.keys(), info_gain_ratio_per_eigen)

# date_set: data set belonging to one node of decision tree
# eigens: map of eigen name and eigen values.
# return: tuple(best eigen name, maximum info gain ratio)
def best_eigen_for_info_gain_ration(data_set, eigens):
    info_gain_ratios = empirical_info_gain_ratio_of_eigens(data_set, eigens)
    l_get_best = lambda igr1, igr2: igr1 if igr1[1] > igr2[1] else igr2
    best_info_gain_ratio = reduce(l_get_best, info_gain_ratios)
    return best_info_gain_ratio

# return: mode of data set
def mode(data_set):
    distinct_data = set(data_set.flat)
    l_dist_sum = lambda d: (d, sum(data_set == d))
    distinct_data_sum_list = map(l_dist_sum, distinct_data)
    l_get_mode = lambda d1, d2: d1 if d1[1] > d2[1] else d2
    mode = reduce(l_get_mode, distinct_data_sum_list)
    return mode[0]