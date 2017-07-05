import numpy as np


def best_split_of_eigen(data_set, eigens, eigen):
    """
    For the specified eigen, find the best eigen value to split data set which generate minimum deviation sum
    :param data_set:
    :param eigens: dictionary of all eigens. Its index is eigen name, and values are eigen values
    :param eigen: specified eigen name
    :return: specified eigen name, best eigen value, minimum deviation sum
    """
    def split_data_set(distinct_value):
        s1 = data_set[eigen_values <= distinct_value]
        # When only 1 distinct value exists or the distinct value is maximum, we get an empty s2.
        s2 = data_set[eigen_values > distinct_value]
        return s1, s2

    # calculate deviation sum of splits
    # deviation sum = variance * (sample number)
    def deviation_sum(splits):
        s1, s2 = splits
        s1_dev_sum = np.var(s1) * len(s1)
        # If s2 is empty, it's impossible to split data set.
        # So, don't let it happen by set s2_dev_sum = float('inf').
        s2_dev_sum = np.var(s2) * len(s2) if len(s2) != 0 else float('inf')
        return s1_dev_sum + s2_dev_sum

    eigen_values = eigens[eigen]
    distinct_value = list(set(eigen_values))
    # get splits per each distinct eigen value
    all_splits = map(split_data_set, distinct_value)
    # calculate deviation sum of splits per each distinct eigen value
    all_dev_sum = list(map(deviation_sum, all_splits))
    # get minimum deviation sum
    min_dev_sum = min(all_dev_sum)
    min_dev_sum_index = all_dev_sum.index(min_dev_sum)
    # get the eigen value that generate the minimum deviation sum
    best_split_value = distinct_value[min_dev_sum_index]
    return eigen, best_split_value, min_dev_sum


def choose_best_split(data_set, eigens):
    """
    Choose best split for data set
    :param data_set:
    :param eigens: dictionary of all eigens. Its index is eigen name, and values are eigen values
    :return: best eigen name, best eigen value, minimum deviation sum
              if g_min_sqr_sum is "inf", it's not possible to split data set
    """
    def best_split_of_all_eigens(eigen):
        return best_split_of_eigen(data_set, eigens, eigen)
    # find best split per eigen
    best_split_per_eigen = map(best_split_of_all_eigens, eigens)

    g_min_dev_sum = float('inf')
    for eigen, split_value, min_dev_sum in best_split_per_eigen:
        if min_dev_sum < g_min_dev_sum:
            g_best_eigen, g_best_split_value, g_min_dev_sum = eigen, split_value, min_dev_sum
    return g_best_eigen, g_best_split_value, g_min_dev_sum


def split_all(data_set, eigens, split_eigen, split_value):
    """
    split data set and eigens using split_eigen and split_value
    :param data_set:
    :param eigens: dictionary of all eigens. Its index is eigen name, and values are eigen values
    :param split_eigen: the chosen eigen returned by choose_best_split
    :param split_value: the chosen eigen value returned by choose_best_split
    :return: (data split, eigen split) of left and right child
    """
    left_child_idx = eigens[split_eigen] <= split_value
    right_child_idx = eigens[split_eigen] > split_value
    for idx in (left_child_idx, right_child_idx):
        data_set_split = data_set[idx]
        eigens_split = dict()
        for eigen in eigens:
            orig_eigen_values = eigens[eigen]
            eigens_split[eigen] = orig_eigen_values[idx]
        yield data_set_split, eigens_split