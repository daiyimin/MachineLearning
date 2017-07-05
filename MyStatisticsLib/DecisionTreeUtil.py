from functools import reduce
import math


def empirical_entropy(date_set):
    """
    calculate entropy of data set
    :param date_set: data set for calculation. The information in data set is the category of data.
    :return: empirical entropy of data set
    """
    def calc_partial_entropy(category):
        cnt = sum(date_set==category)
        prob = cnt/float(total)
        return -1 * prob * math.log(prob, 2)
    total = len(date_set)
    categories = set(date_set.flat)
    entropy_values = map(calc_partial_entropy, categories)
    return sum(entropy_values)


def split_data_set(data_set, eigen_values):
    """
    Split data set to small splits according to eigen values. Each split corresponds to 1 distinct eigen value
    :param data_set: data set to be split
    :param eigen_values: eigen values of the best eigen. The best eigen is chosen by best_eigen_for_info_gain_ration
    :return: a split of data set corresponding to respective eigen value
    """
    distinct_values = set(eigen_values)
    for value in distinct_values:
        yield data_set[eigen_values == value]


def split_all(data_set, eigens, eigen_values):
    """
    Split data set and values of all eigens to small splits according to eigen values of best eigen
    Each split corresponds to 1 distinct eigen value.
    :param data_set: data set to be split
    :param eigens: a dictionary of all eigens and their values
    :param eigen_values: eigen values of the best eigen. The best eigen is chosen by best_eigen_for_info_gain_ration
    :return: value => distinct eigen value
              data_set_split => a split of data set corresponding to the "value"
              eigens_split => a split of dictionary of eigens corresponding to the "value".
    """
    distinct_values = set(eigen_values)
    for value in distinct_values:
        index = (eigen_values == value)
        data_set_split = data_set[index]
        eigens_split = dict()
        for eigen in eigens.keys():
            orig_eigen_values = eigens[eigen]
            eigens_split[eigen] = orig_eigen_values[index]
        yield (value, data_set_split, eigens_split)


def empirical_cond_entropy(data_set, eigen_values):
    """
    Calculate conditional entropy for a specified eigen
    :param data_set: data set for calculation
    :param eigen_values: values of specified eigen (call it eigen A) of this data set.
            note: The real name of eigen A is not passed in, because it's not related to the calculation.
    :return: empirical conditional entropy of data set for eigen A
    """
    def calc_partial_cond_entropy(split):
        prob = len(split)/float(total)
        entropy = empirical_entropy(split)
        return prob*entropy

    total = len(data_set)
    splits = split_data_set(data_set, eigen_values)
    entropy_values = map(calc_partial_cond_entropy, splits)
    return sum(entropy_values)


def empirical_cond_entropy_of_eigens(data_set, eigens):
    """
    Calculate conditional entropy of all eigens
    This function iterate each eigen in eigens, and call empirical_cond_entropy for each eigen.
    :param data_set: data set for calculation
    :param eigens: dictionary of all eigens and their values
    :return: list of tuple. Each tuple is like (eigen X, conditional entropy of eigen X)
    """
    def empirical_cond_entropy_closure(eigen_values):
        return empirical_cond_entropy(data_set, eigen_values)
    entropy_per_eigen = map(empirical_cond_entropy_closure, eigens.values())
    return zip(eigens.keys(), entropy_per_eigen)


def empirical_info_gain_of_eigens(data_set, eigens):
    """
    Calculate information gain of all eigens respectively
    This function iterate each eigen in eigens, and calculate information gain for each eigen
    :param data_set: data set belonging to one node of decision tree
    :param eigens: dictionary of eigen name and eigen values
    :return: list of tuple. Each tuple is like (eigen A, info gain of eigen A)
    """
    def empirical_info_gain_closure(eigen_values):
        cond_entropy = empirical_cond_entropy(data_set, eigen_values)
        info_gain = entropy - cond_entropy
        return info_gain
    entropy = empirical_entropy(data_set)
    info_gain_per_eigen = map(empirical_info_gain_closure, eigens.values())
    return zip(eigens.keys(), info_gain_per_eigen)


def empirical_info_gain_ratio_of_eigens(data_set, eigens):
    """
    Calculate information gain ratio of all eigens respectively
    This function iterate each eigen in eigens, and calculate information gain ratio for each eigen
    :param data_set: data set belonging to one node of decision tree
    :param eigens: dictionary of eigen name and eigen values
    :return: list of tuple. Each tuple is like (eigen A, info gain ratio of eigen A)
    """
    def empirical_info_gain_ratio_closure(eigen_values):
        cond_entropy = empirical_cond_entropy(data_set, eigen_values)
        info_gain = entropy - cond_entropy
        return info_gain/entropy
    entropy = empirical_entropy(data_set)
    info_gain_ratio_per_eigen = map(empirical_info_gain_ratio_closure, eigens.values())
    return zip(eigens.keys(), info_gain_ratio_per_eigen)


def best_eigen_for_info_gain_ration(data_set, eigens):
    """
    Choose the best eigen which has max information gain ratio
    :param data_set: data set belonging to one node of decision tree
    :param eigens: dictionary of eigen name and eigen values
    :return: tuple(best eigen name, maximum info gain ratio)
    """
    info_gain_ratios = empirical_info_gain_ratio_of_eigens(data_set, eigens)
    # lambda that returns the best eigen and maximum information gain ratio
    l_get_best = lambda igr1, igr2: igr1 if igr1[1] > igr2[1] else igr2
    best_info_gain_ratio = reduce(l_get_best, info_gain_ratios)
    return best_info_gain_ratio


def mode(data_set):
    """
    :param data_set:
    :return: mode of data set
    """
    distinct_data = set(data_set.flat)
    #  lambda returns a tuple of (distinct data, its count)
    l_dist_sum = lambda d: (d, sum(data_set == d))
    distinct_data_sum_list = map(l_dist_sum, distinct_data)
    # lambda returns the mode (= a data which has maximum count)
    l_get_mode = lambda d1, d2: d1 if d1[1] > d2[1] else d2
    mode = reduce(l_get_mode, distinct_data_sum_list)
    return mode[0]