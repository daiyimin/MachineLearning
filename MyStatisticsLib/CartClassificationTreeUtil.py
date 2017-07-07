from functools import reduce


def gini(data_set):
    def parital_gini(value):
        count_of_value = sum(data_set == value)
        prob = count_of_value/float(total)
        return -1 * prob * prob
    total = len(data_set)
    gini_index = 1
    distinct_value = set(data_set.flat)
    for value in distinct_value:
        gini_index += parital_gini(value)
    return gini_index


def cond_gini(splits):
    cond_gini_index = 0
    total = 0
    for s in splits:
        cond_gini_index += len(s) * gini(s)
        total = len(s)
    cond_gini_index *= 1/float(total)
    return cond_gini_index


def best_split_of_eigen(data_set, eigens, eigen):
    """
    For the specified eigen, find the best eigen value to split data set which generate minimum conditional gini index
    :param data_set:
    :param eigens: dictionary of all eigens. Its index is eigen name, and values are eigen values
    :param eigen: specified eigen name
    :return: specified eigen name, best eigen value, minimum conditional gini index
    """
    def split_data_set(distinct_value):
        s1 = data_set[eigen_values == distinct_value]
        # When only 1 distinct value exists or the distinct value is maximum, we get an empty s2.
        s2 = data_set[eigen_values != distinct_value]
        return s1, s2

    eigen_values = eigens[eigen]
    distinct_value = list(set(eigen_values))
    if len(distinct_value) == 1:
        # if only one distinct eigen value, then not possible to split the data set
        return eigen, distinct_value, float("inf")

    # get splits per each distinct eigen value
    all_splits = map(split_data_set, distinct_value)
    # calculate conditional gini index per distinct eigen value
    all_cond_gini = list(map(cond_gini, all_splits))
    # get minimum conditional gini index
    min_cond_gini = min(all_cond_gini)
    min_cond_gini_index = all_cond_gini.index(min_cond_gini)
    # get the eigen value that generate the minimum conditional gini index
    best_split_value = distinct_value[min_cond_gini_index]
    return eigen, best_split_value, min_cond_gini


def choose_best_split(data_set, eigens):
    """
    Choose best split for data set
    :param data_set:
    :param eigens: dictionary of all eigens. Its index is eigen name, and values are eigen values
    :return: best eigen name, best eigen value, minimum conditional gini index
    """
    def best_split_of_all_eigens(eigen):
        return best_split_of_eigen(data_set, eigens, eigen)
    # find best split per eigen
    best_split_per_eigen = map(best_split_of_all_eigens, eigens)

    g_min_cond_gini = float('inf')
    for eigen, split_value, min_cond_gini in best_split_per_eigen:
        if min_cond_gini < g_min_cond_gini:
            g_best_eigen, g_best_split_value, g_min_cond_gini = eigen, split_value, min_cond_gini
    return g_best_eigen, g_best_split_value, g_min_cond_gini


def split_all(data_set, eigens, split_eigen, split_value):
    """
    split data set and eigens using split_eigen and split_value
    :param data_set:
    :param eigens: dictionary of all eigens. Its index is eigen name, and values are eigen values
    :param split_eigen: the chosen eigen returned by choose_best_split
    :param split_value: the chosen eigen value returned by choose_best_split
    :return: (data split, eigen split) of left and right child
    """
    left_child_idx = eigens[split_eigen] == split_value
    right_child_idx = eigens[split_eigen] != split_value
    for idx in (left_child_idx, right_child_idx):
        data_set_split = data_set[idx]
        eigens_split = dict()
        for eigen in eigens:
            orig_eigen_values = eigens[eigen]
            eigens_split[eigen] = orig_eigen_values[idx]
        yield data_set_split, eigens_split

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