import DecisionTreeUtil as dtu

class DecisionTree:
    def __init__(self, category=None, info_gain_ratio_threshold=0.1):
        self.category = category
        self.children = dict()
        self.info_gain_ratio_threshold = info_gain_ratio_threshold

    def build(self, data_set, eigens):
        # get categories in the data set
        categories = set(data_set.flat)
        # if only 1 category exists, then return a Decision Tree with only root node.
        if len(categories) == 1:
            return DecisionTree(data_set[0])
        # if no eigen exists, then return Decision Tree with only root node
        if eigens == None or len(eigens) == 0:
            # use mode of all categories as root node category
            mode = dtu.mode(data_set)
            return DecisionTree(mode)

        best_eigen = dtu.best_eigen_for_info_gain_ration(data_set, eigens)
        self.max_info_gain_ratio = best_eigen[1]
        # when max info gain ratio of best eigen is still less than threshold, stop split
        # return this node as leaf node
        if self.max_info_gain_ratio < self.info_gain_ratio_threshold:
            # use mode of all categories as root node category
            mode = dtu.mode(data_set)
            return DecisionTree(mode)

        self.best_eigen_name = best_eigen[0]
        best_eigen_values = eigens.pop(self.best_eigen_name)
        splits = dtu.split_all(data_set, eigens, best_eigen_values)
        # after best eigen is used, pop it from eigens dictionary
        for split in splits:
            child = DecisionTree()
            child.build(split[1], split[2])
            self.children[split[0]] = child