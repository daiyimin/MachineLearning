import MyStatisticsLib.CartDecisionTreeUtil as cdtu
import MyStatisticsLib.TreePlot as plt
import numpy as np


class CartDecisionTree:
    data_set_size_threshold = 5
    deviation_sum_threshold = 1

    def __init__(self, root=False):
        self.root = root
        self.children = list()
        self.split_eigen = None
        self.split_value = None
        self.prediction = None

    def build(self, data_set, eigens):
        self.prediction = np.average(data_set)

        # if data set is too small, return this node as leaf node
        if len(data_set) < CartDecisionTree.data_set_size_threshold:
            return

        self.split_eigen, self.split_value, min_sqr_sum = cdtu.choose_best_split(data_set, eigens)
        # if minimum squared sum is less than threshold, split is good enough. Return this node as leaf node
        if min_sqr_sum < CartDecisionTree.deviation_sum_threshold:
            return

        splits = cdtu.split_all(data_set, eigens, self.split_eigen, self.split_value)
        for data_set_split, eigens_split in splits:
            child = CartDecisionTree()
            child.build(data_set_split, eigens_split)
            self.children.append(child)

    def traverse(self):
        """
        :return:  CART Tree in dictionary format
        """
        if len(self.children) != 0:
            tree = dict()
            sub_tree = dict()
            split_value = str(self.split_value)
            condition = ["<=", ">"]
            for child, cond in zip(self.children, condition):
                sub_tree[cond + split_value] = child.traverse()
            tree[self.split_eigen] = sub_tree
            return tree
        else:
            return self.prediction

    def draw(self):
        """
        draw Decision Tree
        """
        tree_in_dict = self.traverse()
        plt.createPlot(tree_in_dict)