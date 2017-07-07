import MyStatisticsLib.CartClassificationTreeUtil as cctu
import MyStatisticsLib.TreePlot as plt
import numpy as np


class CartClassificationTree:
    data_set_size_threshold = 5
    cond_gini_threshold = 0.1

    def __init__(self, root=False):
        self.root = root
        self.children = list()
        self.split_eigen = None
        self.split_value = None
        self.prediction = None
        self.gini = None
        self.data_set_size = None

    def build(self, data_set, eigens):
        self.prediction = cctu.mode(data_set)
        self.gini = cctu.gini(data_set)
        self.data_set_size = len(data_set)

        # if data set is too small, return this node as leaf node
        if len(data_set) < CartClassificationTree.data_set_size_threshold:
            return
        # if gini index of data set is less than threshold, split is good enough. Return this node as leaf node
        if self.gini < CartClassificationTree.cond_gini_threshold:
            return

        self.split_eigen, self.split_value, min_cond_gini = cctu.choose_best_split(data_set, eigens)
        splits = cctu.split_all(data_set, eigens, self.split_eigen, self.split_value)
        for data_set_split, eigens_split in splits:
            child = CartClassificationTree()
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
            condition = ["==", "!="]
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