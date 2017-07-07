import MyStatisticsLib.CartRegressionTreeUtil as crtu
import MyStatisticsLib.TreePlot as plt
import numpy as np


class CartRegressionTree:
    data_set_size_threshold = 5
    deviation_sum_threshold = 1

    def __init__(self, root=False):
        self.root = root
        self.children = list()
        self.split_eigen = None
        self.split_value = None
        self.prediction = None
        self.deviation_sum = None
        self.data_set_size = None

    def build(self, data_set, eigens):
        """
        构造CART回归树
        :param data_set: 测试数据集，类型是单轴的numpy.array
        :param eigens: 特征字典，字典的键值是特征名，类型是字符串；字典的值是特征数组,类型是单轴的numpy.array
        :return:
        """
        self.prediction = np.average(data_set)
        self.data_set_size = len(data_set)
        self.deviation_sum = crtu.deviation_sum(data_set)

        # 如果数据集太小,终止分片.把当前节点作为叶结点返回.
        if self.data_set_size < CartRegressionTree.data_set_size_threshold:
            return
        # 如果最小的差方和小于预设阈值,说明数据集分片已经足够好.把当前节点作为叶结点返回.
        if self.deviation_sum < CartRegressionTree.deviation_sum_threshold:
            return

        self.split_eigen, self.split_value, min_dev_sum = crtu.choose_best_split(data_set, eigens)
        # 根据选出的分片特征(split_eigen)和特征值(split_value),把数据集和特征字典分片
        splits = crtu.split_all(data_set, eigens, self.split_eigen, self.split_value)
        # 给每个分片创建相应的子树
        for data_set_split, eigens_split in splits:
            child = CartRegressionTree()
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
        draw CART Tree
        """
        tree_in_dict = self.traverse()
        plt.createPlot(tree_in_dict)