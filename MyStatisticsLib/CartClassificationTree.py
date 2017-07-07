import MyStatisticsLib.CartClassificationTreeUtil as cctu
import MyStatisticsLib.TreePlot as plt
import copy

class CartClassificationTree:
    data_set_size_threshold = 5
    cond_gini_threshold = 0.1

    def __init__(self, root=False):
        self.root = root
        self.children = list()
        self.split_eigen = None
        self.split_value = None
        self.prediction = None
        self.gini_index = None
        self.data_set_size = None

    def build(self, data_set, eigens):
        self.prediction = cctu.mode(data_set)
        self.data_set_size = len(data_set)
        self.gini_index = cctu.gini(data_set)

        # 如果数据集太小,终止分片.把当前节点作为叶结点返回.
        if self.data_set_size < CartClassificationTree.data_set_size_threshold:
            return
        # 如果基尼指数小于阈值,说明分片已经足够好.把当前节点作为叶结点返回.
        if self.gini_index < CartClassificationTree.cond_gini_threshold:
            return

        self.split_eigen, self.split_value, min_cond_gini = cctu.choose_best_split(data_set, eigens)
        splits = cctu.split_all(data_set, eigens, self.split_eigen, self.split_value)
        for data_set_split, eigens_split in splits:
            child = CartClassificationTree()
            child.build(data_set_split, eigens_split)
            self.children.append(child)

    def decide(self, eigen):
        """
        根据一个输入样本的特征值进行决策，返回决策结果
        :param eigen: 特征字典，包含一个样本的每个特征的值
        :return:决策结果
        """
        if len(self.children) == 0:
            # 如果是叶结点，直接返回决策结果
            return self.prediction
        else:
            try:
                # 如果不是叶结点，根据当前节点的分片特征和特征值选择一个孩子来决策
                eigen_value = eigen[self.split_eigen]
                child_index = 0 if self.split_value == eigen_value else 1
                chosen_child = self.children[child_index]
                # 返回孩子的决策结果
                return chosen_child.decide(eigen)
            except KeyError as e:
                print("Key doesn't exist: " + e.__str__())

    def decide_all(self, length, eigens):
        """
        可以对多个样本进行决策，返回每个样本决策结果
        :param length: 样本数量
        :param eigens: 特征字典，包含多个样本的每个特征的值
        :return: 返回一组决策结果。格式为generator，可以迭代返回多个决策结果。
        """
        def split_eigens():
            # 分割多个样本的特征字典
            # 返回值是一个generator，可以迭代返回每个样本的特征字典
            for i in range(length):
                eigens_split = dict()
                for eigen in eigens:
                    eigens_split[eigen] = eigens[eigen][i]
                yield eigens_split
        # 分割多个样本的特征字典
        eigens_splits = split_eigens()
        # 对每个样本的特征字典进行决策
        for eigens_split in eigens_splits:
            yield self.decide(eigens_split)

    def depth(self, tree_depth=0):
        """
        计算树深
        :param tree_depth:
        :return: 当前树的总深度
        """
        tree_depth += 1
        if len(self.children) != 0:
            temp_depth = list()
            for child in self.children:
                temp_depth.append(child.depth(tree_depth))
            return max(temp_depth)
        else:
            return tree_depth

    def loss(self):
        """
        计算树的损失
        :return: 树的每个叶结点的损失的列表
        """
        if len(self.children) == 0:
            # 如果是叶结点,返回叶结点的损失(= 训练数据数 * 基尼系数)
            return [ self.data_set_size * self.gini_index ]
        else:
            # 如果是非叶结点，返回它的孩子们的损失列表
            losses = list()
            for child in self.children:
                losses += child.loss()
            return losses

    def choose_best_tree(self, my_choice, alpha = float("inf")):
        """
        选择一个使得alpha=g(t)最小的子树，它是我们剪枝的目标
        :param my_choice: 列表对象，保存找到的最小alpha值和对应的子树。
        :param alpha: alpha
        :return: 在my_choice中，返回找到的最小alpha值和对应的子树
        """
        if len(self.children) != 0:
            # 对除了根结点外的非叶结点计算g(t)
            # 根结点是不剪枝的，所以不用计算它的g(t)
            if not self.root:
                # 计算g(t)值
                node_loss = self.data_set_size * self.gini_index
                tree_loss = self.loss()
                leaf_num = len(tree_loss)
                gt = (node_loss - sum(tree_loss)) / (leaf_num - 1)
                # 如果g(t)小于当前的最小alpha值，把最小alpha值和对应的子树保存到my_choice里
                if gt < alpha:
                    alpha = gt
                    my_choice.clear()
                    my_choice.append((alpha, self))
            # 遍历所有孩子结点，计算它们的g(t)值。如果发现更小的g(t)，则更新my_choice
            for child in self.children:
                child.choose_best_tree(my_choice, alpha)

    def post_prune(self, test_data_set, test_data_eigens):
        """
        根据测试数据对CART树做后剪枝
        :param test_data_set: 测试数据集的类别
        :param test_data_eigens: 测试数据集的特征
        :return: 最优CART树
        """
        # best_trees列表保存剪枝过程中找到的所有alpha和相应的最优CART树
        # 列表中每个元素的格式为(alpha，对应的最优CART树）
        best_trees = list()
        # alpha=0时，整树self为最优CART树
        best_trees.append((0, copy.deepcopy(self)))
        # 一直循环直到剩下根结点和两个叶子组成的树（depth=2）
        while self.depth() > 2:
            choice = list()
            # 寻找产生当前最小alpha的子树
            self.choose_best_tree(choice)
            alpha, subtree = choice[0]
            # 对子树剪枝
            subtree.children.clear()
            # 把当前最小alpha以及剪枝后的最优CART树保存到best_trees列表
            best_trees += [(alpha, copy.deepcopy(self))]

        # 用测试数据集验证best_trees列表中的哪个CART树泛化能力最强
        length = len(test_data_set)
        # max_correct记录最大正确率，best_tree记录对应的最优CART树
        max_correct, best_tree = -1, None
        for alpha, tree in best_trees:
            # 得到当前树对测试数据集的决策结果
            decisions = tree.decide_all(length, test_data_eigens)
            # 对照test_data_set，统计正确率
            correct = 0
            for data, decision in zip(test_data_set, decisions):
                if data == decision:
                    correct += 1
            # 如果正确率最大，则把当前树记录下来
            if max_correct < correct:
                max_correct = correct
                best_tree = tree
        return best_tree

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