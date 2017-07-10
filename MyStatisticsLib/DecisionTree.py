import MyStatisticsLib.DecisionTreeUtil as dtu
import MyStatisticsLib.TreePlot as plt

class DecisionTree:
    info_gain_ratio_threshold = 0.1
    alpha = 0

    def __init__(self, root=False):
        self.category = None
        self.children = dict()
        self.root = root

    def build(self, data_set, eigens):
        """
        build tree with data set and eigens
        :param data_set: data set
        :param eigens: dictionary of eigens. key is eigen name, value is eigen values
        :return:
        """
        # calculate entropy and loss of this node
        self.entropy = dtu.empirical_entropy(data_set)
        self.loss = self.entropy * len(data_set)

        # get categories in the data set
        categories = set(data_set.flat)
        # if only 1 category exists, return this node as leaf node.
        if len(categories) == 1:
            self.category = data_set[0]
            return

        # use mode of all categories as node category
        mode = dtu.mode(data_set)
        self.category = mode

        # if no eigen exists, return this node as leaf node
        if eigens == None or len(eigens) == 0:
            return

        best_eigen = dtu.best_eigen_for_info_gain_ration(data_set, eigens)
        self.best_eigen_name = best_eigen[0]
        self.max_info_gain_ratio = best_eigen[1]

        # when max info gain ratio of best eigen is still less than threshold, stop split
        # return this node as leaf node
        if self.max_info_gain_ratio < DecisionTree.info_gain_ratio_threshold:
            return

        # after best eigen is used, pop it from eigens dictionary
        best_eigen_values = eigens.pop(self.best_eigen_name)
        splits = dtu.split_all(data_set, eigens, best_eigen_values)
        for split in splits:
            best_eigen_distinct_value, data_split, eigens_split = split
            child = DecisionTree()
            child.build(data_split, eigens_split)
            self.children[best_eigen_distinct_value] = child

    def loss_of_leaf(self):
        """
        calculate loss of all leaves of node
        :param node:
        :return: the root of sub-tree
        """
        if len(self.children) != 0:
            loss = 0
            for child in self.children.values():
                # sum up the loss of each leaf
                # Add regulation item (alpha) to each leaf loss.Total regulation item will be multiplied by leaf number
                loss += child.loss_of_leaf() + DecisionTree.alpha
            return loss;
        else:
            # if i am a leaf node, return my loss.
            return self.loss

    def post_prune(self):
        """
        prune tree nodes
        :return:
        """
        if len(self.children) != 0:
            for child in self.children.values():
                child.post_prune()

            # For root node, no need to prune it. Because decision tree needs at least two levels.
            if self.root:
                return

            # if loss of leaf nodes is greater or equal to loss of parent, prune this sub-tree
            if self.loss_of_leaf() >= self.loss + DecisionTree.alpha:
                self.children.clear()

    def decide(self, data):
        """
        make decision based on input data
        :param data: dictionary of a sample of all eigens
        :return:
        """
        if len(self.children) == 0:
            # if i am a leaf node, return my decision
            return self.category
        else:
            try:
                best_eigen_value = data[self.best_eigen_name]
                chosen_child = self.children[best_eigen_value]
                return chosen_child.decide(data)
            except KeyError as e:
                print("Key doesn't exist: " + e.__str__())

    def traverse(self):
        """
        :return:  Decision Tree in dictionary format
        """
        if len(self.children) != 0:
            tree = dict()
            sub_tree = dict()
            for eigen_value, child in self.children.items():
                sub_tree[eigen_value] = child.traverse()
            tree[self.best_eigen_name] = sub_tree
            return tree
        else:
            return self.category

    def draw(self):
        """
        draw Decision Tree
        """
        tree_in_dict = self.traverse()
        plt.createPlot(tree_in_dict)