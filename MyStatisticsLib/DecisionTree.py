import MyStatisticsLib.DecisionTreeUtil as dtu

class DecisionTree:
    info_gain_ratio_threshold = 0.1
    alpha = 0

    def __init__(self, root=False):
        self.category = None
        self.children = dict()
        self.root = root

    # build tree with data set and eigens
    def build(self, data_set, eigens):
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

    # calculate loss of all leaves of node
    # node, the root of sub-tree
    def loss_of_leaf(node):
        if len(node.children) != 0:
            loss = 0
            for child in node.children:
                child_node = node.children[child]
                # sum up the loss of each leaf
                # Add regulation item (alpha) to each leaf loss.Total regulation item will be multiplied by leaf number
                loss += DecisionTree.loss_of_leaf(child_node) + DecisionTree.alpha
            return loss;
        else:
            # if i am a leaf node, return my loss.
            return node.loss

    # prune tree nodes
    def post_prune(self):
        if len(self.children) != 0:
            for child in self.children:
                child_node = self.children[child]
                child_node.post_prune()

            # For root node, no need to prune it. Because decision tree needs at least two levels.
            if self.root:
                return

            # if loss of leaf nodes is greater or equal to loss of parent, prune this sub-tree
            if DecisionTree.loss_of_leaf(self) >= self.loss + DecisionTree.alpha:
                self.children.clear()

    # make decision based on input data
    # data, dictionary of a sample of all eigens
    # return: decision
    def decide(self, data):
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