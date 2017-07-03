import numpy as np
from collections import OrderedDict

# MinKOrderedDict是一个工具类，用来保存最近的K个邻居
# 在遍历KD树过程中，用户可以向MinKOrderedDict添加新发现到的邻居，不用关心邻居和目标点的距离。
# 当MinKOrderedDict中的邻居个数超过K时，它会自动把最远的邻居剔除，始终只保存距离目标点最近的K个邻居。
# 字典的Key是浮点型实数，代表了邻居和目标点的距离。换一个说法，是邻居和目标点组成的超球的半径。
# 字典的Value是一个数组列表（邻居列表），保存了和目标点距离等于Key的所有点。换一个说法，是半径为“Key”的超球上的所有点。每个
# 数组代表一个邻居。数组最后一个数字代表该邻居在构建KD树的原始数据中的下标/位置。
# 例子：MinKOrderedDict([(4.5825756949558398, [array([ 7.,  2.,  2.,  3.])])])
# Key=4.582，代表邻居和目标点的距离。Value的列表中有一个邻居array([7,2,2,3])，邻居的坐标[7,2,2]，邻居在原始数据的位置是3
class MinKOrderedDict(OrderedDict):
    # k: 字典内最多能够保存的最近邻居数
    # capacity: 字典内当前保存的最近邻居数, 默认值为 0
    def __init__(self, k):
        self.k = k
        self.capacity = 0
        OrderedDict.__init__(self)

    # 返回值 字典内当前保存的最近邻居数
    def getCapacity(self):
        return self.capacity

    # 返回值 字典内当前保存的邻居中，离目标点最远的距离(=MaxKey)
    # 换一个说法，字典内当前保存的邻居和目标点组成的超球中，最大的超球半径。
    # 最大的超球里有capacity个邻居。
    def getMaxKey(self):
        sorted_keys = sorted(self.keys())
        return sorted_keys[-1]

    # 重载__setitem__方法
    def __setitem__(self, key, value):
        containsKey = 1 if key in self else 0
        if containsKey:
            # 如果key已经存在，意味着在深度搜索KD树过程中，我们已经发现了另一个邻居和目标点的距离等于key
            # 故，把value中传入的新邻居加入列表末端
            item = OrderedDict.__getitem__(self, key)
            item.append(value)
            OrderedDict.__setitem__(self, key, item)
        else:
            # 如果key不存在，意味这这是发现的第一个和目标点的距离等于key的邻居
            # 故，在字典中加入新的key，value对
            OrderedDict.__setitem__(self, key, [value])
        # capacity加一
        self.capacity += 1
        # 比较capacity和
        if self.capacity > self.k:
            # capacity大于K时，字典中有K+1个最近邻居，把最远的那个删掉
            # 按照和目标点的距离对邻居排序
            sorted_keys = sorted(self.keys())
            # 找到最远的距离，取出对应的邻居列表
            item = OrderedDict.__getitem__(self, sorted_keys[-1])
            if len(item) > 1:
                # 如果列表中有多于1个邻居，那么从中任意删除一个
                item.pop()
            else:
                # 如果列表中只有1个邻居，把key删掉
                OrderedDict.__delitem__(self, sorted_keys[-1])
            # capacity减一
            self.capacity -=1

# KDTree
class KDTree:
    # depth 是当前节点在树中的深度，根节点深度为0
    def __init__(self, depth=0, parent=None):
        self.depth = depth
        self.parent = parent
        self.split = -1
        self.median = -1
        self.data = None
        self.lleaf = None
        self.rleaf = None

    #  构建KD树
    #  data是一个m乘n的矩阵。矩阵的每一行是一个邻居点，行中的每一列是邻居的一个坐标/维度
    #  data中的行数代表邻居个数。列数代表邻居坐标的总维度。
    #  返回值 NULL
    #  example  data = [ [1,2,3], [4,5,6],[7,8,9] ], 其中 [1,2,3]是一个邻居, [4,5,6]和[7,8,9] 是其它的邻居.
    def build(self, data):
        if self.depth == 0:
            # 进入根节点
            # 把原始数据保存在orig_data中
            self.orig_data = data
            # 在邻居数据最后增加一列，标记每个邻居在原始数据中的位置，方便使用KD树查找结果确定邻居的类型
            # 注意，最后一列是不参加KD树的构建和搜索的，build和search代码会排除它们。
            m = len(data)
            orig_idx = np.arange(0,m)
            orig_idx.shape = (m,1)
            data = np.concatenate((data, orig_idx), axis=1)

        # 获取邻居坐标的总维度，-1排除最后一列(原始数据中的位置)。
        n = data.shape[1] - 1
        # 例子里，我们在[depth%n]维度上划分邻居数据
        self.split = self.depth % n

        #  对邻居点的[depth%n]维度坐标排序
        idx = np.argsort(data[:, self.split])
        data = data[idx]

        #  取得中位数
        median_idx = int(len(data)/2)
        self.median = data[median_idx, self.split]

        # 经过邻居[depth%n]维度坐标中位数的超平面把邻居划分为两部分。
        # 这个超平面记为当前节点的超平面Sc，每个非叶节点都有自己的超平面。
        # 把超平面上的点保存在当前节点内
        midx = np.where(data[:, self.split] == self.median)
        if np.size(midx) > 0:
            #  save those points to this tree node
            self.data = data[midx]
        # 把划分超平面右侧的点保存在右子树中，递归实现
        midx = np.where(data[:, self.split] > self.median)
        if np.size(midx) > 0:
            self.rleaf = KDTree(self.depth + 1, self)
            self.rleaf.build(data[midx])
        # 把划分超平面左侧的点保存在左子树中，递归实现
        midx = np.where(data[:, self.split] < self.median)
        if np.size(midx) > 0:
            self.lleaf = KDTree(self.depth + 1, self)
            self.lleaf.build(data[midx])

    def has_lleaf(self):
        return self.lleaf != None

    def has_rleaf(self):
        return self.rleaf != None

    #  search the nearest neighbor of target
    #  target: target data to be searched
    #  min_dist: current minimun dist point tuple, format (min dist, nearest point)
    #  return: nearest neighbor of target
    def searchNearest(self, target, min_dist=(-1,None)):
        # calculate the distance between target and points on the split surface of current node
        # return tuple of min distance and nearest point, (min distance, nearest point)
        def distance(target, node_data):
            data = node_data[:,:-1] # remove the original index column
            squared_difference = np.square(target - data)
            squared_difference_sum = np.sum(squared_difference, axis=1)
            distance = np.sqrt(squared_difference_sum)
            min_distance = distance.min()
            min_idx = np.where(distance == min_distance)
            return (min_distance, node_data[min_idx])

        # get the nearest point on the split surface of current node
        node_distance = distance(target, self.data)
        # if distance of nearest point is less than min distance, update min distance
        if min_dist[0] < 0 or min_dist[0] > node_distance[0]:
            min_dist = node_distance

        #  depth first search in KD tree
        #  self.split is the dimension by which child tree nodes are split
        if target[self.split] <= self.median:
            # if target is less than median on the split dimension, search left child
            if self.has_lleaf():
                min_dist = self.lleaf.searchNearest(target, min_dist)
        else:
            # if target is bigger than median on the split dimension, search right child
            if self.has_rleaf():
                min_dist = self.rleaf.searchNearest(target, min_dist)

        # for a non-leaf node, we need to judge if their children contains a nearest neighbor
        if self.has_lleaf() or self.has_rleaf():
            # first, only when the super-ball intersects with split super-surface of current node
            # then the node's children will probably contains nearest neighbor
            #       D = distance between center of ball and super-surface = "np.abs(target[self.split] - self.median)"
            #       R = radius of ball = min_dist[0]
            #       if ( D < R ) then super-surface intersects with the ball
            # second, because we already search on child of current node during depth first search, now it's time
            # to try the other node
            if np.abs(target[self.split] - self.median) < min_dist[0]:
                # search the other child that didn't tried during depth first search
                if target[self.split] <= self.median:
                    # the same condition is used here as depth first search,
                    # but this time search the right leaf
                    if self.has_rleaf():
                        min_dist = self.rleaf.searchNearest(target, min_dist)
                else:
                    if self.has_lleaf():
                        min_dist = self.lleaf.searchNearest(target, min_dist)
        return min_dist

    #  搜索目标点最近的K个邻居
    #  target: 目标点
    #  k: 待返回的邻居数
    #  返回值 od_ball，数据字典类型。保存了和目标点最近的K个邻居，具体见 MinKOrderedDict
    def search(self, target, k=1, od_ball=None):
        # distance方法，计算目标点和node_data数组中的所有邻居的距离
        # 返回距离数组
        def distance(target, node_data):
            data = node_data[:, :-1]  # remove the original index column
            squared_difference = np.square(target - data)
            squared_difference_sum = np.sum(squared_difference, axis=1)
            distance = np.sqrt(squared_difference_sum)
            return distance

        if self.depth ==0:
            # 进入根节点
            # 初始化od_ball
            od_ball = MinKOrderedDict(k)

            # 如果K值大于原始数据点个数，那么直接返回所有的原始数据点
            m = len(self.orig_data)
            if k >= m:
                # 把原始数据点添加到od_ball中
                # 此时目标点和数据点的距离无关紧要，用range（0，m)代替
                for i in range(0,m):
                    od_ball[i] = self.orig_data[i]
                return od_ball

        # 计算当前节点（划分超平面上）的点和目标点的距离，即超球的半径
        node_radius = distance(target, self.data)
        # 根据半径把点加入od_ball中
        for i in range(0, len(self.data)):
            od_ball[node_radius[i]] = self.data[i]

        # 深度优先搜索KD树，先搜索距离目标点近的子树，递归实现。self.split是划分超平面的维度。
        if target[self.split] <= self.median:
            if self.has_lleaf():
                od_ball = self.lleaf.search(target, k, od_ball)
        else:
            if self.has_rleaf():
                od_ball = self.rleaf.search(target, k, od_ball)

        # 递归搜索返回后，对于非叶节点，判断距离目标点远的子树是否含有比搜索结果更近的邻居
        if self.has_lleaf() or self.has_rleaf():
            # 1）如果，搜索结果中的邻居个数小于k，则需要寻找比结果中超球更大的超球。
            # 2）如果，搜索结果中的邻居个数等于k，则判断结果中最大的超球和当前节点的超平面Sc是否相交。如果相交，说明
            # 距离目标点远的子树内可能有更近的邻居。
            #    相交的条件：在self.split维度上，目标点和当前节点的超平面Sc的距离<结果中最大超球的半径
            #    目标点和当前节点的超平面Sc的距离 = target[self.split] - self.median
            #    结果中最大超球的半径 = od_ball.getMaxKey()
            # 1）或 2）成立时，搜索距离目标点远的子树。
            if od_ball.capacity < k or np.abs(target[self.split] - self.median) < od_ball.getMaxKey():
                if target[self.split] <= self.median:
                    if self.has_rleaf():
                        od_ball = self.rleaf.search(target, k, od_ball)
                else:
                    if self.has_lleaf():
                        od_ball = self.lleaf.search(target, k, od_ball)
        return od_ball