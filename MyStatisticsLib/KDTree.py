import numpy as np
from collections import OrderedDict


class MinKOrderedDict(OrderedDict):
    """
    MinKOrderedDict is a utility class to save the nearest K neighbors
    User can add newly found neighbor into MinKOrderedDict freely. When the number of neighbor is greater than K,
    it will pop the farest neighbor first. The number of neighbor is always equal or less than K.
        key of the dict is distance between a neighbor and target. Each key represent a found super ball. Radius of ball is
            equals to value of key
        value of the dict is a list of neighbors whose is exactly distance "key" away from target. These neighbors are on
            the same super ball (belong to same key).
    """
    # k: the number of nearest neighbor to be saved in the dictionary
    # capacity: the current capacity of dictionary, default value is 0
    def __init__(self, k):
        self.k = k
        self.capacity = 0
        OrderedDict.__init__(self)

    # return capacity
    def getCapacity(self):
        return self.capacity

    # return max key
    # The radius of the biggest super ball equals to MaxKey.
    # There are "capacity(<=K)" points in this biggest super ball.
    def getMaxKey(self):
        sorted_keys = sorted(self.keys())
        return sorted_keys[-1]

    # override the method of setting an item
    def __setitem__(self, key, value):
        containsKey = 1 if key in self else 0
        if containsKey:
            # when key already exists, it means we already find other neighbors who is distance "key" away from target
            # So, append the new value to the list of neighbor.
            item = OrderedDict.__getitem__(self, key)
            item.append(value)
            OrderedDict.__setitem__(self, key, item)
        else:
            # when key doesn't exist, it means this is the first neighbor who is distance "key" away from target
            # So add a new key and value map.
            OrderedDict.__setitem__(self, key, [value])
        # increase capacity
        self.capacity += 1
        # check capacity value with k
        if self.capacity > self.k:
            # capacity is greater than k, we need to remove 1 farest neighbor
            # sort dict by key/distance
            sorted_keys = sorted(self.keys())
            # get the farest neighbor
            item = OrderedDict.__getitem__(self, sorted_keys[-1])
            if len(item) > 1:
                # if more than 1 neighbor share same distance, we just remove 1 of them from list
                item.pop()
            else:
                # otherwise, remove the item
                OrderedDict.__delitem__(self, sorted_keys[-1])
            # decrease capacity
            self.capacity -=1

# KDTree
class KDTree:
    # depth is the depth of tree node
    def __init__(self, depth=0, parent=None):
        self.depth = depth
        self.parent = parent
        self.split = -1
        self.median = -1
        self.data = None
        self.lleaf = None
        self.rleaf = None

    #  data is a m X n matrix. Each row is a separate data. Each column is 1 dimension of data. Type is numpy.array
    #  m is the number of data
    #  n is the dimension of each data
    #  return NULL
    #  example  data = [ [1,2,3], [4,5,6],[7,8,9] ], in which [1,2,3] is a data, [4,5,6] and [7,8,9] are others.
    def build(self, data):
        if self.depth == 0:
            # save the original data in root node
            self.orig_data = data
            # when enter the root node, we add a new column at the end of data
            # which mark the idx of data in its original order
            m = len(data)
            orig_idx = np.arange(0,m)
            orig_idx.shape = (m,1)
            data = np.concatenate((data, orig_idx), axis=1)

        # get the dimension of data, i.e. the size of axis[1], minus 1 for the original index column
        n = data.shape[1] - 1
        # in the example we use depth%n as split dimension
        self.split = self.depth % n

        #  sort all lines(data) by the split'th dimension
        idx = np.argsort(data[:, self.split])
        data = data[idx]

        #  get median of the split'th dimension
        #  because split'th dimension is sorted, so the index of median is in the middle
        median_idx = int(len(data)/2)   # the index of median, start from 0
        self.median = data[median_idx, self.split]

        #  get all the points whose split'th dimension value are equal to median
        midx = np.where(data[:, self.split] == self.median)
        if np.size(midx) > 0:
            #  save those points to this tree node
            self.data = data[midx]
        #  get all the points whose split'th dimension value are bigger than median
        midx = np.where(data[:, self.split] > self.median)
        if np.size(midx) > 0:
            #  save those points in the right child
            self.rleaf = KDTree(self.depth + 1, self)
            self.rleaf.build(data[midx])
        #  get all the points whose split'th dimension value are less than median
        midx = np.where(data[:, self.split] < self.median)
        if np.size(midx) > 0:
            #  save those points in the left child
            self.lleaf = KDTree(self.depth + 1, self)
            self.lleaf.build(data[midx])

    def has_lleaf(self):
        return self.lleaf != None

    def has_rleaf(self):
        return self.rleaf != None

    #  search the nearest neighbor of target
    #  target: target data to be searched, type is numpy.array
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

    #  search the nearest k neighbor of target
    #  target: target data to be searched, type is numpy.array
    #  k: number of neighbor to be searched
    #  od_ball: dict of super balls, ordered by the key(=radius)
    def search(self, target, k=1, od_ball=None):
        # calculate the distance between target and points on the split surface of current node
        # return tuple of min distance and nearest point, (min distance, nearest point)
        def distance(target, node_data):
            data = node_data[:, :-1]  # remove the original index column
            squared_difference = np.square(target - data)
            squared_difference_sum = np.sum(squared_difference, axis=1)
            distance = np.sqrt(squared_difference_sum)
            return distance

        if self.depth ==0:
            od_ball = MinKOrderedDict(k)

            m = len(self.orig_data)  # number of points in KD tree
            if k >= m:
                # add all KD tree points into od_ball
                # radius doesn't has meaning in this case, set them to range(1,m)
                for i in range(0,m):
                    od_ball[i] = self.orig_data[i]
                return od_ball



        # get distance of points on the super-surface of current node
        # the distance is radius of super balls
        node_radius = distance(target, self.data)
        for i in range(0, len(self.data)):
            od_ball[node_radius[i]] = self.data[i]

        #  depth first search in KD tree
        #  self.split is the dimension by which child tree nodes are split
        if target[self.split] <= self.median:
            # if target is less than median on the split dimension, search left child
            if self.has_lleaf():
                od_ball = self.lleaf.search(target, k, od_ball)
        else:
            # if target is bigger than median on the split dimension, search right child
            if self.has_rleaf():
                od_ball = self.rleaf.search(target, k, od_ball)

        # for a non-leaf node, we need to judge if their children contains a nearest neighbor
        if self.has_lleaf() or self.has_rleaf():
            # first, only when biggest super-ball intersects with split super-surface of current node
            # then the node's children will probably contains nearest neighbor
            #       D = distance between center of ball and super-surface = "np.abs(target[self.split] - self.median)"
            #       R = radius of ball = min_dist[0]
            #       if ( D < R ) then super-surface intersects with the ball
            # second, because we already search on child of current node during depth first search, now it's time
            # to try the other node
            if od_ball.capacity < k or np.abs(target[self.split] - self.median) < od_ball.getMaxKey():
                # search the other child that didn't tried during depth first search
                if target[self.split] <= self.median:
                    # the same condition is used here as depth first search,
                    # but this time search the right leaf
                    if self.has_rleaf():
                        od_ball = self.rleaf.search(target, k, od_ball)
                else:
                    if self.has_lleaf():
                        od_ball = self.lleaf.search(target, k, od_ball)

        return od_ball