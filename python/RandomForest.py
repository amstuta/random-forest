#!/usr/bin/env python3

import logging
import random

from DecisionTree import DecisionTreeClassifier


"""
TODO: - add arguments to tree (max_depth, criterion)
      - add same arguments to random forest
      - implement a thread pool to fit multiple trees at the same time
      - add class CSVReader
      - comment functions
      - unittests
"""


class RandomForestClassifier(object):

    def __init__(self, nb_trees, nb_samples):
        self.trees = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        

    def fit(self, data):
        for i in range(self.nb_trees):
            logging.info('Training tree {}'.format(i + 1))

            random_features = random.sample(data, self.nb_samples)
            tree = DecisionTreeClassifier()
            tree.fit(random_features)

            self.trees.append(tree)


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    
    rf = RandomForestClassifier(30, 600)
    #rf.fit(None)
