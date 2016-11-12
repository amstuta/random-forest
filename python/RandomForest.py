#!/usr/bin/env python3

import logging
import random

import CSVReader
from DecisionTree import DecisionTreeClassifier


"""
TODO: - add arguments to tree (max_depth, criterion)
      - add same arguments to random forest
      - implement a thread pool to fit multiple trees at the same time
      - comment functions
      - set code to PEP8 norm
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


    def predict(self, feature):
        predictions = []

        for tree in self.trees:
            predictions.append(tree.predict(feature))

        return max(set(predictions), key=predictions.count)


def test_rf():
    # random_split the features & log error rate
    pass


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)

    data = CSVReader.read_csv("../scala/data/income.csv")
    rf = RandomForestClassifier(30, 1000)
    rf.fit(data)

    print(rf.predict([39, 'State-gov', 'Bachelors', 13, 'Never-married', \
                    'Adm-clerical', 'Not-in-family', 'White', 'Male', \
                    2174, 0, 40, 'United-States']))
