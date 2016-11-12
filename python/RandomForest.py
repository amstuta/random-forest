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
    from sklearn.model_selection import train_test_split

    data = CSVReader.read_csv("../scala/data/income.csv")
    train, test = train_test_split(data, test_size=0.3)

    rf = RandomForestClassifier(nb_trees=60, nb_samples=1000)
    rf.fit(train)

    errors = 0
    features = [ft[:-1] for ft in test]
    values = [ft[-1] for ft in test]

    for feature, value in zip(features, values):
        prediction = rf.predict(feature)
        if prediction != value:
            errors += 1

    logging.info("Error rate: {}".format(errors / len(features) * 100))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    test_rf()
