#!/usr/bin/env python3

import CSVReader
from math import log


class DecisionTreeClassifier:


    class DecisionNode:
        def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
            self.col = col
            self.value = value
            self.results = results
            self.tb = tb
            self.fb = fb


    def __init__(self):
        self.rootNode = None


    def fit(self, rows, criterion=None):
        if not criterion: criterion = self.entropy
        self.rootNode = self.buildTree(rows, criterion)


    def predict(self, observation):
        return self.classify(observation, self.rootNode)


    def divideSet(self, rows, column, value):
       split_function = None
       if isinstance(value, int) or isinstance(value, float):
          split_function = lambda row:row[column] >= value
       else:
          split_function = lambda row:row[column] == value

       set1 = [row for row in rows if split_function(row)]
       set2 = [row for row in rows if not split_function(row)]
       return set1, set2


    def uniqueCounts(self, rows):
        results = {}
        for row in rows:
            r = row[len(row) - 1]
            if r not in results: results[r] = 0
            results[r] += 1
        return results


    def entropy(self, rows):
       log2 = lambda x:log(x) / log(2)
       results = self.uniqueCounts(rows)
       ent = 0.0
       for r in results.keys():
          p = float(results[r]) / len(rows)
          ent = ent - p * log2(p)
       return ent


    def buildTree(self, rows, scoref):
        if len(rows) == 0: return self.DecisionNode()
        current_score = scoref(rows)

        best_gain = 0.0
        best_criteria = None
        best_sets = None
        column_count = len(rows[0]) - 1

        for col in range(0, column_count):
            column_values = {}
            for row in rows:
                column_values[row[col]] = 1
            for value in column_values.keys():
                set1, set2 = self.divideSet(rows, col, value)

                p = float(len(set1)) / len(rows)
                gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
                if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = (set1, set2)

        if best_gain > 0:
            trueBranch = self.buildTree(best_sets[0], scoref)
            falseBranch = self.buildTree(best_sets[1], scoref)
            return self.DecisionNode(col=best_criteria[0],value=best_criteria[1],
                                    tb=trueBranch,fb=falseBranch)
        else:
            return self.DecisionNode(results=self.uniqueCounts(rows))


    def classify(self, observation, tree):
        if tree.results != None:
            return tree.results
        else:
            v = observation[tree.col]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value: branch = tree.tb
                else: branch = tree.fb
            else:
                if v == tree.value: branch = tree.tb
                else: branch = tree.fb
            return self.classify(observation, branch)



def test_tree():
    data = CSVReader.read_csv("../scala/data/income.csv")
    tree = DecisionTreeClassifier()
    tree.fit(data)

    print(tree.predict([39, 'State-gov', 'Bachelors', 13, 'Never-married', \
                        'Adm-clerical', 'Not-in-family', 'White', 'Male', \
                        2174, 0, 40, 'United-States']))


if __name__=='__main__':
    test_tree()
