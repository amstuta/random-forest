## Implementation of Decision Tree and Random Forest classifiers in Python and Scala languages

### Python
- Both classifiers use Python3 and don't need any third-party library
- The decision tree example can be launched by running:
```sh
cd python
./DecisionTree.py
```
- The random forest example cam be launched the same way:
```sh
cd python
./RandomForest.py
```

### Scala
The scala version contains a main function running the same example using the decision tree and the random forest classifier.
It can be launched like this:
```sh
cd scala
sbt run
```

The error rate of the two algorithms will be printed out.

### Dataset

The dataset used in the examples is the "Census Income" dataset.
It classifies incomes of approximately 32000 people into two categories: above and below 50,000$ a year.

This dataset is distributed by the UCI machine learning repository and can be found here: http://archive.ics.uci.edu/ml/datasets/Adult.
