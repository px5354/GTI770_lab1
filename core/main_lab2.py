#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Course :
    GTI770 — Systèmes intelligents et apprentissage machine

Project :
    Lab # 1 - Extraction de primitives

Students :
    ARRON VUONG     -   VUOA09109300
    PHILIPPE LE     -   LEXP12119302
    SAMUEL GERVAIS  -   GERS04029200

Group :
    GTI770-H18-02

Notes : This file is the main program file. Please note that this project might be over-commented compared to a real,
        industry-class framework.
        This framework is used as a part of the GTI770 course at Ecole de technologie superieure of Montreal. The amount
        of comments is only to make sure everyone from all level can clearly understand the code given to complete
        the class assignment.
"""

import os
import numpy as np
import timeit

from commons.helpers.graphics.plot import Plot
from sklearn import tree
from sklearn.model_selection import train_test_split
from classifiers.galaxy_classifiers.decision_tree_classifier import TreeClassifier
from classifiers.galaxy_classifiers.knn_classifier import KNNClassifier
from classifiers.galaxy_classifiers.gaussian_naive_bayes_classifier import GaussianNaiveBayesClassifier


def get_decision_tree_score(X_train, X_test, y_train, y_test, max_depth=None, noise=0):
    """ get the decision tree score

    Use scikit-learn methods to compute a decision tree score

    Args:
        X_train: The training values
        X_test: The test values
        y_train: Training labels
        y_test: Test labels
        max_depth: maximum depth for the tree
        noise: noise to apply to training set

    Returns:
        The score.
    """

    if noise != 0:
        noise_value = np.random.normal(0, 0.10, [X_test.shape[0], X_test.shape[1]])
        X_test = X_test + noise_value

    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    clf = clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    if max_depth is None:
        print("max depth: None")
    else:
        print("max depth: " + str(max_depth))

    print("noise: " + str(noise))
    print(score)

    return score

def main():
    """
        Program's entry point.
    """
    start = timeit.default_timer()

    # The desired validation size.
    validation_size = 0.2

    environ_path = os.environ["VIRTUAL_ENV"]
    spam_csv_file = environ_path + "/data/csv/spam/spam.csv"


    knn = KNNClassifier()
    bayes = GaussianNaiveBayesClassifier()
    tree = TreeClassifier()

    feature = np.loadtxt(open(spam_csv_file, "rb"), delimiter=",", skiprows=0)

    features = [feature]

    x = np.vstack(feature).T
    y = np.vstack(feature).T


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    X = np.vstack((features1, features2, features3, features4)).T
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    get_decision_tree_score(X_train, X_test, y_train, y_test, max_depth=None)
    get_decision_tree_score(X_train, X_test, y_train, y_test, max_depth=2)
    get_decision_tree_score(X_train, X_test, y_train, y_test, max_depth=3)
    get_decision_tree_score(X_train, X_test, y_train, y_test, max_depth=4)
    get_decision_tree_score(X_train, X_test, y_train, y_test, max_depth=5)
    get_decision_tree_score(X_train, X_test, y_train, y_test, max_depth=10)

    get_decision_tree_score(X_train, X_test, y_train, y_test, max_depth=5, noise=0.05)
    get_decision_tree_score(X_train, X_test, y_train, y_test, max_depth=5, noise=0.10)
    get_decision_tree_score(X_train, X_test, y_train, y_test, max_depth=5, noise=0.15)

    # Plot.plot_tree_decision_surface(X, labels, ["RB_ratio", "entropy", "gini", "light radius diff"],
    #                                 ["RB_ratio", "entropy", "gini", "light radius diff"], result_tree_path)

    for i in range(0, 3):
        Plot.plot_feature_comparison(features[i], features[i], labels, result_feat_comparison_path + str(i) + ".png")


if __name__ == '__main__':
    main()