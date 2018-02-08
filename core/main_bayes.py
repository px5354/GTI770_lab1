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

Notes : This file is to generate everything we want from feature vectors computed in the main file.
        This file will generate classify the galaxies with the feature vectors
        It will also generate a plot for comparing different features and
        generate a plot of the tree decision surface of the different features
"""

import os
import numpy as np
from commons.helpers.graphics.plot import Plot
from sklearn import tree
from sklearn.model_selection import train_test_split

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
    galaxy_feature1_vector_path = os.environ[
                                      "VIRTUAL_ENV"] + "/data/csv/galaxy/exported_personal_feature1.csv"
    galaxy_feature2_vector_path = os.environ[
                                      "VIRTUAL_ENV"] + "/data/csv/galaxy/exported_personal_feature3.csv"
    galaxy_feature3_vector_path = os.environ[
                                      "VIRTUAL_ENV"] + "/data/csv/galaxy/exported_personal_feature4.csv"
    galaxy_feature4_vector_path = os.environ[
                                      "VIRTUAL_ENV"] + "/data/csv/galaxy/exported_personal_feature2.csv"
    galaxy_label_path = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/exported_personal_label_galaxy.csv"

    result_feat_comparison_path = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/feature_comparison_"

    result_tree_path = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/decision_tree.png"

    features1 = np.loadtxt(open(galaxy_feature1_vector_path, "rb"), delimiter=",", skiprows=0)
    features2 = np.loadtxt(open(galaxy_feature2_vector_path, "rb"), delimiter=",", skiprows=0)
    features3 = np.loadtxt(open(galaxy_feature3_vector_path, "rb"), delimiter=",", skiprows=0)
    features4 = np.loadtxt(open(galaxy_feature4_vector_path, "rb"), delimiter=",", skiprows=0)

    features = [features1, features2, features3, features4]
    labels = np.loadtxt(open(galaxy_label_path, "rb"), delimiter=",", skiprows=0)

    X = np.vstack((features1, features2, features3, features4)).T
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 0)

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