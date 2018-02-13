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
from sklearn import tree
from sklearn.model_selection import train_test_split
from commons.helpers.dataset.context import Context
from commons.helpers.dataset.strategies.spam_dataset.feature_strategy import SpamDataSetFeatureStrategy

def extract_smaller_size_of_dataset(dataset, ratio):
    """ get smaller size of the dataset according to a ratio

    Args:
        dataset: dataset to extract smaller size of
        ratio: ratio of the smaller size dataset

    Returns:
        The smaller sized dataset according to the ratio
    """
    dataset
    return features_with_noise


def apply_noise_to_features(dataset, noise):
    """ get dataset with a percentage of noise

    Add a randomize value to apply noise to the dataset with
    mu, sigma = 0, 0.10

    https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python#14058425

    Args:
        dataset: dataset to apply noise to
        noise: noise to apply to dataset

    Returns:
        The dataset with a percentage of noise
    """
    mu, sigma = 0, 0.10
    noise_value = np.random.normal(mu, sigma, [dataset.shape[0], dataset.shape[1]])
    features_with_noise = dataset + noise_value
    return features_with_noise

def get_decision_tree_score(X_train, X_test, y_train, y_test, max_depth=None):
    """ get the decision tree score

    Use scikit-learn methods to compute a decision tree score

    Args:
        X_train: The training values
        X_test: The test values
        y_train: Training labels
        y_test: Test labels
        max_depth: maximum depth for the tree

    Returns:
        The score.
    """

    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    clf = clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    if max_depth is None:
        print("max depth: None")
    else:
        print("max depth: " + str(max_depth))

    print(score)

    return score

def main():

    # The desired validation size.
    validation_size = 0.2

    spam_feature_dataset_strategy = SpamDataSetFeatureStrategy()
    context = Context(spam_feature_dataset_strategy)

    spam_feature_csv_file = os.environ["VIRTUAL_ENV"] + "/data/csv/spam/spam.csv"


    spam_feature_dataset = context.load_dataset(csv_file=spam_feature_csv_file, one_hot=False,
                                                validation_size=np.float32(validation_size))

    noises_array = [0, 0.05, 0.10, 0.20]

    for i in noises_array:
        noise = i
        print("noise: " + str(noise))
        spam_train_features = apply_noise_to_features(spam_feature_dataset.train.get_features, noise)
        spam_train_labels = spam_feature_dataset.train.get_labels

        spam_valid_features = apply_noise_to_features(spam_feature_dataset.valid.get_features, noise)
        spam_valid_labels = spam_feature_dataset.valid.get_labels

        get_decision_tree_score(spam_train_features, spam_valid_features, spam_train_labels, spam_valid_labels,
                                max_depth=None)
        get_decision_tree_score(spam_train_features, spam_valid_features, spam_train_labels, spam_valid_labels,
                                max_depth=3)
        get_decision_tree_score(spam_train_features, spam_valid_features, spam_train_labels, spam_valid_labels,
                                max_depth=5)
        get_decision_tree_score(spam_train_features, spam_valid_features, spam_train_labels, spam_valid_labels,
                                max_depth=10)
        print("___________________________________")

    print("hello")

if __name__ == '__main__':
    main()