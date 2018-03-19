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
import tensorflow as tf
import matplotlib.pyplot as plt
import random

from random import choice

from sklearn.cross_validation import StratifiedShuffleSplit

from classifiers.galaxy_classifiers.decision_tree_classifier import TreeClassifier

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from classifiers.galaxy_classifiers.knn_classifier import KNNClassifier
from classifiers.galaxy_classifiers.gaussian_naive_bayes_classifier import GaussianNaiveBayesClassifier
from classifiers.galaxy_classifiers.multinomial_naive_bayes_classifier import MultinomialNaiveBayesClassifier

from commons.helpers.dataset.context import Context
from commons.helpers.dataset.strategies.spam_dataset.feature_strategy import SpamDataSetFeatureStrategy
from commons.helpers.dataset.strategies.galaxy_dataset.feature_strategy import GalaxyDataSetFeatureStrategy

from commons.preprocessors.discretization.context import DiscretizerContext
from commons.preprocessors.discretization.strategies.unsupervised.unsupervised_discretization_strategy import \
    UnsupervisedDiscretizationStrategy
from commons.preprocessors.discretization.strategies.supervised.supervised_discretization_strategy import \
    SupervisedDiscretizationStrategy

from classifiers.galaxy_classifiers.mlp_tensorboard import MLPClassifierTensorBoard

from sklearn.model_selection import GridSearchCV
from classifiers.galaxy_classifiers.linear_svm_classifier import LinearSVMClassifier
from classifiers.galaxy_classifiers.rbf_svm_classifier import SVMClassifier

def get_galaxy_dataset(validation_size):

    stategy = GalaxyDataSetFeatureStrategy()
    csv_file = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/galaxy_feature_vectors.csv"
    context = Context(stategy)
    dataset = context.load_dataset(csv_file=csv_file, one_hot=False, validation_size=np.float32(validation_size))

    return dataset

def get_linear_svm_classifier(X_train, y_train, C, class_weight):
    """ get the decision tree classifier

    Use scikit-learn methods to get a linear svm classifier

    Args:
        X_train: The training values
        y_train: Training labels
        max_depth: maximum depth for the tree

    Returns:
        The Classifier.
    """

    clf = LinearSVMClassifier(C, class_weight)
    clf.train(X_train, y_train)

    return clf

def get_rbf_svm_classifier(X_train, y_train, C, gamma, kernel):
    """ get the decision tree classifier

    Use scikit-learn methods to get a linear svm classifier

    Args:
        X_train: The training values
        y_train: Training labels
        max_depth: maximum depth for the tree

    Returns:
        The Classifier.
    """

    clf = SVMClassifier(C, gamma, kernel)
    clf.train(X_train, y_train)

    return clf

def get_linear_svm_results(c_params, X_train, y_train, X_test, y_test):
    """ return decision tree results

    compute models accuracy with scikit-learn functions

    Args:
        tree_params: tree classifier params
        X_train: train data
        y_train: train label
        X_test: valid data
        y_test: valid label
    Returns:
        results
    """
    # ------------------------ linear SVM ------------------------
    results = list()
    class_weight = 'balanced'

    for c_param in c_params:

        trained_classifier = get_linear_svm_classifier(X_train, y_train, c_param, class_weight)
        params = "c=" + str(c_param)

        y_pred = trained_classifier.predict(X_test)
        y_true = y_test

        score_result = trained_classifier.score(X_test, y_test)
        f1_score_result = f1_score(y_true, y_pred, average='weighted')

        results.append([params, score_result, f1_score_result])

    print("LINEAR SVM: ", results)
    return results

def get_rbf_svm_results(c_params, g_params, X_train, y_train, X_test, y_test):

    results = list()
    class_weight = {'balanced'}

    for c_param in c_params:
        for g_param in g_params:
            trained_classifier = get_rbf_svm_classifier(X_train, y_train, c_param, g_param, 'rbf')
            params = "c=" + str(c_param) + ";gamma=" + str(g_param)

            y_pred = trained_classifier.predict(X_test)
            y_true = y_test

            score_result = trained_classifier.score(X_test, y_test)
            f1_score_result = f1_score(y_true, y_pred, average='weighted')

            results.append([params, score_result, f1_score_result])

    print("RBF SVM: ", results)
    return results

def main():

    galaxy_dataset = get_galaxy_dataset(0.2)
    galaxy_X_train = galaxy_dataset.train.get_features
    galaxy_y_train = galaxy_dataset.train.get_labels
    galaxy_X_test = galaxy_dataset.valid.get_features
    galaxy_y_test = galaxy_dataset.valid.get_labels

    # train_path = os.environ["VIRTUAL_ENV"] + "/data"
    C = [0.001, 0.1, 1.0, 10.0]
    gamma = [0.001, 0.1, 1.0, 10.0]
    # linear_svm_results = get_linear_svm_results(C, galaxy_X_train, galaxy_y_train, galaxy_X_test, galaxy_y_test)

    rbf_svm_results = get_rbf_svm_results(C, gamma, galaxy_X_train, galaxy_y_train, galaxy_X_test, galaxy_y_test)

    param_grid = dict(gamma=gamma, C=C)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVMClassifier(C, gamma, 'rbf'), param_grid=param_grid, cv=cv, n_jobs=8, cache_size=2048)
    grid.fit(galaxy_X_train, galaxy_y_train)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))


if __name__ == '__main__':
    main()