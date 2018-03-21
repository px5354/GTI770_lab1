#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Course :
    GTI770 — Systèmes intelligents et apprentissage machine

Project :
    Lab # 1 - Extraction de primitives

Students :
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

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC

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
from sklearn.svm import LinearSVC

def get_galaxy_dataset(validation_size):

    stategy = GalaxyDataSetFeatureStrategy()
    csv_file = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/galaxy_feature_vectors.csv"
    context = Context(stategy)
    dataset = context.load_dataset(csv_file=csv_file, one_hot=False, validation_size=np.float32(validation_size))
    return dataset

def get_specific_features(X_train, X_test, features_indexes):

    filtered_X_train = X_train[:, features_indexes]
    filtered_X_test = X_test[:, features_indexes]

    return filtered_X_train, filtered_X_test

def get_best_params_for_model(param_grid, model, train_X, train_y):

    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(model, param_grid=param_grid, cv=cv, n_jobs=8)
    grid.fit(train_X, train_y)
    return grid

def train_set_with_size(dataSet, proportion, state):
    """ get dataset with a certain size

    Use scikit-learn methods to get a certain size dataset

    Args:
        dataSet: dataset to split
        proportion: the proportion of the new dataset
        state: the random state for splitting
    Returns:
        New splitted dataset.
    """

    if (proportion == 1):
        features = dataSet.valid.get_features
        labels = dataSet.valid.get_labels
    else:
        _, features, _, labels = train_test_split(dataSet.valid.get_features, dataSet.valid.get_labels, test_size=proportion, random_state=state)

    return features, labels

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
    mu = 0
    sigma = noise
    noise_value = np.random.normal(mu, sigma, [dataset.shape[0], dataset.shape[1]])
    features_with_noise = dataset + noise_value
    return features_with_noise

def split_data_for_k_fold(X, y, n_splits=10):
    """ splits data for k fold cross validation

    Use scikit-learn methods to splits data for k fold cross validation

    Args:
        X: features
        y: labels
        n_splits: parameters for k fold cross validation
    Returns:
        training data, training labels, valid data, valid labels
    """
    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return X_train, y_train, X_test, y_test

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

    clf = SVMClassifier(C, gamma, kernel, cache_size=2048)
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

    # print("LINEAR SVM: ", results)
    return results

def get_rbf_svm_results(c_params, g_params, X_train, y_train, X_test, y_test):

    results = list()

    for c_param in c_params:
        for g_param in g_params:
            trained_classifier = get_rbf_svm_classifier(X_train, y_train, c_param, g_param, 'rbf')
            params = "c=" + str(c_param) + ";gamma=" + str(g_param)

            y_pred = trained_classifier.predict(X_test)
            y_true = y_test

            score_result = trained_classifier.score(X_test, y_test)
            f1_score_result = f1_score(y_true, y_pred, average='weighted')

            results.append([params, score_result, f1_score_result])

    # print("RBF SVM: ", results)
    return results

def main():
    validation_size = 0.20
    features_indexes = [3, 4, 5, 18, 22, 23]
    cross_validation_size = 1
    print("VALIDATION SIZE: ", str(validation_size))

    galaxy_dataset = get_galaxy_dataset(validation_size)

    # --------------------------------- VALIDATION METHOD --------------------------------------------

    # HOLD OUT
    # X_train, X_test = get_specific_features(galaxy_dataset.train.get_features, galaxy_dataset.valid.get_features, features_indexes)
    # y_train = galaxy_dataset.train.get_labels
    # y_test = galaxy_dataset.valid.get_labels

    # K FOLD CV
    # all_galaxy_dataset = get_galaxy_dataset(cross_validation_size)
    # X_train, y_train, X_test, y_test = split_data_for_k_fold(all_galaxy_dataset.valid.get_features,all_galaxy_dataset.valid.get_labels)
    # X_train, X_test = get_specific_features(X_train, X_test, features_indexes)

    # --------------------------------- END VALIDATION METHOD --------------------------------------------

    # C = [0.001, 0.1, 1.0, 10.0]
    # gamma = [0.001, 0.1, 1.0, 10.0]
    # linear_svm_results = get_linear_svm_results(C, X_train, y_train, X_test, y_test)
    #
    # rbf_svm_results = get_rbf_svm_results(C, gamma, X_train, y_train, X_test, y_test)
    #
    # print("LINEAR SVM: ", linear_svm_results)
    # print("RBF SVM: ", rbf_svm_results)
    # print("___________________________________________________________")

    #-----------------------------START code for getting the best parameters--------------------------------------------
    # param_grid = dict(gamma=gamma, C=C)
    # rbf_grid = get_best_params_for_model(param_grid, SVC(cache_size=2048), galaxy_X_train, galaxy_y_train)
    #
    # print("The best parameters for rbf svm are %s with a score of %0.2f"
    #       % (rbf_grid.best_params_, rbf_grid.best_score_))

    # param_grid = dict(C=C)
    # linear_svm_grid = get_best_params_for_model(param_grid, LinearSVC(), galaxy_X_train, galaxy_y_train)
    #
    # print("The best parameters for linear svm are %s with a score of %0.2f"
    #       % (linear_svm_grid.best_params_, linear_svm_grid.best_score_))
    # -----------------------------END code for getting the best parameters---------------------------------------------

    # -------------------- Apply noise to best models --------------------
    C_linear = [1.0]
    C_rbf = [10.0]
    gamma_rbf = [0.1]
    noises = [0, 0.05, 0.10, 0.20]
    proportions = [0.25, 0.5, 0.75, 1]
    state = 1

    for proportion in proportions:
        for noise in noises:
            print("noise: " + str(noise))
            print("proportion: " + str(proportion))

            features, labels = train_set_with_size(galaxy_dataset, proportion, state)

            #--------------------------------- VALIDATION METHOD --------------------------------------------

            # HOLD OUT
            # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=validation_size, random_state=state)
            # X_train, X_test = get_specific_features(X_train, X_test, features_indexes)

            # K FOLD CV
            all_galaxy_dataset = get_galaxy_dataset(cross_validation_size)
            X_train, y_train, X_test, y_test = split_data_for_k_fold(all_galaxy_dataset.valid.get_features,
                                                                     all_galaxy_dataset.valid.get_labels)
            X_train, X_test = get_specific_features(X_train, X_test, features_indexes)

            # --------------------------------- END VALIDATION METHOD ----------------------------------------

            X_train_with_noise = apply_noise_to_features(X_train, noise)

            noise_linear_svm_results = get_linear_svm_results(C_linear, X_train_with_noise, y_train, X_test, y_test)
            noise_rbf_svm_results = get_rbf_svm_results(C_rbf, gamma_rbf, X_train_with_noise, y_train, X_test, y_test)


            print("NOISE LINEAR SVM: ", noise_linear_svm_results)
            print("NOISE RBF SVM: ", noise_rbf_svm_results)
            print("___________________________________________________________")



if __name__ == '__main__':
    main()