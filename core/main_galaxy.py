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
from classifiers.galaxy_classifiers.decision_tree_classifier import TreeClassifier
from sklearn.model_selection import KFold
import random
from random import choice
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
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score


def plot_hyper_parameters_comparison(params_array, results, title, xlabel_name, filename, results2 = None):
    """ save plot bar of parameters comparison

    Use matplotlib methods to compute a decision tree score

    Args:
        results: results to compare
        title: plot title
        xlabel_name: xlabel name
        filename: filename

    """
    x = params_array
    y_score = list()
    y_f1_score = list()
    for result in results:
        y_score.append(result[1])
        y_f1_score.append(result[2])

    # Set graphics properties
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=14)
    plt.xticks(x, x)
    ax.set_xlabel(xlabel_name, fontsize=12)
    ax.set_ylabel("score", fontsize=12)
    ax.grid(True, linestyle='-', color='0.75')

    # Plot
    ax.margins(0.05)

    if results2 is None :
        ax.plot(x, y_score, marker='o', linestyle='-', ms=10, label="score")
        ax.plot(x, y_f1_score, marker='o', linestyle='-', ms=10, label="f1_score")
    else:
        y2_score = list()
        y2_f1_score = list()
        result_name = results[0][0].split(";weights=")[1]
        result2_name = results2[0][0].split(";weights=")[1]
        for result in results2:
            y2_score.append(result[1])
            y2_f1_score.append(result[2])
        ax.plot(x, y_score, marker='o', linestyle='-', ms=10, label="score_" + result_name)
        ax.plot(x, y_f1_score, marker='o', linestyle='-', ms=10, label="f1_score_" + result_name)
        ax.plot(x, y2_score, marker='o', linestyle='-', ms=10, label="score_" + result2_name)
        ax.plot(x, y2_f1_score, marker='o', linestyle='-', ms=10, label="f1_score_" + result2_name)


    ax.legend()
    plt.savefig(filename)

def plot_bar_hyper_parameters_comparison(results, title, xlabel_name, filename):
    """ save plot bar of parameters comparison

    Use matplotlib methods to compute a decision tree score

    Args:
        results: results to compare
        title: plot title
        xlabel_name: xlabel name
        filename: filename

    """
    x = list()
    y_score = list()
    y_f1_score = list()
    for result in results:
        x.append(result[0])
        y_score.append(result[1])
        y_f1_score.append(result[2])

    # Set graphics properties
    fig, ax = plt.subplots(figsize=(10, 6))
    index = np.arange(len(results))
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, y_score, bar_width,
                     alpha=opacity,
                     color='b',
                     label='score')

    rects2 = plt.bar(index + bar_width, y_f1_score, bar_width,
                     alpha=opacity,
                     color='g',
                     label='f1_score')

    plt.xlabel(xlabel_name)
    plt.ylabel('score')
    plt.title(title)
    plt.xticks(index + bar_width, x)
    plt.tick_params(axis='x', which='major', labelsize=6)
    plt.tick_params(axis='both', which='minor', labelsize=5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)

# def extract_smaller_size_of_dataset(dataset, ratio):
#     """ get smaller size of the dataset according to a ratio
#
#     Args:
#         dataset: dataset to extract smaller size of
#         ratio: ratio of the smaller size dataset
#
#     Returns:
#         The smaller sized dataset according to the ratio
#     """
#     dataset
#     return features_with_noise


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

def get_decision_tree(X_train, y_train, max_depth=None):
    """ get the decision tree classifier

    Use scikit-learn methods to compute a decision tree score

    Args:
        X_train: The training values
        y_train: Training labels
        max_depth: maximum depth for the tree

    Returns:
        The Classifier.
    """

    clf = TreeClassifier(max_depth=max_depth)
    clf.train(X_train, y_train)

    return clf

def get_knn(X_train, y_train, n_neighbors, weights):
    """ get the knn classifier

    Use scikit-learn methods to compute a decision tree score

    Args:
        X_train: The training values
        y_train: Training labels
        n_neighbors: number of neighbors
        weights: weight function

    Returns:
        The Classifier.
    """

    knn = KNNClassifier(n_neighbors, weights)
    knn.train(X_train, y_train)

    return knn

def get_gaussian_naive_bayes(X_train, y_train, priors=None):
    """ get the gaussian classifier

    Use scikit-learn methods to compute a decision tree score

    Args:
        X_train: The training values
        y_train: Training labels
        priors: Classes probabilities

    Returns:
        The Classifier.
    """

    gnb = GaussianNaiveBayesClassifier(priors=priors)
    gnb.train(X_train, y_train)

    return gnb

def get_multinomial_naive_bayes(X_train, y_train, fit_prior=False, class_prior=None):
    """ get the multinomial naive bayes classifier

    Use scikit-learn methods to compute a decision tree score

    Args:
        X_train: The training values
        y_train: Training labels
        fit_prior: Boolean to fit before or not
        class_priors: Classes probabilities

    Returns:
        The Classifier.
    """

    mnb = MultinomialNaiveBayesClassifier(fit_prior, class_prior)
    print(X_train)
    mnb.train(X_train, y_train)

    return mnb

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


def get_tree_results(tree_params, X_train, y_train, X_test, y_test):
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
    # ------------------------ TREE ------------------------
    results_tree = list()

    for max_depth in tree_params:

        if (max_depth == 0):
            trained_tree_classifier = get_decision_tree(X_train, y_train, max_depth=None)
            params = "max_depth=None"
        else:
            trained_tree_classifier = get_decision_tree(X_train, y_train, max_depth=max_depth)
            params = "max_depth=" + str(max_depth)

        y_pred = trained_tree_classifier.predict(X_test)
        y_true = y_test

        score_result = trained_tree_classifier.score(X_test, y_test)
        f1_score_result = f1_score(y_true, y_pred, average='weighted')

        results_tree.append([params, score_result, f1_score_result])

    print("TREE: ", results_tree)
    return results_tree

def get_knn_results(neighbors, weights, X_train, y_train, X_test, y_test):
    """ return knn results

    compute models accuracy with scikit-learn functions

    Args:
        neighbors: number of neighbors
        weights: weight function
        X_train: train data
        y_train: train label
        X_test: valid data
        y_test: valid label
    Returns:
        results
    """
    # ------------------------ KNN ------------------------
    results_knn = list()
    for neighbor in neighbors:
        for weight in weights:
            trained_knn_classifier = get_knn(X_train, y_train, neighbor, weight)
            params = "n_neighbors=" + str(neighbor) + ";weights=" + weight

            y_pred = trained_knn_classifier.predict(X_test)
            y_true = y_test

            score_result = trained_knn_classifier.score(X_test, y_test)
            f1_score_result = f1_score(y_true, y_pred, average='weighted')

            results_knn.append([params, score_result, f1_score_result])

    print("KNN: ", results_knn)
    return results_knn

def get_naive_bayes_results(multinomial_naive_bayes_datasets, class_prob, is_k_fold=False):
    """ return naive bayes results

    compute models accuracy with scikit-learn functions

    Args:
        multinomial_naive_bayes_datasets: datasets params
        class_prob: classes probabilities
        is_k_fold: if we apply k fold
    Returns:
        results
    """

    naive_bayes_results = list()
    normal_dataset = multinomial_naive_bayes_datasets[0][1]

    if is_k_fold:
        X_train, y_train, X_test, y_test = split_data_for_k_fold(normal_dataset.valid.get_features,
                                                                 normal_dataset.valid.get_labels)
    else :
        X_train = normal_dataset.train.get_features
        y_train = normal_dataset.train.get_labels
        X_test = normal_dataset.valid.get_features
        y_test = normal_dataset.valid.get_labels

    # Gaussian Naive Bayes
    naive_bayes_classifier = get_gaussian_naive_bayes(X_train, y_train, class_prob)

    y_pred = naive_bayes_classifier.predict(X_test)
    y_true = y_test

    gaussian_score_result = naive_bayes_classifier.score(X_test, y_test)
    gaussian_f1_score_result = f1_score(y_true, y_pred, average='weighted')

    # Multinomial Naive Bayes

    # for i, mnb_dataset in enumerate(multinomial_naive_bayes_datasets):
    #     dataset = mnb_dataset[1]
    #
    #     if is_k_fold:
    #         X_train, y_train, X_test, y_test = split_data_for_k_fold(dataset.valid.get_features,
    #                                                                  dataset.valid.get_labels)
    #     else:
    #         X_train = dataset.train.get_features
    #         y_train = dataset.train.get_labels
    #         X_test = dataset.valid.get_features
    #         y_test = dataset.valid.get_labels
    #
    #     if i == 0:
    #         naive_bayes_classifier = get_multinomial_naive_bayes(X_train, y_train, True, class_prob)
    #     else:
    #         naive_bayes_classifier = get_multinomial_naive_bayes(X_train, y_train)
    #
    #     y_pred = naive_bayes_classifier.predict(X_test)
    #     y_true = y_test
    #
    #     score_result = naive_bayes_classifier.score(X_test, y_test)
    #     f1_score_result = f1_score(y_true, y_pred, average='weighted')
    #
    #     if i == 0:
    #         naive_bayes_results.append(['multinomial', score_result, f1_score_result])
    #     else:
    #         naive_bayes_results.append(['multinomial_' + mnb_dataset[0], score_result, f1_score_result])

    naive_bayes_results.append(['gaussian', gaussian_score_result, gaussian_f1_score_result])
    print("NAIVE BAYES: ", naive_bayes_results)
    return naive_bayes_results

def main():

    # GALAXY
    # context = Context(galaxy_feature_dataset_strategy)
    # galaxy_dataset = context.load_dataset(csv_file=galaxy_feature_csv_file, one_hot=False,
    #                                     validation_size=np.float32(validation_size))
    # galaxy_dataset_train = galaxy_dataset.train
    # galaxy_dataset_valid = galaxy_dataset.valid

    # The desired validation size.
    validation_size = 0.2

    # Load data
    galaxy_feature_dataset_strategy = GalaxyDataSetFeatureStrategy()
    galaxy_feature_csv_file = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/galaxy_feature_vectors.csv"

    stategy = galaxy_feature_dataset_strategy
    file = galaxy_feature_csv_file

    context = Context(stategy)
    spam_dataset = context.load_dataset(csv_file=file, one_hot=False,
                                        validation_size=np.float32(validation_size))

    preprocessor_context = DiscretizerContext(SupervisedDiscretizationStrategy())

    supervised_discretised_dataset = preprocessor_context.discretize(data_set=spam_dataset,
                                                                     validation_size=np.float32(validation_size))

    preprocessor_context.set_strategy(UnsupervisedDiscretizationStrategy())

    unsupervised_discretised_dataset = preprocessor_context.discretize(data_set=spam_dataset,
                                                                       validation_size=np.float32(validation_size))

    spam_multinomial_datasets = [["spam_dataset", spam_dataset],
                                        ["supervised_discretised_dataset", supervised_discretised_dataset],
                                        ["unsupervised_discretised_dataset", unsupervised_discretised_dataset]]


    # -------------------- NORMAL VALIDATION --------------------
    spam_X_train = spam_dataset.train.get_features
    spam_y_train = spam_dataset.train.get_labels
    spam_X_test = spam_dataset.valid.get_features
    spam_y_test = spam_dataset.valid.get_labels
    spam_class_prob = [0.356, 0.62, 0.024]

    # params
    tree_params = [0, 3, 5, 10]
    neighbors_params = [3, 5, 10]
    weights_params = ['uniform', 'distance']

    # results
    tree_results = get_tree_results(tree_params, spam_X_train, spam_y_train, spam_X_test, spam_y_test)
    knn_results = get_knn_results(neighbors_params, weights_params, spam_X_train, spam_y_train, spam_X_test, spam_y_test)
    naive_bayes_results = get_naive_bayes_results(spam_multinomial_datasets, spam_class_prob)

    print("NORMAL TREE: ", tree_results)
    print("NORMAL KNN: ", knn_results)
    print("NORMAL NAIVE BAYES: ", naive_bayes_results)

    # Plot results
    results_knn_uniform = list()
    results_knn_distance = list()

    for result in knn_results:
        if result[0].split(";weights=")[1] == "uniform":
            results_knn_uniform.append(result)
        else:
            results_knn_distance.append(result)


    plot_hyper_parameters_comparison(tree_params, tree_results, "Decision Tree", "max_depth",
                                     os.environ["VIRTUAL_ENV"] + "/data/csv/spam/decision_tree_spam.png")

    plot_hyper_parameters_comparison(neighbors_params, results_knn_uniform, "KNN with different weights", "n_neighbors",
                                     os.environ["VIRTUAL_ENV"] + "/data/csv/spam/knn_spam.png", results_knn_distance)

    plot_bar_hyper_parameters_comparison(naive_bayes_results, "Naive Bayes with different params",
                                         "parameters",
                                         os.environ["VIRTUAL_ENV"] + "/data/csv/spam/naive_bayes_spam.png")

    # -------------------- CROSS VALIDATION --------------------
    cross_validation_size = 1
    cross_spam_dataset = context.load_dataset(csv_file=file, one_hot=False, validation_size=np.float32(cross_validation_size))
    spam_cross_X_train, spam_cross_y_train, spam_cross_X_test, spam_cross_y_test = split_data_for_k_fold(cross_spam_dataset.valid.get_features, cross_spam_dataset.valid.get_labels)

    preprocessor_context = DiscretizerContext(SupervisedDiscretizationStrategy())
    supervised_discretised_dataset = preprocessor_context.discretize(data_set=spam_dataset, validation_size=np.float32(cross_validation_size))

    preprocessor_context.set_strategy(UnsupervisedDiscretizationStrategy())
    unsupervised_discretised_dataset = preprocessor_context.discretize(data_set=spam_dataset, validation_size=np.float32(cross_validation_size))

    spam_cross_multinomial_datasets = [["spam_dataset", cross_spam_dataset],["supervised_discretised_dataset", supervised_discretised_dataset], ["unsupervised_discretised_dataset", unsupervised_discretised_dataset]]

    # Params
    cross_tree_params = [10]
    cross_neighbors_params = [10]
    cross_weights_params = ['distance']

    # Results
    cross_tree_results = get_tree_results(cross_tree_params, spam_cross_X_train, spam_cross_y_train, spam_cross_X_test, spam_cross_y_test)
    cross_knn_results = get_knn_results(cross_neighbors_params, cross_weights_params, spam_cross_X_train, spam_cross_y_train, spam_cross_X_test, spam_cross_y_test)
    cross_naive_bayes_results = get_naive_bayes_results(spam_cross_multinomial_datasets, spam_class_prob, True)
    print("CROSS TREE: ", cross_tree_results)
    print("CROSS KNN: ", cross_knn_results)
    print("CROSS NAIVE BAYES: ", cross_naive_bayes_results)
    # -------------------- PART 3 --------------------
    # noises = [0, 0.05, 0.10, 0.20]
    # proportions = [0.25, 0.5, 0.75, 1]
    # state = 1
    #
    # for proportion in proportions:
    #     for noise in noises:
    #         print("noise: " + str(noise))
    #         print("proportion: " + str(proportion))
    #
    #         features, labels = train_set_with_size(cross_spam_dataset, proportion, state)
    #         # train_features = apply_noise_to_features(train_features, noise)
    #
    #         X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=validation_size, random_state=state)
    #
    #         X_train_with_noise = apply_noise_to_features(X_train, noise)
    #
    #         noise_tree_results = get_tree_results(cross_tree_params, X_train_with_noise, y_train, X_test, y_test)
    #         noise_knn_results = get_knn_results(cross_neighbors_params, cross_weights_params, X_train_with_noise, y_train, X_test, y_test)
    #
    #         naive_bayes_results = list()
    #
    #         # Gaussian Naive Bayes
    #         naive_bayes_classifier = get_gaussian_naive_bayes(X_train, y_train, spam_class_prob)
    #
    #         y_pred = naive_bayes_classifier.predict(X_test)
    #         y_true = y_test
    #
    #         gaussian_score_result = naive_bayes_classifier.score(X_test, y_test)
    #         gaussian_f1_score_result = f1_score(y_true, y_pred, average='weighted')
    #
    #
    #         noise_naive_bayes_results = ["gaussian_naive_bayes", gaussian_score_result, gaussian_f1_score_result]
    #         print("NOISE TREE: ", noise_tree_results)
    #         print("NOISE KNN: ", noise_knn_results)
    #         print("NOISE NAIVE BAYES: ", noise_naive_bayes_results)
    #         print("___________________________________________________________")


if __name__ == '__main__':
    main()