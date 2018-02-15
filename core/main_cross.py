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
import random
from random import choice
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from classifiers.galaxy_classifiers.knn_classifier import KNNClassifier
from classifiers.galaxy_classifiers.gaussian_naive_bayes_classifier import GaussianNaiveBayesClassifier
from classifiers.galaxy_classifiers.multinomial_naive_bayes_classifier import MultinomialNaiveBayesClassifier
from commons.helpers.dataset.context import Context
from commons.helpers.dataset.strategies.spam_dataset.feature_strategy import SpamDataSetFeatureStrategy
from commons.preprocessors.discretization.context import DiscretizerContext
from commons.preprocessors.discretization.strategies.unsupervised.unsupervised_discretization_strategy import \
    UnsupervisedDiscretizationStrategy
from commons.preprocessors.discretization.strategies.supervised.supervised_discretization_strategy import \
    SupervisedDiscretizationStrategy
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score


def plot_hyper_parameters_comparison(params_array, results, title, xlabel_name, filename):

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

    ax.plot(x, y_score, marker='o', linestyle='-', ms=10, label="score")
    ax.plot(x, y_f1_score, marker='o', linestyle='-', ms=10, label="f1_score")

    ax.legend()
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

    clf = TreeClassifier(max_depth=max_depth)
    clf.train(X_train, y_train)

    return clf

def get_knn(X_train, y_train, n_neighbors, weights):

    knn = KNNClassifier(n_neighbors, weights)
    knn.train(X_train, y_train)

    return knn

def get_gaussian_naive_bayes(X_train, y_train, priors=None):

    gnb = GaussianNaiveBayesClassifier(priors=priors)
    gnb.train(X_train, y_train)

    return gnb

def get_multinomial_naive_bayes(X_train, y_train, fit_prior=False, class_prior=None):

    mnb = MultinomialNaiveBayesClassifier(fit_prior, class_prior)
    mnb.train(X_train, y_train)

    return mnb

def train_set_with_size(trainSet, proportion, state):

    features = trainSet.get_features
    labels = trainSet.get_labels

    _, train_features, _, train_labels = train_test_split(features, labels, test_size=proportion, random_state =state)

    return train_features, train_labels


def main():
    # CROSS-VALIDATION

    # The desired validation size.
    validation_size = 1.0

    spam_feature_dataset_strategy = SpamDataSetFeatureStrategy()
    context = Context(spam_feature_dataset_strategy)

    spam_feature_csv_file = os.environ["VIRTUAL_ENV"] + "/data/csv/spam/spam.csv"

    spam_dataset = context.load_dataset(csv_file=spam_feature_csv_file, one_hot=False, validation_size=np.float32(validation_size))

    spam_dataset_kfold = spam_dataset.valid.get_features
    spam_labels_kfold = spam_dataset.valid.get_labels

    X = spam_dataset_kfold
    y = spam_labels_kfold

    kf = KFold(n_splits=10)
    kf.get_n_splits(X)

    print(kf)

    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]



    # ------------------------ TREE ------------------------
    best_tree_params = [10]
    best_results_tree = list()

    for max_depth in best_tree_params:

        if(max_depth == 0):
            trained_tree_classifier = get_decision_tree(X_train, y_train, max_depth=None)
            params = "max_depth=None"
        else:
            trained_tree_classifier = get_decision_tree(X_train, y_train, max_depth=max_depth)
            params = "max_depth=" + str(max_depth)

        y_pred = trained_tree_classifier.predict(X_test)
        y_true = y_test

        score_result = trained_tree_classifier.score(X_test, y_test)
        f1_score_result = f1_score(y_true, y_pred, average='weighted')

        best_results_tree.append([params, score_result, f1_score_result])

    print("TREE: ", best_results_tree)

    # ------------------------ KNN ------------------------
    best_neighbors_param = [10]
    best_weight_param = ['distance']
    best_results_knn = list()

    for n_neighbors in best_neighbors_param:
        for weights in best_weight_param:
            trained_knn_classifier = get_knn(X_train, y_train, n_neighbors, weights)
            params = "n_neighbors=" + str(n_neighbors) + ";weights=" + weights

            y_pred = trained_knn_classifier.predict(X_test)
            y_true = y_test

            score_result = trained_knn_classifier.score(X_test, y_test)
            f1_score_result = f1_score(y_true, y_pred, average='weighted')

            best_results_knn.append([params, score_result, f1_score_result])

    print("KNN: ", best_results_knn)


    # For Gaussian and Multinomial
    spam_class_probs = [0.4003, 0.5997]

    # ------------------------ GAUSSIAN BAYES NAIVES ------------------------
    best_results_gaussian = list()

    naive_bayes_classifier = get_gaussian_naive_bayes(X_train, y_train, spam_class_probs)

    y_pred = naive_bayes_classifier.predict(X_test)
    y_true = y_test

    score_result = naive_bayes_classifier.score(X_test, y_test)
    f1_score_result = f1_score(y_true, y_pred, average='weighted')

    best_results_gaussian.append(['[0.4003, 0.5997]', score_result, f1_score_result])

    print("GAUSSIAN: ", best_results_gaussian)

    # # ------------------------ MULTINOMINIAL BAYES NAIVES ------------------------
    # preprocessor_context = DiscretizerContext(SupervisedDiscretizationStrategy())
    # supervised_discretised_dataset = preprocessor_context.discretize(data_set=spam_dataset, validation_size=np.float32(validation_size))
    #
    # preprocessor_context.set_strategy(UnsupervisedDiscretizationStrategy())
    # unsupervised_discretised_dataset = preprocessor_context.discretize(data_set=spam_dataset, validation_size=np.float32(validation_size))
    #
    # multinomial_naive_bayes_datasets = [["spam_dataset", spam_dataset],
    #                                     ["supervised_discretised_dataset", supervised_discretised_dataset],
    #                                     ["unsupervised_discretised_dataset", unsupervised_discretised_dataset]]
    #
    # best_results_multinomial = list()
    #
    #
    # for i, mnb_dataset in enumerate(multinomial_naive_bayes_datasets):
    #
    #     if i == 0:
    #         naive_bayes_classifier = get_multinomial_naive_bayes(mnb_dataset[1].train.get_features, mnb_dataset[1].train.get_labels, True, spam_class_probs)
    #     else:
    #         naive_bayes_classifier = get_multinomial_naive_bayes(mnb_dataset[1].train.get_features, mnb_dataset[1].train.get_labels)
    #
    #     y_pred = naive_bayes_classifier.predict(mnb_dataset[1].valid.get_features)
    #     y_true = mnb_dataset[1].valid.get_labels
    #
    #     score_result = naive_bayes_classifier.score(mnb_dataset[1].valid.get_features,
    #                                                 mnb_dataset[1].valid.get_labels)
    #     f1_score_result = f1_score(y_true, y_pred, average='weighted')
    #
    #     if i == 0:
    #         best_results_multinomial.append([mnb_dataset[0] + ';class_probs=[0.4003, 0.5997]', score_result, f1_score_result])
    #     else:
    #         best_results_multinomial.append([mnb_dataset[0], score_result, f1_score_result])
    #
    # print("MULTINOMIAL: ", best_results_multinomial)

    # CROSS-VALIDATION



    noises = [0, 0.05, 0.10, 0.20]
    proportions = [0.20, 0.5, 0.75, 1]
    state = 1

    # for proportion in proportions:
    #     for noise in noises:
    #         print("noise: " + str(noise))
    #         print("proportion: " + str(proportion))
    #
    #         train_features, train_labels = train_set_with_size(spam_dataset_train, proportion, state)
    #         train_features = apply_noise_to_features(train_features, noise)
    #
    #         valid_features = apply_noise_to_features(spam_dataset_valid.get_features, noise)
    #         valid_labels = spam_dataset_valid.get_labels
    #
    #         get_decision_tree_score(train_features, valid_features, train_labels, valid_labels, max_depth=None)
    #         get_decision_tree_score(train_features, valid_features, train_labels, valid_labels, max_depth=3)
    #         get_decision_tree_score(train_features, valid_features, train_labels, valid_labels, max_depth=5)
    #         get_decision_tree_score(train_features, valid_features, train_labels, valid_labels, max_depth=10)
    #         print("___________________________________________________________")
    #
    #         state = state + 1

    preprocessor_context = DiscretizerContext(SupervisedDiscretizationStrategy())

    supervised_discretised_dataset = preprocessor_context.discretize(data_set=spam_dataset,
                                                                     validation_size=np.float32(validation_size))

    preprocessor_context.set_strategy(UnsupervisedDiscretizationStrategy())

    unsupervised_discretised_dataset = preprocessor_context.discretize(data_set=spam_dataset,
                                                                       validation_size=np.float32(validation_size))



    tree_params_array = [0, 3, 5, 10]
    knn_params_array = [3, 5, 10]

    results_tree = list()
    results_knn = list()
    results_gaussian_naive_bayes = list()
    results_multinomial_naive_bayes = list()

    multinomial_naive_bayes_datasets = [["spam_dataset", spam_dataset],
                                        ["supervised_discretised_dataset", supervised_discretised_dataset],
                                        ["unsupervised_discretised_dataset", unsupervised_discretised_dataset]]
    # decision tree
    for max_depth in tree_params_array:

        if(max_depth == 0):
            trained_tree_classifier = get_decision_tree(spam_dataset_train.get_features, spam_dataset_train.get_labels,
                                                        max_depth=None)
            params = "max_depth=None"
        else:
            trained_tree_classifier = get_decision_tree(spam_dataset_train.get_features, spam_dataset_train.get_labels,
                                                        max_depth=max_depth)
            params = "max_depth=" + str(max_depth)

        y_pred = trained_tree_classifier.predict(spam_dataset_valid.get_features)
        y_true = spam_dataset_valid.get_labels

        score_result = trained_tree_classifier.score(spam_dataset_valid.get_features,
                                                             spam_dataset_valid.get_labels)
        f1_score_result = f1_score(y_true, y_pred, average='weighted')

        results_tree.append([params, score_result, f1_score_result])

    # knn
    for n_neighbors in knn_params_array:
        for weights in ['uniform', 'distance']:
            trained_knn_classifier = get_knn(spam_dataset_train.get_features, spam_dataset_train.get_labels,
                                             n_neighbors, weights)
            params = "n_neighbors=" + str(n_neighbors) + ";weights=" + weights

            y_pred = trained_knn_classifier.predict(spam_dataset_valid.get_features)
            y_true = spam_dataset_valid.get_labels

            score_result = trained_knn_classifier.score(spam_dataset_valid.get_features,
                                                        spam_dataset_valid.get_labels)
            f1_score_result = f1_score(y_true, y_pred, average='weighted')

            results_knn.append([params, score_result, f1_score_result])

    # Gaussian Naive Bayes
    naive_bayes_classifier = get_gaussian_naive_bayes(spam_dataset_train.get_features,
                                                           spam_dataset_train.get_labels,
                                                           spam_class_probs)

    y_pred = naive_bayes_classifier.predict(spam_dataset_valid.get_features)
    y_true = spam_dataset_valid.get_labels

    score_result = trained_knn_classifier.score(spam_dataset_valid.get_features,
                                                spam_dataset_valid.get_labels)
    f1_score_result = f1_score(y_true, y_pred, average='weighted')

    results_gaussian_naive_bayes.append(['[0.4003, 0.5997]', score_result, f1_score_result])

    # Multinomial Naive Bayes

    for i, mnb_dataset in enumerate(multinomial_naive_bayes_datasets):

        if i == 0:
            naive_bayes_classifier = get_multinomial_naive_bayes(mnb_dataset[1].train.get_features, mnb_dataset[1].train.get_labels,
                                                                 True, spam_class_probs)
        else:
            naive_bayes_classifier = get_multinomial_naive_bayes(mnb_dataset[1].train.get_features, mnb_dataset[1].train.get_labels)

        y_pred = naive_bayes_classifier.predict(mnb_dataset[1].valid.get_features)
        y_true = mnb_dataset[1].valid.get_labels

        score_result = naive_bayes_classifier.score(mnb_dataset[1].valid.get_features,
                                                    mnb_dataset[1].valid.get_labels)
        f1_score_result = f1_score(y_true, y_pred, average='weighted')

        if i == 0:
            results_multinomial_naive_bayes.append([mnb_dataset[0] + ';class_probs=[0.4003, 0.5997]', score_result, f1_score_result])
        else:
            results_multinomial_naive_bayes.append([mnb_dataset[0], score_result, f1_score_result])

    print(results_tree)
    print(tree_params_array)
    print(results_knn)
    print(knn_params_array)
    # print(results_gaussian_naive_bayes)
    # print(results_multinomial_naive_bayes)


if __name__ == '__main__':
    main()