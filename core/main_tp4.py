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
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize, scale
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

from commons.helpers.dataset.context import Context
from commons.helpers.dataset.strategies.music_genre_dataset.jmirmfcc_strategy import MusicGenreJMIRMFCCsStrategy
from commons.helpers.dataset.strategies.music_genre_dataset.jmirderivatives_strategy \
    import MusicGenreJMIRDERIVATIVESsStrategy
from commons.helpers.dataset.strategies.music_genre_dataset.ssd_strategy import MusicGenreSSDsStrategy

from classifiers.galaxy_classifiers.knn_classifier import KNNClassifier

music_labels = ["BIG_BAND", "BLUES_CONTEMPORARY", "COUNTRY_TRADITIONAL", "DANCE", "ELECTRONICA", "EXPERIMENTAL",
                "FOLK_INTERNATIONAL", "GOSPEL", "GRUNGE_EMO", "HIP_HOP_RAP", "JAZZ_CLASSIC", "METAL_ALTERNATIVE",
                "METAL_DEATH", "METAL_HEAVY", "POP_CONTEMPORARY", "POP_INDIE", "POP_LATIN", "PUNK", "REGGAE",
                "RNB_SOUL", "ROCK_ALTERNATIVE", "ROCK_COLLEGE", "ROCK_CONTEMPORARY", "ROCK_HARD",
                "ROCK_NEO_PSYCHEDELIA"]


def get_dataset(validation_size, strategy, csv_file):
    context = Context(strategy)
    dataset = context.load_dataset(csv_file=csv_file, one_hot=False, validation_size=np.float32(validation_size))
    return dataset


def remove_unused_columns(file_name):
    df = pd.read_csv(file_name, header=None)
    df.drop(df.columns[[0, 1]], axis=1, inplace=True)
    new_file_name = file_name.split(".")[0] + "_clean.csv"
    df.to_csv(new_file_name, header=False, index=False)


def balance_classes(file_name, labels, up_down_sample_n):
    df = pd.read_csv(file_name, header=None)
    # Down-sample Majority Class
    # https://elitedatascience.com/imbalanced-classes
    for i, label in enumerate(labels):
        df_label = df[df[df.shape[1] - 1] == label]
        if df_label.shape[0] < up_down_sample_n:
            to_replace = True
        else:
            to_replace = False

        # Downsample majority class
        df_label_resampled = resample(df_label,
                                      replace=to_replace,  # sample without replacement
                                      n_samples=up_down_sample_n,  # to match minority class
                                      random_state=123)  # reproducible results
        if i == 0:
            new_df = df_label_resampled
        else:
            new_df = pd.concat([df_label_resampled, new_df], ignore_index=True)

    new_df.to_csv(file_name, header=False, index=False)


# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
def normalize_data(X):
    # X = normalize(X, axis=0, norm='max')
    X = scale(X)
    return X


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


def plot_hyper_parameters_comparison(params_array, results, title, xlabel_name, filename, results2=None):
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

    if results2 is None:
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


def main():
    # mfcc classes mean = 10967.44 ~ 10967
    csv_path = os.environ["VIRTUAL_ENV"] + "/data/music/tagged_feature_sets/"
    mfcc_path = csv_path + "msd-jmirmfccs_dev/"
    spectral_derivatives_path = csv_path + "msd-jmirderivatives_dev/"
    ssd_path = csv_path + "msd-ssd_dev/"
    real_mfcc_file = mfcc_path + "msd-jmirmfccs_dev_clean.csv"
    real_spectral_derivatives_file = spectral_derivatives_path + "msd-jmirderivatives_dev_clean.csv"
    real_ssd_file = ssd_path + "msd-ssd_dev_clean.csv"

    classes_mean = 10967

    # remove_unused_columns(mfcc_path + "msd-jmirmfccs_dev.csv")
    # remove_unused_columns(spectral_derivatives_path + "msd-jmirderivatives_dev.csv")
    # remove_unused_columns(ssd_path + "msd-ssd_dev.csv")
    # balance_classes(real_mfcc_file, music_labels, classes_mean)
    # balance_classes(real_spectral_derivatives_file, music_labels, classes_mean)
    # balance_classes(real_ssd_file, music_labels, classes_mean)

    dataset_mfcc = get_dataset(0.2, MusicGenreJMIRMFCCsStrategy(), real_mfcc_file)
    dataset_spectral_derivatives = get_dataset(0.2, MusicGenreJMIRDERIVATIVESsStrategy(),
                                               real_spectral_derivatives_file)
    dataset_sdd = get_dataset(0.2, MusicGenreSSDsStrategy(), real_ssd_file)

    norm_train_mfcc = normalize_data(dataset_mfcc.train.get_features)
    norm_valid_mfcc = normalize_data(dataset_mfcc.valid.get_features)
    norm_train_spec_deriv = normalize_data(dataset_spectral_derivatives.train.get_features)
    norm_valid_spec_deriv = normalize_data(dataset_spectral_derivatives.valid.get_features)
    norm_train_ssd = normalize_data(dataset_sdd.train.get_features)
    norm_valid_ssd = normalize_data(dataset_sdd.valid.get_features)

    # https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    pca_mfcc = PCA(.95)
    pca_spec_deriv = PCA(.95)
    pca_ssd = PCA(.95)

    pca_mfcc.fit(norm_train_mfcc)
    pca_spec_deriv.fit(norm_train_spec_deriv)
    pca_ssd.fit(norm_train_ssd)

    print("PCA MFCC")
    print(pca_mfcc.explained_variance_ratio_)
    print(pca_mfcc.singular_values_)

    print("PCA DERIV")
    print(pca_spec_deriv.explained_variance_ratio_)
    print(pca_spec_deriv.singular_values_)

    print("PCA SSD")
    print(pca_ssd.explained_variance_ratio_)
    print(pca_ssd.singular_values_)

    norm_train_mfcc = pca_mfcc.transform(norm_train_mfcc)
    norm_valid_mfcc = pca_mfcc.transform(norm_valid_mfcc)
    norm_train_spec_deriv = pca_spec_deriv.transform(norm_train_spec_deriv)
    norm_valid_spec_deriv = pca_spec_deriv.transform(norm_valid_spec_deriv)
    norm_train_ssd = pca_ssd.transform(norm_train_ssd)
    norm_valid_ssd = pca_ssd.transform(norm_valid_ssd)

    # -------------------- MFCC  --------------------
    spam_X_train = norm_train_mfcc.get_features
    spam_y_train = norm_train_mfcc.get_labels
    spam_X_test = norm_valid_mfcc.get_features
    spam_y_test = norm_valid_mfcc.get_labels
    spam_class_prob = [0.4003, 0.5997]

    # params
    neighbors_params = [3, 5, 10]
    weights_params = ['uniform', 'distance']

    # results
    knn_results = get_knn_results(neighbors_params, weights_params, spam_X_train, spam_y_train, spam_X_test,
                                  spam_y_test)

    # Plot results
    results_knn_uniform = list()
    results_knn_distance = list()

    for result in knn_results:
        if result[0].split(";weights=")[1] == "uniform":
            results_knn_uniform.append(result)
        else:
            results_knn_distance.append(result)

    print("NORMAL KNN: ", knn_results)

    plot_hyper_parameters_comparison(neighbors_params, results_knn_uniform, "KNN with different weights", "n_neighbors",
                                     os.environ["VIRTUAL_ENV"] + "/data/csv/spam/knn_spam.png", results_knn_distance)

    print("hello")


if __name__ == '__main__':
    main()
