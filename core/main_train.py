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
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize, scale, StandardScaler
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

from commons.helpers.dataset.context import Context
from commons.helpers.dataset.strategies.music_genre_dataset.jmirmfcc_strategy import MusicGenreJMIRMFCCsStrategy
from commons.helpers.dataset.strategies.music_genre_dataset.jmirderivatives_strategy \
    import MusicGenreJMIRDERIVATIVESsStrategy
from commons.helpers.dataset.strategies.music_genre_dataset.ssd_strategy import MusicGenreSSDsStrategy

from classifiers.galaxy_classifiers.knn_classifier import KNNClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pickle

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
def standardize_data(X, feature_set_name=None):
    # X = normalize(X, axis=0, norm='max')
    # get_params save it to scale prediction dataset
    scaler = StandardScaler()
    X_new = scaler.fit_transform(X)
    if feature_set_name is not None:
        pickle.dump(scaler, open(feature_set_name + '_scaler_obj.sav', 'wb'))
    return X_new

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

def get_rand_forest_results(max_depths, n_estimators, X_train, y_train, X_test, y_test):
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
    results = list()
    for md in max_depths:
        for ne in n_estimators:
            clf = RandomForestClassifier(n_jobs=8, oob_score=True, max_depth=md, n_estimators=ne)
            clf.fit(X_train, y_train)
            params = "max_depth=" + str(md) + ";n_estimators=" + str(ne)

            y_pred = clf.predict(X_test)
            y_true = y_test

            score_result = clf.score(X_test, y_test)
            train_score_result = clf.score(X_train, y_train)
            f1_score_result = f1_score(y_true, y_pred, average='weighted')

            results.append([params, score_result, f1_score_result, train_score_result])

    print("Random Forest: ", results)
    return results


def get_voting_classifier_results(voting, weights, clf1, clf2, clf3, X_train, y_train, X_test, y_test):
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
    results = list()
    for v in voting:
        for w in weights:
            clf = VotingClassifier(voting=v, weights=w, estimators=[('gnb', clf1), ('knn', clf2), ('rf', clf3)], n_jobs=8)
            clf.fit(X_train, y_train)
            params = "voting=" + v + ";weights=" + str(w)

            y_pred = clf.predict(X_test)
            y_true = y_test

            score_result = clf.score(X_test, y_test)
            train_score_result = clf.score(X_train, y_train)
            f1_score_result = f1_score(y_true, y_pred, average='weighted')

            results.append([params, score_result, f1_score_result, train_score_result])

    print("Voting Classifier: ", results)
    return results

def get_best_params_for_model(param_grid, model, train_X, train_y):

    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=1)
    grid = GridSearchCV(model, param_grid=param_grid, cv=cv, n_jobs=8)
    grid.fit(train_X, train_y)
    return grid

def main():
    start_time = time.time()
    # classes mean = 7182.2 ~ 7200


    # PRETRAITEMENT DES DONNEES
    csv_path = os.environ["VIRTUAL_ENV"] + "/data/music/tagged_feature_sets/"
    mfcc_path = csv_path + "msd-jmirmfccs_dev/"
    spectral_derivatives_path = csv_path + "msd-jmirderivatives_dev/"
    ssd_path = csv_path + "msd-ssd_dev/"
    real_mfcc_file = mfcc_path + "msd-jmirmfccs_dev_clean.csv"
    real_spectral_derivatives_file = spectral_derivatives_path + "msd-jmirderivatives_dev_clean.csv"
    real_ssd_file = ssd_path + "msd-ssd_dev_clean.csv"

    classes_mean = 7200

    remove_unused_columns(mfcc_path + "msd-jmirmfccs_dev.csv")
    remove_unused_columns(spectral_derivatives_path + "msd-jmirderivatives_dev.csv")
    remove_unused_columns(ssd_path + "msd-ssd_dev.csv")
    balance_classes(real_mfcc_file, music_labels, classes_mean)
    balance_classes(real_spectral_derivatives_file, music_labels, classes_mean)
    balance_classes(real_ssd_file, music_labels, classes_mean)

    dataset_mfcc = get_dataset(0.2, MusicGenreJMIRMFCCsStrategy(), real_mfcc_file)
    dataset_spectral_derivatives = get_dataset(0.2, MusicGenreJMIRDERIVATIVESsStrategy(),
                                               real_spectral_derivatives_file)
    dataset_sdd = get_dataset(0.2, MusicGenreSSDsStrategy(), real_ssd_file)

    norm_train_mfcc = standardize_data(dataset_mfcc.train.get_features, 'mfcc')
    norm_valid_mfcc = standardize_data(dataset_mfcc.valid.get_features)
    norm_train_spec_deriv = standardize_data(dataset_spectral_derivatives.train.get_features, 'deriv')
    norm_valid_spec_deriv = standardize_data(dataset_spectral_derivatives.valid.get_features)
    norm_train_ssd = standardize_data(dataset_sdd.train.get_features, 'ssd')
    norm_valid_ssd = standardize_data(dataset_sdd.valid.get_features)

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

    # ENTRAINEMENT DES MODELES

    clf1 = GaussianNB()
    clf2 = KNeighborsClassifier(n_jobs=8, n_neighbors=21, weights='distance')
    clf3_1 = RandomForestClassifier(n_jobs=8, oob_score=True, max_depth=15, n_estimators=50)
    clf3_2 = RandomForestClassifier(n_jobs=8, oob_score=True, max_depth=19, n_estimators=50)

    # get_rand_forest_results([15], [50], norm_train_mfcc, dataset_mfcc.train.get_labels,
    #                         norm_valid_mfcc, dataset_mfcc.valid.get_labels)
    #
    # get_rand_forest_results([19], [50], norm_train_spec_deriv, dataset_spectral_derivatives.train.get_labels,
    #                         norm_valid_spec_deriv, dataset_spectral_derivatives.valid.get_labels)
    #
    # get_rand_forest_results([19], [50], norm_train_ssd, dataset_sdd.train.get_labels,
    #                         norm_valid_ssd, dataset_sdd.valid.get_labels)

    # voting_params = ['soft']
    # weights_params = [[1, 2, 3]]

    # get_voting_classifier_results(voting_params, weights_params, clf1, clf2, clf3_1, norm_train_mfcc,
    #                               dataset_mfcc.train.get_labels, norm_valid_mfcc, dataset_mfcc.valid.get_labels)
    #
    # get_voting_classifier_results(voting_params, weights_params, clf1, clf2, clf3_2, norm_train_spec_deriv,
    #                               dataset_spectral_derivatives.train.get_labels, norm_valid_spec_deriv,
    #                               dataset_spectral_derivatives.valid.get_labels)
    #
    # get_voting_classifier_results(voting_params, weights_params, clf1, clf2, clf3_2, norm_train_ssd,
    #                               dataset_sdd.train.get_labels, norm_valid_ssd, dataset_sdd.valid.get_labels)

    eclf_mfcc = VotingClassifier(voting='hard', weights=[1, 2, 3], estimators=[('gnb', clf1), ('knn', clf2), ('rf', clf3_1)], n_jobs=8)
    eclf_deriv = VotingClassifier(voting='hard', weights=[1, 2, 3], estimators=[('gnb', clf1), ('knn', clf2), ('rf', clf3_2)], n_jobs=8)
    eclf_ssd = VotingClassifier(voting='hard', weights=[1, 2, 3], estimators=[('gnb', clf1), ('knn', clf2), ('rf', clf3_2)], n_jobs=8)

    eclf_mfcc.fit(norm_train_mfcc, dataset_mfcc.train.get_labels)
    print(eclf_mfcc.score(norm_valid_mfcc, dataset_mfcc.valid.get_labels))

    eclf_deriv.fit(norm_train_spec_deriv, dataset_spectral_derivatives.train.get_labels)
    print(eclf_deriv.score(norm_valid_spec_deriv, dataset_spectral_derivatives.valid.get_labels))

    eclf_ssd.fit(norm_train_ssd, dataset_sdd.train.get_labels)
    print(eclf_ssd.score(norm_valid_ssd, dataset_sdd.valid.get_labels))

    pickle.dump(eclf_mfcc, open('mfcc_model.sav', 'wb'))
    pickle.dump(eclf_deriv, open('deriv_model.sav', 'wb'))
    pickle.dump(eclf_ssd, open('ssd_model.sav', 'wb'))

    print("--- %s seconds ---" % (time.time() - start_time))
    print("hello")


if __name__ == '__main__':
    main()
