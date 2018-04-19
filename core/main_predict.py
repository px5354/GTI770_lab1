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
import sys
import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize, scale
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

from commons.helpers.dataset.context import Context
from commons.helpers.dataset.strategies.music_genre_dataset.jmirmfcc_strategy import MusicGenreJMIRMFCCsStrategy
from commons.helpers.dataset.strategies.music_genre_dataset.jmirderivatives_strategy \
    import MusicGenreJMIRDERIVATIVESsStrategy
from commons.helpers.dataset.strategies.music_genre_dataset.ssd_strategy import MusicGenreSSDsStrategy
#
# from classifiers.galaxy_classifiers.knn_classifier import KNNClassifier
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from classifiers.music_genre_classifiers.random_forest_classifier import RandForestClassifier
# from sklearn.ensemble import VotingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
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

# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
def standardize_data(X, scaler):
    # X = normalize(X, axis=0, norm='max')
    # get_params save it to scale prediction dataset
    X_new = scaler.fit_transform(X)
    return X_new

def convert_labels(results):
    for i, val in enumerate(results):
        results[i] = music_labels[val]
    return results

def main():
    start_time = time.time()
    print('input_dir: ' + sys.argv[1])
    print('output_file: ' + sys.argv[2])
    # Load scalers
    mfcc_scaler = pickle.load(open('mfcc_scaler_obj.sav', 'rb'))
    deriv_scaler = pickle.load(open('deriv_scaler_obj.sav', 'rb'))
    ssd_scaler = pickle.load(open('ssd_scaler_obj.sav', 'rb'))

    # Load models
    mfcc_model = pickle.load(open('mfcc_model.sav', 'rb'))
    deriv_model = pickle.load(open('deriv_model.sav', 'rb'))
    ssd_model = pickle.load(open('ssd_model.sav', 'rb'))

    output_file = sys.argv[2]
    csv_path = sys.argv[1]
    mfcc_path = csv_path + "msd-jmirmfccs_test_nolabels/"
    spectral_derivatives_path = csv_path + "msd-jmirderivatives_test_nolabels/"
    ssd_path = csv_path + "msd-ssd_test_nolabels/"
    real_mfcc_file = mfcc_path + "msd-jmirmfccs_test_nolabels_clean.csv"
    real_spectral_derivatives_file = spectral_derivatives_path + "msd-jmirderivatives_test_nolabels_clean.csv"
    real_ssd_file = ssd_path + "msd-ssd_test_nolabels_clean.csv"

    # PRETRAITEMENT DES DONNEES
    print("PRETRAITEMENT ")
    remove_unused_columns(mfcc_path + "msd-jmirmfccs_test_nolabels.csv")
    remove_unused_columns(spectral_derivatives_path + "msd-jmirderivatives_test_nolabels.csv")
    remove_unused_columns(ssd_path + "msd-ssd_test_nolabels.csv")

    dataset_mfcc = get_dataset(1, MusicGenreJMIRMFCCsStrategy(), real_mfcc_file)
    dataset_spectral_derivatives = get_dataset(1, MusicGenreJMIRDERIVATIVESsStrategy(),
                                               real_spectral_derivatives_file)
    dataset_sdd = get_dataset(1, MusicGenreSSDsStrategy(), real_ssd_file)

    norm_mfcc = standardize_data(dataset_mfcc.valid.get_features, mfcc_scaler)
    norm_spec_deriv = standardize_data(dataset_spectral_derivatives.valid.get_features, deriv_scaler)
    norm_ssd = standardize_data(dataset_sdd.valid.get_features, ssd_scaler)

    # # https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    pca_mfcc = PCA(.95)
    pca_spec_deriv = PCA(.95)
    pca_ssd = PCA(.95)

    pca_mfcc.fit(norm_mfcc)
    pca_spec_deriv.fit(norm_spec_deriv)
    pca_ssd.fit(norm_ssd)

    norm_mfcc = pca_mfcc.transform(norm_mfcc)
    norm_spec_deriv = pca_spec_deriv.transform(norm_spec_deriv)
    norm_ssd = pca_ssd.transform(norm_ssd)

    # PREDICTION
    mfcc_predicts = mfcc_model.predict(norm_mfcc)
    deriv_predicts = deriv_model.predict(norm_spec_deriv)
    ssd_predicts = ssd_model.predict(norm_ssd)

    if len(mfcc_predicts) == len(deriv_predicts) and len(ssd_predicts) == len(deriv_predicts):
        voted_predictions = list()
        for i in range(len(mfcc_predicts)):
            mfcc_pred = mfcc_predicts[i]
            deriv_pred = deriv_predicts[i]
            ssd_pred = ssd_predicts[i]
            if mfcc_pred == deriv_pred and mfcc_pred == ssd_pred:
                voted_predictions.append(mfcc_pred)
            elif mfcc_pred == deriv_pred and mfcc_pred != ssd_pred:
                voted_predictions.append(mfcc_pred)
            elif mfcc_pred == ssd_pred and mfcc_pred != deriv_pred:
                voted_predictions.append(mfcc_pred)
            elif ssd_pred == deriv_pred and mfcc_pred != ssd_pred:
                voted_predictions.append(ssd_pred)
            else:
                voted_predictions.append(ssd_pred)

    print(voted_predictions)
    results = convert_labels(voted_predictions)
    print(results)
    np.savetxt(output_file, results, delimiter=',', fmt='%s')

    # SAUVEGARDE DES DONNEES
    # file = open(sys.argv[2], 'w').write()

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
