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
from sklearn.preprocessing import normalize, scale
from sklearn.utils import resample
from sklearn.decomposition import PCA

from commons.helpers.dataset.context import Context
from commons.helpers.dataset.strategies.music_genre_dataset.jmirmfcc_strategy import MusicGenreJMIRMFCCsStrategy
from commons.helpers.dataset.strategies.music_genre_dataset.jmirderivatives_strategy \
    import MusicGenreJMIRDERIVATIVESsStrategy
from commons.helpers.dataset.strategies.music_genre_dataset.ssd_strategy import MusicGenreSSDsStrategy
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
        # new_columns = df.iloc[[-1]].values[0]
        # print(new_columns)
        # test1 = df.head()
        # test = pd.DataFrame(df.values, columns=new_columns)
        # test2 = df.head()
        # df.columns = new_columns
        # df2 = df.filter(regex=label)
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
    dataset_spectral_derivatives = get_dataset(0.2, MusicGenreJMIRDERIVATIVESsStrategy(), real_spectral_derivatives_file)
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

    print("hello")

if __name__ == '__main__':
    main()