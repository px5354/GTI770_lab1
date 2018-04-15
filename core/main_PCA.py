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
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from commons.helpers.dataset.context import Context
from commons.helpers.dataset.strategies.music_genre_dataset.jmirmfcc_strategy import MusicGenreJMIRMFCCsStrategy

def get_dataset(validation_size, strategy, csv_file):
    context = Context(strategy)
    dataset = context.load_dataset(csv_file=csv_file, one_hot=False, validation_size=np.float32(validation_size))
    return dataset

def remove_unused_columns(file_name):
    df = pd.read_csv(file_name, header=None)
    df.drop(df.columns[[0, 1]], axis=1, inplace=True)
    new_file_name = file_name.split(".")[0] + "_clean.csv"
    df.to_csv(new_file_name, header=False, index=False)

def normalize_data(X):
    X = normalize(X)
    return X

def main():
    csv_path = os.environ["VIRTUAL_ENV"] + "/data/music/tagged_feature_sets/"
    mfcc_path = csv_path + "msd-jmirmfccs_dev/"
    remove_unused_columns(mfcc_path + "msd-jmirmfccs_dev.csv")
    dataset = get_dataset(0.2, MusicGenreJMIRMFCCsStrategy(), mfcc_path + "msd-jmirmfccs_dev_clean.csv")
    norm_train = normalize_data(dataset.train.get_features)
    norm_valid = normalize_data(dataset.valid.get_features)


    print("hello")

if __name__ == '__main__':
    main()