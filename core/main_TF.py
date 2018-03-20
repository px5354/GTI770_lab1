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


def get_galaxy_dataset(validation_size):

    stategy = GalaxyDataSetFeatureStrategy()
    csv_file = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/galaxy_feature_vectors.csv"
    context = Context(stategy)
    dataset = context.load_dataset(csv_file=csv_file, one_hot=False, validation_size=np.float32(validation_size))

    return dataset

def main():

    #load graph from console: tensorboard --gdir=/home/ens/AK86280/Documents/GTI770_lab1/project/data
    #directory: ~/Documents/GTI770_lab1/project$
    #http://localhost:6006

    galaxy_dataset = get_galaxy_dataset(0.0)
    train_path = os.environ["VIRTUAL_ENV"] + "/data/"
    batch_size = 100
    image_size = 74
    learning_rate = 0.0005
    dropout_probability = 0.5
    number_of_steps = 4000
    number_of_classes = 3
    number_of_channels = 2
    number_of_hidden_layer = 2

    tb_classifier = MLPClassifierTensorBoard(train_path,
                                             batch_size,
                                             image_size,
                                             learning_rate,
                                             dropout_probability,
                                             number_of_steps,
                                             number_of_classes,
                                             number_of_channels,
                                             number_of_hidden_layer)

    tb_classifier.train(galaxy_dataset)

if __name__ == '__main__':
    main()