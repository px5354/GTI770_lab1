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

Notes : This file is the main program file. Please note that this project might be over-commented compared to a real,
        industry-class framework.
        This framework is used as a part of the GTI770 course at Ecole de technologie superieure of Montreal. The amount
        of comments is only to make sure everyone from all level can clearly understand the code given to complete
        the class assignment.
"""

import timeit
import numpy as np
import os

from core.feature_extraction.galaxy.galaxy_processor import GalaxyProcessor
from commons.helpers.dataset.context import Context
from commons.preprocessors.discretization.context import DiscretizerContext
from commons.helpers.dataset.strategies.galaxy_dataset.feature_strategy import GalaxyDataSetFeatureStrategy
from commons.helpers.dataset.strategies.galaxy_dataset.image_strategy import GalaxyDataSetImageStrategy
from commons.helpers.dataset.strategies.galaxy_dataset.label_strategy import GalaxyDataSetLabelStrategy
from commons.helpers.dataset.strategies.spam_dataset.feature_strategy import SpamDataSetFeatureStrategy
from commons.preprocessors.discretization.context import DiscretizerContext
from commons.preprocessors.discretization.strategies.unsupervised.unsupervised_discretization_strategy import \
    UnsupervisedDiscretizationStrategy
from commons.preprocessors.discretization.strategies.supervised.supervised_discretization_strategy import \
    SupervisedDiscretizationStrategy


def main():
    """
        Program's entry point.
    """

    start = timeit.default_timer()

    # The desired validation size.
    validation_size = 0.2

    # Get the ground truth CSV file from script's parameters.
    galaxy_csv_file = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/galaxy.csv"
    galaxy_feature_csv_file = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/galaxy_feature_vectors.csv"
    spam_feature_csv_file = os.environ["VIRTUAL_ENV"] + "/data/csv/spam/spam.csv"

    galaxy_images_path = os.environ["VIRTUAL_ENV"] + "/data/images/"
    galaxy_feature_vector_export_path = os.environ[
                                            "VIRTUAL_ENV"] + "/data/csv/galaxy/exported_personal_galaxy_feature_vectors.csv"
    galaxy_mlp_export_path = os.environ["VIRTUAL_ENV"] + "/data/models/exports/MLP/my_mlp"

    # Create instance of data set loading strategies.
    galaxy_image_data_set_strategy = GalaxyDataSetImageStrategy()
    galaxy_feature_data_set_strategy = GalaxyDataSetFeatureStrategy()
    galaxy_label_data_set_strategy = GalaxyDataSetLabelStrategy()

    spam_feature_dataset_strategy = SpamDataSetFeatureStrategy()

    # Set the context to galaxy image data set loading strategy.
    context = Context(galaxy_image_data_set_strategy)
    img_dataset = context.load_dataset(csv_file=galaxy_csv_file, one_hot=True,
                                       validation_size=np.float32(validation_size))

    # Set the context to galaxy feature data set loading strategy.
    context.set_strategy(galaxy_feature_data_set_strategy)
    feature_oneHot_dataset = context.load_dataset(csv_file=galaxy_feature_csv_file, one_hot=True,
                                                  validation_size=np.float32(0.2))

    feature_dataset = context.load_dataset(csv_file=galaxy_feature_csv_file, one_hot=False,
                                           validation_size=np.float32(0.2))

    # Set the context to galaxy label data set loading strategy.
    context.set_strategy(galaxy_label_data_set_strategy)
    label_dataset = context.load_dataset(csv_file=galaxy_csv_file, one_hot=False,
                                         validation_size=np.float32(validation_size))

    context.set_strategy(spam_feature_dataset_strategy)
    spam_feature_dataset = context.load_dataset(csv_file=spam_feature_csv_file, one_hot=False,
                                                validation_size=np.float32(validation_size))

    # For TP02, set the discretization strategy and discretize data.
    # preprocessor_context = DiscretizerContext(SupervisedDiscretizationStrategy())
    #
    # supervised_discretised_dataset = preprocessor_context.discretize(data_set=feature_dataset,
    #                                                                  validation_size=np.float32(validation_size))
    #
    # preprocessor_context.set_strategy(UnsupervisedDiscretizationStrategy())
    #
    # unsupervised_discretised_dataset = preprocessor_context.discretize(data_set=feature_dataset,
    #                                                                    validation_size=np.float32(validation_size))

    # # Process galaxies.
    galaxy_processor = GalaxyProcessor(galaxy_images_path)
    # features = galaxy_processor.process_galaxy(label_dataset)
    features, labels = galaxy_processor.process_galaxy(label_dataset)

    # # Save extracted features to file.
    galaxy_feature_vector_path = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/exported_personal_feature"
    galaxy_label_path = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/exported_personal_label_galaxy.csv"

    stop = timeit.default_timer()
    print(stop - start)


    for i in range(0, len(features)):
        temp_filepath = galaxy_feature_vector_path + str(i) +".csv"
        np.savetxt(temp_filepath, features[i], delimiter=",")
        print("File saved in directory " + temp_filepath)

    np.savetxt(galaxy_label_path, labels, delimiter=",")
    print("File saved in directory " + galaxy_label_path)

if __name__ == '__main__':
    main()