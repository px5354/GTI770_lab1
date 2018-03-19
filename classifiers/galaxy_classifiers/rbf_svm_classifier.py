#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Course :
    GTI770 — Systèmes intelligents et apprentissage machine

Project :
    Lab # X - Lab's name

Students :
    Names — Permanent Code

Group :
    GTI770-H18-0X
"""

from __future__ import division, print_function, absolute_import

from sklearn.svm import SVC


class SVMClassifier(object):

    def __init__(self, C, gamma, kernel):
        self.model = SVC()
        self.model.C = C
        self.model.gamma = gamma
        self.model.kernel = kernel

    def standardize(self, X):
        """ Standardize the data.

        Args:
            X: The input vector [n_sample, n_feature].

        Returns:
            X: The input vector with standardized values.
        """

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = ((X - mean) / std)

        return X

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, value):
        return self.model.predict(value)

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)
