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
"""

from sklearn.tree import DecisionTreeClassifier


class TreeClassifier(object):
    """ An object containing a decision tree classifier. """

    def __init__(self, max_depth):
        self.model = DecisionTreeClassifier(max_depth=max_depth)

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
