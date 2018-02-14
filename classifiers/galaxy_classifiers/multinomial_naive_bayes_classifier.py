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

from sklearn.naive_bayes import MultinomialNB


class MultinomialNaiveBayesClassifier(object):
    """ A Naive Bayes Classifier object."""

    def __init__(self, alpha=1.0, fit_prior=False, class_prior=None):
        self.model = MultinomialNB(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)

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