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

Notes : This file is a file that will use
"""

import os
import numpy as np
from commons.helpers.graphics.plot import Plot
from sklearn import tree
from sklearn.model_selection import train_test_split
import operator

def main():
    galaxy_feature1_vector_path = os.environ[
                                      "VIRTUAL_ENV"] + "/data/csv/galaxy/exported_personal_feature1.csv"
    galaxy_feature2_vector_path = os.environ[
                                      "VIRTUAL_ENV"] + "/data/csv/galaxy/exported_personal_feature3.csv"
    galaxy_feature3_vector_path = os.environ[
                                      "VIRTUAL_ENV"] + "/data/csv/galaxy/exported_personal_feature4.csv"
    galaxy_feature4_vector_path = os.environ[
                                      "VIRTUAL_ENV"] + "/data/csv/galaxy/exported_personal_feature2.csv"
    galaxy_label_path = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/exported_personal_label_galaxy.csv"
    result_path = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/feature_comparison_"
    result_tree_path = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/decision_tree.png"

    features1 = np.loadtxt(open(galaxy_feature1_vector_path, "rb"), delimiter=",", skiprows=0)
    features2 = np.loadtxt(open(galaxy_feature2_vector_path, "rb"), delimiter=",", skiprows=0)
    features3 = np.loadtxt(open(galaxy_feature3_vector_path, "rb"), delimiter=",", skiprows=0)
    features4 = np.loadtxt(open(galaxy_feature4_vector_path, "rb"), delimiter=",", skiprows=0)
    labels = np.loadtxt(open(galaxy_label_path, "rb"), delimiter=",", skiprows=0)

    # max_index, max_value = max(enumerate(features4), key=operator.itemgetter(1))

    feature_vector = np.vstack((features1, features2))
    # feature_vector = feature_vector[::-1]
    feature_vector = feature_vector.reshape((feature_vector.shape[1], feature_vector.shape[0]))
    print(feature_vector.shape)
    # feature_vector = feature_vector[:, None]

    X = np.vstack((features1, features2, features3, features4)).T#features1 #[[0, 0], [1, 1], [2, 2]]
    y = labels
    # iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 0)

    clf = tree.DecisionTreeClassifier(max_depth = None)
    clf = clf.fit(X_train, y_train)
    # clf = clf.fit(iris.data, iris.target)
    result = clf.score(X_test, y_test)
    # result = clf.predict(iris.data[:1, :])
    print("max depth None")
    print(result)

    clf = tree.DecisionTreeClassifier(max_depth = 2)
    clf = clf.fit(X_train, y_train)
    # clf = clf.fit(iris.data, iris.target)
    result = clf.score(X_test, y_test)
    # result = clf.predict(iris.data[:1, :])
    print("max depth 2")
    print(result)

    clf = tree.DecisionTreeClassifier(max_depth = 3)
    clf = clf.fit(X_train, y_train)
    # clf = clf.fit(iris.data, iris.target)
    result = clf.score(X_test, y_test)
    # result = clf.predict(iris.data[:1, :])
    print("max depth 3")
    print(result)

    clf = tree.DecisionTreeClassifier(max_depth = 4)
    clf = clf.fit(X_train, y_train)
    # clf = clf.fit(iris.data, iris.target)
    result = clf.score(X_test, y_test)
    # result = clf.predict(iris.data[:1, :])
    print("max depth 4")
    print(result)

    clf = tree.DecisionTreeClassifier(max_depth = 5)
    clf = clf.fit(X_train, y_train)
    # clf = clf.fit(iris.data, iris.target)
    result = clf.score(X_test, y_test)
    # result = clf.predict(iris.data[:1, :])
    print("max depth 5")
    print(result)

    clf = tree.DecisionTreeClassifier(max_depth=10)
    clf = clf.fit(X_train, y_train)
    # clf = clf.fit(iris.data, iris.target)
    result = clf.score(X_test, y_test)
    # result = clf.predict(iris.data[:1, :])
    print("max depth 10")
    print(result)

    clf = tree.DecisionTreeClassifier(max_depth=10)
    clf = clf.fit(X_train, y_train)
    noise = np.random.normal(0, 0.05, [X_test.shape[0], X_test.shape[1]])
    # clf = clf.fit(iris.data, iris.target)
    result = clf.score(X_test + noise, y_test)
    # result = clf.predict(iris.data[:1, :])
    print("noise 5%")
    print(result)

    noise = np.random.normal(0, 0.10, [X_test.shape[0], X_test.shape[1]])
    # clf = clf.fit(iris.data, iris.target)
    result = clf.score(X_test + noise, y_test)
    # result = clf.predict(iris.data[:1, :])
    print("noise 10%")
    print(result)

    noise = np.random.normal(0, 0.15, [X_test.shape[0], X_test.shape[1]])
    # clf = clf.fit(iris.data, iris.target)
    result = clf.score(X_test + noise, y_test)
    # result = clf.predict(iris.data[:1, :])
    print("noise 15%")
    print(result)



    # Plot.plot_tree_decision_surface(X, labels, ["RB_ratio", "entropy", "gini", "light radius diff"], ["RB_ratio", "entropy", "gini", "light radius diff"], result_tree_path)

    # plt.scatter(features1, labels)
    # plt.savefig(os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/test.png")

    # for i in range(0, 2):
    # Plot.plot_feature_comparison(features1, features2, labels, result_path +"0.png")

    # for i in range(0, 3):
    #     Plot.plot_feature_comparison(features1[:, i], features2[:, 0], labels, result_path + str(i) + ".png")



if __name__ == '__main__':
    main()