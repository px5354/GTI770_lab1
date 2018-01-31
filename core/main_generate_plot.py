
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from commons.helpers.graphics.plot import Plot
from sklearn import tree
from sklearn.datasets import load_iris

def main():
    galaxy_feature1_vector_path = os.environ[
                                      "VIRTUAL_ENV"] + "/data/csv/galaxy/exported_personal_feature2.csv"
    galaxy_feature2_vector_path = os.environ[
                                      "VIRTUAL_ENV"] + "/data/csv/galaxy/exported_personal_feature3.csv"
    galaxy_label_path = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/exported_personal_label_galaxy.csv"
    result_path = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/feature_comparison_"

    features1 = np.loadtxt(open(galaxy_feature1_vector_path, "rb"), delimiter=",", skiprows=0)
    features2 = np.loadtxt(open(galaxy_feature2_vector_path, "rb"), delimiter=",", skiprows=0)
    labels = np.loadtxt(open(galaxy_label_path, "rb"), delimiter=",", skiprows=0)

    X = np.vstack((features1, features2)).T#features1 #[[0, 0], [1, 1], [2, 2]]
    Y = labels
    # iris = load_iris()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    # clf = clf.fit(iris.data, iris.target)
    result = clf.predict([[5.]])
    # result = clf.predict(iris.data[:1, :])
    print(result)
    # plt.scatter(features1, labels)
    # plt.savefig(os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/test.png")

    # for i in range(0, 2):
    # Plot.plot_feature_comparison(features1, features2, labels, result_path +"0.png")

    # for i in range(0, 3):
    #     Plot.plot_feature_comparison(features1[:, i], features2[:, 0], labels, result_path + str(i) + ".png")



if __name__ == '__main__':
    main()