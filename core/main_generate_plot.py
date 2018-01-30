
import os
import numpy as np
from commons.helpers.graphics.plot import Plot

def main():
    galaxy_feature1_vector_path = os.environ[
                                      "VIRTUAL_ENV"] + "/data/csv/galaxy/exported_personal_feature1.csv"
    galaxy_feature2_vector_path = os.environ[
                                      "VIRTUAL_ENV"] + "/data/csv/galaxy/exported_personal_feature2.csv"
    galaxy_label_path = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/exported_personal_label_galaxy.csv"
    result_path = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/feature_comparison_"

    features1 = np.loadtxt(open(galaxy_feature1_vector_path, "rb"), delimiter=",", skiprows=0)
    features2 = np.loadtxt(open(galaxy_feature2_vector_path, "rb"), delimiter=",", skiprows=0)
    labels = np.loadtxt(open(galaxy_label_path, "rb"), delimiter=",", skiprows=0)

    for i in range(0, 2):
        Plot.plot_feature_comparison(features1[:10000], features2[:, i][:10000], labels[:10000], result_path + str(i) + ".png")

    # for i in range(0, 3):
    #     Plot.plot_feature_comparison(features1[:, i], features2[:, 0], labels, result_path + str(i) + ".png")

if __name__ == '__main__':
    main()