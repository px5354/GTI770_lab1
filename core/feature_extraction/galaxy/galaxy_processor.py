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

import cv2
import os
import math
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from PIL import ImageEnhance, Image
from scipy.stats.mstats import mquantiles, kurtosis, skew

class GalaxyProcessor(object):
    """ Process galaxy images and extract the features."""

    def __init__(self, image_path):
        self._image_path = image_path

    def get_image_path(self):
        return self._image_path

    def process_galaxy(self, dataset):
        """ Process a galaxy image.

        Get all the features from a galaxy image.

        Args:
            galaxy_id: a string containing the galaxy ID

        Returns:
             An array containing the image's features.
        """
        features = list()
        features1 = list()
        features2 = list()
        features3 = list()
        features4 = list()
        features5 = list()
        features6 = list()
        features7 = list()
        features8 = list()
        labels = list()

        for sample, label in zip(dataset.train._img_names, dataset.train._labels):

            # Get the file name associated with the galaxy ID.
            file = self.get_image_path() + str(sample[0]) + ".jpg"

            # Compute the features and append to the list.
            # feature_vector = self.get_features(file, sample[0], label[0])
            # features.append(feature_vector)

            features_array, label = self.get_features(file, sample[0], label[0])
            features1.append(features_array[0])
            features2.append(features_array[1])
            features3.append(features_array[2])
            features4.append(features_array[3])
            features5.append(features_array[4])
            features6.append(features_array[5])
            features7.append(features_array[6])
            features8.append(features_array[7])
            labels.append(label)

        for sample, label in zip(dataset.valid._img_names, dataset.valid._labels):

            # Get the file name associated with the galaxy ID.
            file = self.get_image_path() + str(sample[0]) + ".jpg"

            # Compute the features and append to the list.
            # feature_vector = self.get_features(file, sample[0], label[0])
            # features.append(feature_vector)

            feature_array, label = self.get_features(file, sample[0], label[0])
            features1.append(features_array[0])
            features2.append(features_array[1])
            features3.append(features_array[2])
            features4.append(features_array[3])
            features5.append(features_array[4])
            features6.append(features_array[5])
            features7.append(features_array[6])
            features8.append(features_array[7])
            labels.append(label)
        feature_array_final = [features1, features2, features3, features4, features5, features6]
        return feature_array_final, labels

    def load_image(self, filepath):
        """ Load an image using OpenCV library.

        Load an image using OpenCV library.

        Args:
            filepath: the path of the file to open.

        Returns:
             An image in OpenCV standard format.
        """
        return cv2.imread(filename=filepath)

    def crop_image(self, image, width, height):
        """ Crop an image.

        Utility method to crop an image using Python arrays.

        Args:
            image: a pointer to an OpenCV image matrix.
            width: the resulting width.
            height: the resulting height.

        Returns:
             A 2D array representing the cropped image.
        """
        return image[width:height, width:height]

    def gaussian_filter(self, image, kernel_width, kernel_height):
        """ Apply a gaussian filter.

        Apply a gaussian filter on an image.

        Args:
            image: an OpenCV standard image format. 
            kernel_width: the kernel width of the filter.
            kernel_height: the kernel height of the filter.

        Returns:
             The image with applied gaussian filter.
        """
        return cv2.GaussianBlur(image, (kernel_width, kernel_height), 2.0)

    def rescale(self, image, min=0, max=255):
        """ Rescale the colors of an image.

        Utility method to rescale colors from an image. 

        Args: 
            image: an OpenCV standard image format.
            min: The minimum color value [0, 255] range.
            max: The maximum color value [0, 255] range.
        
        Returns:
            The image with rescaled colors.
        """
        image = image.astype('float')
        image -= image.min()
        image /= image.max()
        image = image * (max - min) + min

        return image

    def saturate(self, image, q0=0.01, q1=0.99):
        """ Stretch contrasts of an image. 
        
        Utility method to saturate the contrast of an image. 

        Args:
            image: an OpenCV standard image format.
            q0: minimum coefficient.
            q1: maximum coefficient.

        Returns:
            The image with saturated contrasts. 
        """
        image = image.astype('float')
        if q0 is None:
            q0 = 0
        if q1 is None:
            q1 = 1
        q = mquantiles(image[np.nonzero(image)].flatten(), [q0, q1])
        image[image < q[0]] = q[0]
        image[image > q[1]] = q[1]

        return image

    def largest_connected_component(self, image, labels, nb_labels):
        """ Select the largest connected component.

        Select the largest connected component which is closest to the center using a weighting size/distance**2.

        Args:
            image: an OpenCV standard image format.
            labels: image labels.
            nb_labels: number of image labels.

        Returns: 
            A thresholded image of the largest connected component.
        """
        sizes = np.bincount(labels.flatten(),
                            minlength=nb_labels + 1)
        centers = nd.center_of_mass(image, labels, range(1, nb_labels + 1))

        distances = list(map(lambda args:
                             (image.shape[0] / 2 - args[1]) ** 2 + (image.shape[1] / 2 - args[0]) ** 2,
                             centers))

        distances = [1.0] + distances
        distances = np.array(distances)
        sizes[0] = 0
        sizes[sizes < 20] = 0
        sizes = sizes / (distances + 0.000001)
        best_label = np.argmax(sizes)
        thresholded = (labels == best_label) * 255

        return thresholded

    def recenter(self, image, x, y, interpolation=cv2.INTER_LINEAR):
        """ Recenter an image. 

        Recenter an image around x and y.

        Args: 
            image: an OpenCV standard image format.
            x: integer representing an "X" coordinate.
            y: integer representing an "Y" coordinate.
            interpoolation: interpolation method.

        Returns:
            The recentered image.
        """
        cx = float(image.shape[1]) / 2
        cy = float(image.shape[0]) / 2

        # Compute the translation matrix.
        translation_matrix = np.array([[1, 0, cx - x], [0, 1, cy - y]], dtype='float32')

        # Compute the afine transform.
        recentered_image = cv2.warpAffine(image, translation_matrix, image.shape[:2], flags=interpolation)

        return recentered_image

    def compose(self, matrix1, matrix2):
        """ Composes affine transformations.
        
        Compute the resulting transformation matrix based on two supplied transformation matrix.

        Args: 
            matrix1: The first matrix transform.
            matrix2: The second matrix transform.

        Returns:
            The composition matrix of the affine transforms.
        """
        n1 = np.eye(3, dtype='float32')
        n2 = np.eye(3, dtype='float32')
        n1[:2] = matrix1
        n2[:2] = matrix2
        n3 = np.dot(n1, n2)

        return n3[:2]

    def rotate(self, image, x, y, angle, interpolation=cv2.INTER_LINEAR):
        """ Rotate an image.

        Rotate an image by an angle in degrees around specific point.
        
        Source : http://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point

        Args:
            image: an OpenCV standard image format.
            x: integer representing an "X" coordinate
            y: integer representing an "Y" coordinate.
            angle: the angle of rotation.
            interpolation: interpolation method. 

        Returns:
            The rotated image.
        """
        # Get the image center.
        cx = float(image.shape[1]) / 2
        cy = float(image.shape[0]) / 2

        # Compute a translation matrix to recenter the image.
        translation_matrix = np.array([[1, 0, cx - x], [0, 1, cy - y]], dtype='float32')

        # Compute a rotation matrix to rotate the image.
        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        # Compose the affine transformation.
        m = self.compose(rotation_matrix, translation_matrix)

        # Compute the rotation.
        rotated_image = cv2.warpAffine(image, m, image.shape[:2], flags=interpolation)

        return rotated_image

    def random_colors(self, labels):
        """ Color with random colors components in an image.
        
        For debug purpose. 

        Args: 
            labels: some image labels.

        Returns:
            Colored image segment.
        """
        idx = np.nonzero(labels)
        nb_labels = labels.max()
        colors = np.random.random_integers(0, 255, size=(nb_labels + 1, 3))
        seg = np.zeros((labels.shape[0], labels.shape[1], 3), dtype='uint8')
        seg[idx] = colors[labels[idx].astype('int')]

        return seg

    def fit_ellipse(self, points, factor=1.96):
        """  Fit points to ellipse.

        Fit an ellips to points passed in parameters. 
        
        Theorical source : http://en.wikipedia.org/wiki/1.96

        Args:
            points: image points. 
            factor: the 1.96 factor in order to contain 95% of the galaxy.

        Returns:
            The center of the ellipse, the singular values, and the angle.
        
        """
        points = points.astype('float')
        center = points.mean(axis=0)
        points -= center

        U, S, V = np.linalg.svd(points, full_matrices=False)

        S /= np.sqrt(len(points) - 1)
        S *= factor
        angle = math.atan2(V[0, 1], V[0, 0]) / math.pi * 180

        return center, 2 * S, angle

    def gini(self, x):
        """ Get the Gini coefficient.

        The Gini coefficient (sometimes expressed as a Gini ratio or a normalized Gini index)
        is a measure of statistical dispersion and is the most commonly used measure of inequality.

        Source : http://www.ellipsix.net/blog/2012/11/the-gini-coefficient-for-distribution-inequality.html

        Args:
            x: the pixels representing an image.
            filename: filename of the image.
        """

        # requires all values in x to be zero or positive numbers, otherwise results are undefined
        x = x.flatten()
        n = len(x)
        s = x.sum()
        r = np.argsort(np.argsort(-x))  # calculates zero-based ranks
        if s == 0 or n == 0:
            return 1.0
        else:
            return 1.0 - (2.0 * (r * x).sum() + s) / (n * s)

    def get_entropy(self, image):
        """ Get the image's entropy.

        Entrpy is a scalar value representing the entropy of grayscale image.
        Entropy is a statistical measure of randomness that can be used to characterize 
        the texture of the input image.

        Source : http://stackoverflow.com/questions/16647116/faster-way-to-analyze-each-sub-window-in-an-image

        Args: 
            image: an OpenCV standard image format.

        Returns:
            Image's entropy as a floating point value.
        """
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        logs = np.log2(hist + 0.00001)

        return -1 * (hist * logs).sum()

    def get_gray_float_image(self, image):
        """ get image as grey scale image in float format.

        Transform the image into grey scale and transform values into floating point integer.

        Args:
            image: an OpenCV standard color image format.

        Returns:
             Image in gray scale in floating point values.
        """
        return cv2.cvtColor(image.astype("uint8"), cv2.COLOR_BGR2GRAY).astype("float")

    def get_gray_image(self, image):
        """ Get an image in gray scales.

        Transform an image in grey scale. Returns values as integer.

        Args:
            image : an OpenCV standard image format.

        Returns:
             Image in gray scale.
        """
        return cv2.cvtColor(image.astype("uint8"), cv2.COLOR_BGR2GRAY)

    def remove_starlight(self, image_color, image_gray):
        """ Removes the star light in images.

        Calclates the median in color and gray scale image to clean the image's background.

        Args:
             image_color: an OpenCV standard color image format.
             image_gray: an OpenCV standard gray scale image format.

        Returns:
            An image cleaned from star light.
        """
        t = np.max(np.median(image_color[np.nonzero(image_gray)], axis=0))
        image_color[image_color < t] = t

        return self.rescale(image_color).astype("uint8")

    def get_center_of_mass(self, image, labels, nb_labels):
        """ Get the center of mass of the galaxy.


        Args:
            image: an OpenCV standard gray scale image format
            labels: the image's labels
            nb_labels: the image's number of labels

        Returns:
            The thresholded galaxy image with it's center coordinates.
        """
        thresholded = self.largest_connected_component(image=image, labels=labels, nb_labels=nb_labels)
        center = nd.center_of_mass(image, thresholded)

        return thresholded, center

    def get_light_radius(self, image, r=[0.1, 0.8]):
        """ Get the radius of the light in the image.

        Args:
            image: an OpenCV standard gray scale image format
            r: probability list

        Returns:
            The light radius as a floating point value.
        """
        image = image.astype('float')
        idx = np.nonzero(image)
        s = image[idx].sum()
        mask = np.ones(image.shape)
        mask[int(image.shape[0] / 2), int(image.shape[1] / 2)] = 0
        edt = nd.distance_transform_edt(mask)
        edt[edt >= image.shape[1] / 2] = 0
        edt[image == 0] = 0
        q = mquantiles(edt[np.nonzero(edt)].flatten(), r)
        res = []
        for q0 in q:
            res.append(image[edt < q0].sum() / s)

        return res

    def get_color_histogram(self, img_color):
        """ Get the color histograms from a color image.

        Args:
            img_color: an OpenCV standard color image format.

        Returns:
            The BGR color histograms.
        """
        blue_histogram = cv2.calcHist(img_color, [0], None, [256], [0, 256])
        green_histogram = cv2.calcHist(img_color, [1], None, [256], [0, 256])
        red_histogram = cv2.calcHist(img_color, [2], None, [256], [0, 256])

        return np.array([blue_histogram, green_histogram, red_histogram])

    def gerv_shits(self, image):
        img_gray = self.get_gray_float_image(image)
        image = self.remove_starlight(image, img_gray)
        image = self.crop_image_with_extremes(image)
        cv2.imwrite(os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/image.jpg", image)

        img = cv2.imread(os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/image.jpg")
        thresh = self.white_image(img)
        cv2.imwrite(os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/thresh.jpg", thresh)

        # ret, thresh = cv2.threshold(thresh, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnt = contours[0]

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (124, 252, 0), 2)

        cv2.imwrite(os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/img_coutour.jpg", img)

    def white_image(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        return thresh

    def crop_image_with_extremes(self, image):

        # load the image, convert it to grayscale, and blur it slightly
        white_image = self.white_image(image)

        # find contours in thresholded image, then grab the largest
        # one
        cnts = cv2.findContours(white_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]
        c = max(cnts, key=cv2.contourArea)

        # determine the most extreme points along the contour
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        # draw the outline of the object, then draw each of the
        # extreme points, where the left-most is red, right-most
        # is green, top-most is blue, and bottom-most is teal
        # cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
        # cv2.circle(image, extLeft, 4, (0, 0, 255), -1)
        # cv2.circle(image, extRight, 4, (0, 255, 0), -1)
        # cv2.circle(image, extTop, 4, (255, 0, 0), -1)
        # cv2.circle(image, extBot, 4, (255, 255, 0), -1)

        points = image
        lenght_DimX = len(points)
        lenght_DimY = len(points[0])

        extraSpace = 0

        x1 = extLeft[0] - extraSpace
        if x1 < 0:
            x1 = extLeft[0]

        x2 = extRight[0] + extraSpace
        if x2 > lenght_DimX:
            x1 = extRight[0]

        y1 = extTop[1] - extraSpace
        if y1 < 0:
            y1 = extTop[1]

        y2 = extBot[1] + extraSpace
        if y2 > lenght_DimY:
            y2 = extBot[1]

        image = image[y1:y2, x1:x2]

        return image

    def correlation_image(self, image):
        points = image.astype('float')

        lenght_DimX = len(points)
        lenght_DimY = len(points[0])

        listX = []
        listY = []

        for x in range(0, lenght_DimX - 1):
            for y in range(0, lenght_DimY - 1):
                if points[x][y] > 250:
                    listX.append(x)
                    listY.append(y)

        correlation = np.corrcoef(listX, listY)[0, 1]
        return correlation

    def get_nonZeroHistogramIndexes(self, color_histogram, color):


        if(color == "blue"):
            colorIndex = 0
        if (color == "green"):
            colorIndex = 1
        if (color == "red"):
            colorIndex = 2

        histogramIndexes = np.nonzero(color_histogram[colorIndex])[0]

        return histogramIndexes

    def get_nonZeroHistogramValues(self, color_histogram, color):

        if (color == "blue"):
            colorIndex = 0
        if (color == "green"):
            colorIndex = 1
        if (color == "red"):
            colorIndex = 2

        histogramIndexes = self.get_nonZeroHistogramIndexes(color_histogram, color)

        histogramValues = color_histogram[colorIndex][histogramIndexes]

        return histogramValues

    def get_histogramPixelCount(self, color_histogram, color):

        pixelCount = len(self.get_nonZeroHistogramValues(color_histogram, color))

        return pixelCount

    def get_mean(self, color_histogram, color):

        mean = np.mean(self.get_nonZeroHistogramValues(color_histogram, color))

        return mean

    def get_img_contrast(self, image_file,enhanceNumber):

        pil_image = Image.open(image_file).convert('RGB')
        converter = ImageEnhance.Color(pil_image)
        pil_image = converter.enhance(enhanceNumber)
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR
        img_color_contrast = open_cv_image[:, :, ::-1].copy()


        return img_color_contrast

    def remove_little_area(self, image):
        # find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

        # connectedComponentswithStats yields every seperated component with information on each of them, such as size
        # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1];
        nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        min_size = 350

        # your answer image
        img2 = np.zeros((output.shape))
        # for every component in the image, you keep it only if it's above min_size

        bigsize = max(sizes)

        for i in range(0, nb_components):
            if sizes[i] >= bigsize - 20:
                img2[output == i + 1] = 255

        return img2

    def circularity(self, image):

        white_image = self.white_image(image)
        im2, contours, hierarchy = cv2.findContours(white_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnt = contours[0]

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        c = (4 * math.pi * area) / (perimeter * perimeter)

        return c

    def get_aspect_ratio(self, image):

        height, width = image.shape

        return width / height


    def get_features(self, image_file, id, label):
        """ Get the image's features.

        A wrapping method to get the image's features.

        Place your code here.

        Args:
            image_file: the image's file being processed.

        Returns:
            features: a feature vector of N dimensions for N features.
        """

        print("Processing file : " + image_file)

        # Declare a list for storing computed features.
        features = list()
        features1 = list()
        features2 = list()
        features3 = list()
        features4 = list()
        features5 = list()
        features6 = list()

        # CONTRAST
        # img_color_contrast = self.get_img_contrast(image_file,1.5)
        # cv2.imwrite(os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/contrast_img.jpg", img_color_contrast)


        img_color = cv2.imread(filename=image_file)
        # test_ccv = ccv(img_color)
        height, width, dim = img_color.shape
        cv2.imwrite(os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/before_img.jpg", img_color)
        img_color = self.gaussian_filter(img_color, -100, -100)
        img_color = self.remove_starlight(img_color, self.get_gray_image(img_color))
        img_color = cv2.fastNlMeansDenoisingColored(img_color, None, 2, 2, 7, 21)


        clean_img = self.crop_image_with_extremes(img_color)

        white_img = self.white_image(clean_img)
        white_img = self.remove_little_area(white_img)


        cv2.imwrite(os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/after_img.jpg", clean_img)
        cv2.imwrite(os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/white_img.jpg", white_img)

        # white_img = cv2.imread(os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/white_img.jpg")

        # A feature given to student as example. Not used in the following code.
        color_histogram = self.get_color_histogram(img_color=clean_img)

        fig1 = plt.figure()
        plt.plot(color_histogram[0])
        fig1.savefig(os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/blue.png")


        fig1 = plt.figure()
        plt.plot(color_histogram[2])
        fig1.savefig(os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/red.png")

        non_zero_blue_indexes = self.get_nonZeroHistogramIndexes(color_histogram, "blue")
        non_zero_blue = self.get_nonZeroHistogramValues(color_histogram, "blue")
        px_blue_count = self.get_histogramPixelCount(color_histogram, "blue")

        max_blue_x = non_zero_blue_indexes.max()
        max_blue = color_histogram[0].max()

        non_zero_red_indexes = self.get_nonZeroHistogramIndexes(color_histogram, "red")
        non_zero_red = self.get_nonZeroHistogramValues(color_histogram, "red")
        px_red_count = self.get_histogramPixelCount(color_histogram, "red")

        max_red_x = non_zero_red_indexes.max()
        max_red = color_histogram[2].max()


        mean_blue = self.get_mean(color_histogram, "blue")

        mean_red = self.get_mean(color_histogram, "red")

        std_red = non_zero_red.std()
        std_blue = non_zero_blue.std()

        std_RB_ratio = std_red / std_blue

        RB_ratio = mean_red / mean_blue



        AR = self.get_aspect_ratio(white_img)

        entropy = self.get_entropy(self.get_gray_image(clean_img))
        light_radius = self.get_light_radius(self.get_gray_image(clean_img))
        light_radius_diff = light_radius[1] - light_radius[0]
        # fitted_ellipse_center, fitted_ellipse_singular_values, fitted_ellipse_angle = self.fit_ellipse(self.get_gray_float_image(clean_img))
        gini_coeff = self.gini(self.get_gray_image(clean_img))

        # mean, eigvec = cv2.PCACompute(matrix_test, mean=None)

        center_y, center_x = int(height/2) - 1, int(width/2) - 1
        # test = np.array([clean_img[center_y][center_x][0], clean_img[center_y][center_x][1], clean_img[center_y][center_x][2]])
        features1 = np.append(features1, RB_ratio)
        features2 = np.append(features2, light_radius_diff)
        features3 = np.append(features3, entropy)
        features4 = np.append(features4, gini_coeff)
        features5 = np.append(features5, AR)
        features6 = np.append(features6, std_RB_ratio)
        features7 = np.append(features5, max_blue_x)
        features8 = np.append(features5, max_red_x)

        features_array = [features1, features2, features3, features4, features5, features6, features7, features8]

        # features = np.append(features, features1)
        # features = np.append(features, features2)
        # sat_img = self.saturate(img_color, 0.95, 1)
        # cv2.imwrite(os.environ["VIRTUAL_ENV"] +"/data/csv/galaxy/sat_img.jpg", sat_img)
        # return features
        return features_array, label
