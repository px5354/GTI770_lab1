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

Note : This file contains everything for extracting the features from galaxies

"""

import cv2
import os
import math
import numpy as np
import scipy.ndimage as nd
from PIL import ImageEnhance, Image
from scipy.stats.mstats import mquantiles

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
        labels = list()

        for sample, label in zip(dataset.train._img_names, dataset.train._labels):

            # Get the file name associated with the galaxy ID.
            file = self.get_image_path() + str(sample[0]) + ".jpg"

            # Compute the features and append to the list.
            features_array, label = self.get_features(file, sample[0], label[0])
            num_of_features = len(features_array)
            if len(features) != num_of_features:
                features = [[] for _ in range(num_of_features)]

            for i in range(0, num_of_features):
                features[i].append(features_array[i])

            labels.append(label)

        for sample, label in zip(dataset.valid._img_names, dataset.valid._labels):

            # Get the file name associated with the galaxy ID.
            file = self.get_image_path() + str(sample[0]) + ".jpg"

            # Compute the features and append to the list.
            feature_array, label = self.get_features(file, sample[0], label[0])

            features[i].append(features_array[i])

            labels.append(label)

        return features, labels

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

    def white_image(self, image):
        """ Change the colors of an image.

        Using openCV method to change colors from an image.

        Args:
            image: an OpenCV standard image format.

        Returns:
            The image in black and white a.k.a gray scale.
        """

        gray = cv2.cvtColor(image.astype("uint8"), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        return thresh

    def crop_image_with_extremes(self, image, extraspace):
        """ Crop the image with extremes values.

        Using openCV filters and thresh to help find contours.
        Using openCV contours to detect the shape.
        Using openCV method to crop the image from the extremes values from left, right, top and bottom.

        Args:
            image: an OpenCV standard image format.
            extraSpace: a value to add to the extremes.

        Returns:
            The image cropped from extremes values.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 0:
            cnts = 0
        if len(cnts[1]) == 0:
            cnts = 0
        if cnts is None:
            cnts = 0
        cnts = cnts[1]
        c = max(cnts, key=cv2.contourArea)

        # determine the most extreme points along the contour
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        lenght_DimX = len(image)
        lenght_DimY = len(image[0])

        #extraSpace is used here because sometimes we want more than the fit size of the shape.
        #So, we verify to make sure it's not going outside of the screen.
        x1 = extLeft[0] - extraspace
        if x1 < 0:
            x1 = extLeft[0]

        x2 = extRight[0] + extraspace
        if x2 > lenght_DimX:
            x1 = extRight[0]

        y1 = extTop[1] - extraspace
        if y1 < 0:
            y1 = extTop[1]

        y2 = extBot[1] + extraspace
        if y2 > lenght_DimY:
            y2 = extBot[1]

        image = image[y1:y2, x1:x2]

        return image

    def correlation(self, image):
        """ Give the correlation of the image.

        Using numpy corrcoef methods to give the correlation between two array of data.

        Args:
            image: a black and white image.

        Returns:
            The correlation of white in the image.
        """

        points = image.astype('float')

        rows, cols = image.shape

        listX = []
        listY = []

        for x in range(rows):
            for y in range(cols):
                if points[x][y] == 255:
                    listX.append(x)
                    listY.append(y)

        correlation = np.corrcoef(listX, listY)[0, 1]

        return correlation

    def remove_little_shapes(self, image):
        """ Removing little shapes in the image.

        Using OpenCV connectedComponentsWithStats to get all the shapes

        Args:
            image: a black and white image.

        Returns:
            The image with one shape and nothing around it.
        """

        components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
        sizes = stats[1:, -1];

        img2 = np.zeros((output.shape))

        for i in range(0, components - 1):
            if sizes[i] == max(sizes):
                img2[output == i + 1] = 255

        return img2

    def get_black_proportion(self, image):
        """ GIve proportion of black in the image.

        Using some image proporties to calculate the proportion

        Args:
            image: a black and white image.

        Returns:
            The proportion value of black in %.
        """

        points = image.astype('float')
        if(image.shape is None):
            return 0
        rows, cols = image.shape
        black = 0
        count = 0

        for x in range(rows):
            for y in range(cols):
                count += 1
                if points[x][y] == 0:
                    black += 1

        proportion = black / count * 100

        return proportion

    def get_circularity(self, image):
        """ Give the circularity of the shape in the image.

        Using OpenCV findContours to get the shape of the image.
        Using OpenCV contourArea to get the area.
        Using OpenCV arcLength to get the perimeters of the shape.

        Args:
            image: an OpenCV standard image format.

        Returns:
            The circularity value of the shape in the image in %.
        """

        white_image = self.white_image(image)
        if white_image is None:
            return 0
        im2, contours, hierarchy = cv2.findContours(white_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return 0
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        circularity = (4 * math.pi * area) / (perimeter * perimeter) * 100

        return circularity

    def get_aspect_ratio(self, image):
        """ Give the aspect ratio of the shape in the image.

        Using OpenCV findContours to get the shape of the image.
        Using OpenCV minAreaRect to get the width and height.
        From that, we compute the ratio between width and height.

        Args:
            image: an OpenCV standard image format.

        Returns:
            The aspect ratio value of the shape in the image.
        """
        white_image = self.white_image(image)
        if white_image is None:
            return 0
        im2, contours, hierarchy = cv2.findContours(white_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return 0
        cnt = contours[0]
        rect = cv2.minAreaRect(cnt)
        width = rect[1][0]
        height = rect[1][1]
        if (width == 0) or (height == 0):
            return 0
        else:
            ar = width/height

        return ar


    def get_non_zero_histogram_indexes(self, color_histogram, color):
        """ Give the bin container with at least one pixel from a color histogram
        for a specific color

        Args:
            color_histogram: an OpenCV standard color histogram.
            color: the name of a specific color (blue, green or red)

        Returns:
            An array with bin containers with at least one pixel.
        """
        if(color == "blue"):
            colorIndex = 0
        if (color == "green"):
            colorIndex = 1
        if (color == "red"):
            colorIndex = 2

        histogramIndexes = np.nonzero(color_histogram[colorIndex])[0]

        return histogramIndexes

    def get_non_zero_histogram_values(self, color_histogram, color):
        """ Give the numbers of pixels for each bin container without
         zero values from a color histogram for a specific color

        Args:
            color_histogram: an OpenCV standard color histogram.
            color: the name of a specific color (blue, green or red)

        Returns:
            An array with numbers of pixels for each bin container without zero values
        """

        if (color == "blue"):
            colorIndex = 0
        if (color == "green"):
            colorIndex = 1
        if (color == "red"):
            colorIndex = 2

        histogramIndexes = self.get_non_zero_histogram_indexes(color_histogram, color)
        histogramValues = color_histogram[colorIndex][histogramIndexes]

        return histogramValues

    def get_histogram_pixel_count(self, color_histogram, color):

        pixelCount = len(self.get_non_zero_histogram_values(color_histogram, color))

        return pixelCount

    def get_mean(self, color_histogram, color):

        mean = np.mean(self.get_non_zero_histogram_values(color_histogram, color))

        return mean

    def get_img_contrast(self, image_file,enhanceNumber):

        pil_image = Image.open(image_file).convert('RGB')
        converter = ImageEnhance.Color(pil_image)
        pil_image = converter.enhance(enhanceNumber)
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR
        img_color_contrast = open_cv_image[:, :, ::-1].copy()

        return img_color_contrast

    def features_color_RB_ratio(self, image):
        """ Color features

        Using home made methods to prepare the image.

        Args:
            image: an OpenCV standard image format.

        Returns:
            the max blue bin, the max red bin, the red to blue ratio and red to blue standard deviation
        """

        img_color = image
        img_color = self.gaussian_filter(img_color, -100, -100)
        img_color = self.remove_starlight(img_color, self.get_gray_image(img_color))
        clean_img = self.crop_image_with_extremes(img_color, 0)

        # when the crop doesnt work
        if (clean_img is None):
            return 0

        color_histogram = self.get_color_histogram(img_color=clean_img)

        non_zero_blue = self.get_non_zero_histogram_values(color_histogram, "blue")
        non_zero_red = self.get_non_zero_histogram_values(color_histogram, "red")

        mean_blue = self.get_mean(color_histogram, "blue")
        mean_red = self.get_mean(color_histogram, "red")
        if mean_blue == 0:
            RB_ratio = 0
        else:
            RB_ratio = mean_red / mean_blue

        return RB_ratio

    def features_color_std_RB_ratio(self, image):
        """ Color features

        Using home made methods to prepare the image.

        Args:
            image: an OpenCV standard image format.

        Returns:
            the max blue bin, the max red bin, the red to blue ratio and red to blue standard deviation
        """

        img_color = image
        img_color = self.gaussian_filter(img_color, -100, -100)
        img_color = self.remove_starlight(img_color, self.get_gray_image(img_color))
        clean_img = self.crop_image_with_extremes(img_color, 0)

        if (clean_img is None):
            return 0

        color_histogram = self.get_color_histogram(img_color=clean_img)

        non_zero_blue = self.get_non_zero_histogram_values(color_histogram, "blue")
        non_zero_red = self.get_non_zero_histogram_values(color_histogram, "red")

        std_red = non_zero_red.std()
        std_blue = non_zero_blue.std()

        if std_blue == 0:
            std_RB_ratio = 0
        else:
            std_RB_ratio = std_red / std_blue

        return std_RB_ratio

    def feature_circularity(self, image):
        """ Feature to give the circularity

        Using home made methods to prepare the image.

        Args:
            image: an OpenCV standard image format.

        Returns:
            The circularity value in %.
        """

        temp_path = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/temp_image.jpg"
        cropped_img = self.crop_image_with_extremes(image, 0)

        if (cropped_img is None):
            return 0

        white_image = self.white_image(cropped_img)
        clean_image = self.remove_little_shapes(white_image)
        cv2.imwrite(temp_path, clean_image)
        final_image = cv2.imread(temp_path)
        circularity = self.get_circularity(final_image)
        cv2.imwrite(os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/circularity_image.jpg", final_image)

        return circularity

    def feature_correlation(self, image):
        """ Feature to give the correlation

        Using home made methods to prepare the image.

        Args:
            image: an OpenCV standard image format.

        Returns:
            The correlation value.
        """

        cropped_img = self.crop_image_with_extremes(image, 0)

        if (cropped_img is None):
            return 0

        white_image = self.white_image(cropped_img)
        clean_image = self.remove_little_shapes(white_image)
        correlation = self.correlation(clean_image)

        cv2.imwrite(os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/correlation_image.jpg", clean_image)

        return correlation

    def feature_aspect_ratio(self, image):
        """ Feature to give the aspect ratio

        Using home made methods to prepare the image.

        Args:
            image: an OpenCV standard image format.

        Returns:
            The aspect ratio.
        """

        temp_path = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/temp_image.jpg"
        cropped_img = self.crop_image_with_extremes(image, 0)

        if (cropped_img is None):
            return 0

        white_image = self.white_image(cropped_img)
        clean_image = self.remove_little_shapes(white_image)
        cv2.imwrite(temp_path, clean_image)
        final_image = cv2.imread(temp_path)
        ar = self.get_aspect_ratio(final_image)

        cv2.imwrite(os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/aspect_ratio_image.jpg", clean_image)

        return ar

    def feature_black_proportion(self, image):
        """ Feature to give the black proportion

        Using home made methods to prepare the image

        Args:
            image: an OpenCV standard image format.

        Returns:
            The black proportion value in %.
        """

        cropped_img = self.crop_image_with_extremes(image, 0)
        white_image = self.white_image(cropped_img)
        if white_image is None:
            return 0
        clean_image = self.remove_little_shapes(white_image)
        proportion = self.get_black_proportion(clean_image)

        return proportion

    def feature_entropy(self, image):
        """ Feature to give the entropy

        Using home made methods to prepare the image

        Args:
            image: an OpenCV standard image format.

        Returns:
            The entropy of the image.
        """

        cropped_img = self.crop_image_with_extremes(image, 0)
        entropy = self.get_entropy(self.get_gray_image(cropped_img))

        return entropy

    def feature_gini(self, image):
        """ Feature to give the gini coefficient

        Using home made methods to prepare the image

        Args:
            image: an OpenCV standard image format.

        Returns:
            The gini coefficient of the image.
        """

        cropped_img = self.crop_image_with_extremes(image, 0)
        if (cropped_img is None):
            return 0
        gini = self.gini(self.get_gray_image(cropped_img))

        return gini

    def get_features(self, image_file, id, label):
        """ Get the image's features.

        A wrapping method to get the image's features.

        Place your code here.

        Args:
            image_file: the image's file being processed.

        Returns:
            features: an array of feature vectors of N dimensions for N features.
        """

        print("Processing file : " + image_file)

        original_image = cv2.imread(filename=image_file)

        RB_ratio = self.features_color_RB_ratio(original_image)
        std_RB_ratio = self.features_color_std_RB_ratio(original_image)
        circularity = self.feature_circularity(original_image)
        black_proportion = self.feature_black_proportion(original_image)
        aspect_ratio = self.feature_aspect_ratio(original_image)
        entropy = self.feature_entropy(original_image)

        # Declare a list for storing computed features.
        features1 = list()
        features2 = list()
        features3 = list()
        features4 = list()
        features5 = list()
        features6 = list()

        features1 = np.append(features1, RB_ratio)
        features2 = np.append(features2, std_RB_ratio)
        features3 = np.append(features3, circularity)
        features4 = np.append(features4, black_proportion)
        features5 = np.append(features5, aspect_ratio)
        features6 = np.append(features6, entropy)

        features = [features1, features2, features3, features4, features5, features6]

        return features, label
