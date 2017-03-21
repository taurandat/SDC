import numpy as np
import cv2
from skimage.feature import hog


class FeatureExtractor(object):
    """
    Helps extracting features from an image in regions.
    """

    def __init__(self, image, orient=10, pix_per_cell=8, cell_per_block=2):
        """
        Initialises an instance.
        """
        self.image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        (self.h, self.w, self.d) = self.image.shape
        self.hog_features = []
        self.pix_per_cell = pix_per_cell
        for channel in range(self.d):
            self.hog_features.append(
                hog(self.image[:, :, channel],
                    orientations=orient,
                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block),
                    transform_sqrt=True,
                    visualise=False,
                    feature_vector=False)
            )
        self.hog_features = np.asarray(self.hog_features)

    def hog(self, x, y, k):
        """
        Gets HoG features for specified region of the image.
        """
        hog_k = (k // self.pix_per_cell) - 1

        hog_x = max((x // self.pix_per_cell) - 1, 0)
        hog_x = self.hog_features.shape[2] - hog_k \
            if hog_x + hog_k > self.hog_features.shape[2] \
            else hog_x

        hog_y = max((y // self.pix_per_cell) - 1, 0)
        hog_y = self.hog_features.shape[1] - hog_k \
            if hog_y + hog_k > self.hog_features.shape[1] \
            else hog_y

        return np.ravel(self.hog_features[:, hog_y:hog_y + hog_k, hog_x:hog_x + hog_k, :, :, :])

    def bin_spatial(self, image, size=(16, 16)):
        """
        Computes spatial vector.
        """
        return cv2.resize(image, size).ravel()

    # Define a function to compute color histogram features
    def color_hist(self, image, nbins=16, bins_range=(0, 256)):
        """
        Computes feature vector based on color channel histogram.
        """
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(image[:, :, 0],
                                     bins=nbins,
                                     range=bins_range)
        channel2_hist = np.histogram(image[:, :, 1],
                                     bins=nbins,
                                     range=bins_range)
        channel3_hist = np.histogram(image[:, :, 2],
                                     bins=nbins,
                                     range=bins_range)

        # Concatenate the histograms into a single feature vector
        return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    def feature_vector(self, x=0, y=0, k=64):
        """
        Calculates combined feature vector based on spatial, color histogram and Hog features for specified region.
        Region defaults to entire image.
        """
        features = []

        spatial_features = self.bin_spatial(self.image[y:y + k, x:x + k, :])
        features.append(spatial_features)

        hist_features = self.color_hist(self.image[y:y + k, x:x + k, :])
        features.append(hist_features)

        hog_features = self.hog(x, y, k)
        features.append(hog_features)

        return np.concatenate(features)
