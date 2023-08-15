import numpy as np
from scipy.spatial import cKDTree


class InverseDistance:
    """
    Inverse distance interpolator
    """
    def __init__(self, nb_points=20):
        self.nb_near_points = nb_points
        self.power = 1.0
        self.tol = 1e-9
        self.var = None
        self.prediction = []
        self.tree = []
        self.training_points = []
        self.training_data = []

    def interpolate(self, training_points, training_data):
        """
        Define the KDtree

        :param training_points: array with the training points
        :param training_data: data at the training points
        :return:
        """

        # assign to variables
        self.training_points = np.array(training_points)  # training points
        self.training_data = np.array(training_data)  # data at the training points

        # compute Euclidean distance from grid to training
        self.tree = cKDTree(self.training_points.reshape(-1, 1))

    def predict(self, prediction_points):
        """
        Perform interpolation with inverse distance method

        :param prediction_points: prediction points
        :return:
        """

        if len(self.training_points) <= self.nb_near_points:
            self.nb_near_points = len(self.training_points)
        # get distances and indexes of the closest nb_points
        dist, idx = self.tree.query(prediction_points.reshape(-1, 1), self.nb_near_points)
        dist += self.tol  # to overcome division by zero

        # Compute weights
        weights = 1.0 / (dist ** self.power)

        # Normalize weights
        normalized_weights = weights / np.sum(weights, axis=1, keepdims=True)

        # Perform weighted average using broadcasting
        zn = np.sum(self.training_data[idx] * normalized_weights[:, :, np.newaxis], axis=1)

        self.prediction = zn
