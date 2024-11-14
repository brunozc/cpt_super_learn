from typing import List
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial import cKDTree


class InterpolatorAbc(ABC):
    """
    Abstract class for interpolators
    """

    @abstractmethod
    def interpolate(self):
        """
        Interpolate the data
        """
        raise NotImplementedError("Interpolate method must be implemented")

    @abstractmethod
    def predict(self, prediction_points: List[float]):
        """
        Predict the data at the prediction points
        """
        raise NotImplementedError("Predict method must be implemented")


class InverseDistance(InterpolatorAbc):
    """
    Inverse distance interpolator
    """
    def __init__(self, nb_points: int):
        """
        Initialize the interpolator
        Parameters:
        -----------
        :param nb_points: number of points to consider
        """

        if nb_points < 1:
            raise ValueError("nb_points must be greater than 1")

        self.nb_near_points = nb_points
        self.power = 1.0
        self.tol = 1e-9
        self.var = None
        self.prediction = []
        self.tree = []
        self.training_points = []
        self.training_data = []

    def interpolate(self, training_points: List[float], training_data: List[float]):
        """
        Define the KDtree

        Parameters:
        -----------
        :param training_points: array with the training points
        :param training_data: data at the training points
        :return:
        """

        # assign to variables
        self.training_points = np.array(training_points)  # training points
        self.training_data = np.array(training_data)  # data at the training points

        # compute Euclidean distance from grid to training
        self.tree = cKDTree(self.training_points.reshape(-1, 1))

    def predict(self, prediction_points: List[float]):
        """
        Perform interpolation with inverse distance method

        Parameters:
        -----------
        :param prediction_points: prediction points
        :return: prediction
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
