from typing import List
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial import cKDTree
from keras.models import load_model
from CPTSuperLearn.utils import IC_normalization, reverse_IC_normalization


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


class SchemaGANInterpolator(InterpolatorAbc):
    """
    Interpolator using SchemaGAN model for interpolation.
    For more information on SchemaGAN, see https://github.com/fabcamo/schemaGAN.

    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = load_model(model_path)
        self.size_x = 512
        self.size_y = 32
        self.training_points = []
        self.training_data = []
        self.prediction = []

    def interpolate(self, training_points: np.ndarray, training_data: np.ndarray):
        # check the size of the data
        if training_data.shape[1] != self.size_y:
            raise ValueError(f"Data must have shape (:, {self.size_y})")
        # create a zeros array
        self.training_data = np.zeros((self.size_x, self.size_y))
        # fill the array with the training data
        for counter, column in enumerate(training_points):
            self.training_data[column, :] = training_data[counter, :]
        self.training_points = training_points

    def predict(self, prediction_points: np.ndarray):
        # check the size of the data
        if prediction_points.shape[0] != self.size_x:
            raise ValueError(f"Data must have shape ({self.size_x}, :)")
        # reshape the data
        normalize_training_data = np.reshape(self.training_data.T, (1, self.size_y, self.size_x, 1))
        normalize_training_data = IC_normalization(normalize_training_data)
        prediction = self.model.predict(normalize_training_data, verbose=0)
        prediction = reverse_IC_normalization(np.squeeze(prediction.T))
        self.prediction = prediction


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
