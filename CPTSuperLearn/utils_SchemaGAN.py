import numpy as np
import pandas as pd

MAX_IC_VALUE = 4.5  # Maximum expected IC value
MIN_IC_VALUE = 0  # Minimum expected IC value, it's not really zero,


def IC_normalization(data: np.ndarray):
    """
    Normalize IC values in the data from [0 - MaxIC] to [-1 - 1].

    Parameters:
    data (list): List containing the source and target data.

    Returns:
    list: A list containing the normalized source and target data.
    """

    # Define the maximum and minimum values of IC in the source and target images

    # but when deleting data it will be zero

    # Calculate the range of the data
    data_range = MAX_IC_VALUE - MIN_IC_VALUE

    # Scale the source and target data to the range [-1, 1]
    # Formula used for normalization is:
    # normalized_data = 2 * (original_data / data_range) - 1
    src_normalized = 2 * (data / data_range) - 1

    return src_normalized




def reverse_IC_normalization(data: np.ndarray):
    """
    Reverse the normalization of IC values in the data from [-1 - 1] to [0 - MaxIC].

    Parameters:
    data (np.array): Array containing the normalized data.

    Returns:
    np.array: An array containing the rescaled data.
    """


    # Calculate the range of the data
    data_range = MAX_IC_VALUE - MIN_IC_VALUE

    # Rescale the data to the original range [min_IC_value, max_IC_value]
    # Formula used for rescaling is:
    # rescaled_data = (normalized_data + 1) * (data_range / 2) + min_IC_value
    X = (data + 1) * (data_range / 2) + MIN_IC_VALUE

    return X
