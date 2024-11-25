import numpy as np
import pandas as pd


def IC_normalization(data):
    """
    Normalize IC values in the data from [0 - MaxIC] to [-1 - 1].

    Parameters:
    data (list): List containing the source and target data.

    Returns:
    list: A list containing the normalized source and target data.
    """

    # Define the maximum and minimum values of IC in the source and target images
    max_IC_value = 4.5  # Maximum expected IC value
    min_IC_value = 0  # Minimum expected IC value, it's not really zero,
    # but when deleting data it will be zero

    # Calculate the range of the data
    data_range = max_IC_value - min_IC_value

    # Scale the source and target data to the range [-1, 1]
    # Formula used for normalization is:
    # normalized_data = 2 * (original_data / data_range) - 1
    src_normalized = 2 * (data / data_range) - 1

    return src_normalized




def reverse_IC_normalization(data):
    """
    Reverse the normalization of IC values in the data from [-1 - 1] to [0 - MaxIC].

    Parameters:
    data (np.array): Array containing the normalized data.

    Returns:
    np.array: An array containing the rescaled data.
    """

    # Define the maximum and minimum values of IC in the source and target images
    max_IC_value = 4.5  # Maximum expected IC value
    min_IC_value = 0  # Minimum expected IC value, it's not really zero,
    # but when deleting data it will be zero

    # Calculate the range of the data
    data_range = max_IC_value - min_IC_value

    # Rescale the data to the original range [min_IC_value, max_IC_value]
    # Formula used for rescaling is:
    # rescaled_data = (normalized_data + 1) * (data_range / 2) + min_IC_value
    X = (data + 1) * (data_range / 2) + min_IC_value

    return X


def preprocess_data(data: pd.DataFrame, columns_to_keep_index: list) -> np.ndarray:
    missing_data, full_data = [], []
    # Define the column name of interest
    value_name = 'IC'

    # Initialize a list to store data grouped by z values
    data_z = []
    # Group the dataframe by the 'z' column
    grouped = data.groupby("z")
    # Iterate over the groups and extract the 'IC' column data
    for name, group in grouped:
        data_z.append(list(group[value_name]))
    # Convert the list to a numpy array of floats
    data_z = np.array(data_z, dtype=float)
    data_z = np.transpose(data_z)
    # Apply missing data to the field
    data_m = np.zeros_like(data_z)
    # Set the values in data_m to 1 for the columns that are to be kept
    for column_index in columns_to_keep_index:
        data_m[column_index, :] = np.ones_like(data_m[column_index, :])
    miss_list = np.multiply(data_z, data_m)
    miss_list = np.transpose(miss_list)

    # Return the lists of missing and full data arrays
    return miss_list, data_z