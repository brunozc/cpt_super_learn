import os
from typing import List, Tuple
import numpy as np


def read_data_file(file_name: str) -> np.ndarray:
    """
    Reads a data file

    Parameters:
    -----------
    :param file_name: file name
    :return: data
    """

    # read data
    with open(file_name, "r") as fi:
        data = fi.read().splitlines()
        data = [np.array(i.split(",")[1:]).astype(float) for i in data[1:]]

    # sort data
    data = sorted(data, key=lambda x: (x[0], x[1]))

    return os.path.splitext(os.path.basename(file_name))[0], np.array(data)


def input_random_data_file(training_data_folder: str) -> Tuple[str, np.ndarray]:
    """
    Reads an random input data file for the data folder
    It sorts the files so that the order is consistent

    Parameters:
    -----------
    :param training_data_folder: folder with the training data
    :return: file name and data

    Returns:
    --------
    :return: file name and data
    """

    # randomly initialise the field
    files = os.listdir(training_data_folder)
    files.sort()

    idx = np.random.randint(len(files))

    file_id, data = read_data_file(os.path.join(training_data_folder, files[idx]))

    return file_id, data


def write_score(episode: List, total_score: List, output_file: str):
    """
    Writes the score to a file

    Parameters:
    -----------
    :param total_score: list with the score
    :param output_file: output file
    """

    # check if folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as fo:
        for ep, score in zip(episode, total_score):
            fo.write(f"{ep};{score}\n")


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculate the moving average

    Parameters:
    -----------
    :param data: data
    :param window_size: window size

    Returns:
    --------
    :return: moving average
    """

    if data.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if data.size < window_size:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    s=np.r_[2*data[0] - data[window_size:1:-1], data, 2 * data[-1]-data[-1:-window_size:-1]]
    w = np.ones(window_size, 'd')
    y = np.convolve(w/w.sum(), s, mode='same')

    return y[window_size-1:-window_size+1]