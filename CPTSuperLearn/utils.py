import os
from typing import List
import numpy as np


def input_random_data_file(training_data_folder: str) -> np.ndarray:
    """
    Reads an random input data file for the data folder
    It sorts the files so that the order is consistent

    Parameters:
    -----------
    :param training_data_folder: folder with the training data
    :return: file name and data
    """

    # randomly initialise the field
    files = os.listdir(training_data_folder)
    files.sort()

    idx = np.random.randint(len(files))

    # read data
    with open(os.path.join(training_data_folder, files[idx]), "r") as fi:
        data = fi.read().splitlines()
        # data = [np.array(i.split(";")).astype(float) for i in data[1:]]
        data = [np.array(i.split(",")[1:]).astype(float) for i in data[1:]]

    # sort data
    data = sorted(data, key=lambda x: (x[0], x[1]))

    return files[idx].split(".txt")[0], np.array(data)

def write_rmse(episode: List, total_score: List, output_file: str):
    """
    Writes the RMSE to a file

    Parameters:
    -----------
    :param total_score: list with the RMSE
    :param output_file: output file
    """

    # check if folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as fo:
        fo.write("Episode;RMSE\n")
        for ep, score in zip(episode, total_score):
            fo.write(f"{ep};{score}\n")