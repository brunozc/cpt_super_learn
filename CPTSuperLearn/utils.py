import os
from typing import List, Tuple
import requests
import zipfile
import numpy as np
import yaml
from schema import Schema, And, Or, SchemaError
from tqdm import tqdm


MAX_IC_VALUE = 4.5  # Maximum expected IC value
MIN_IC_VALUE = 0  # Minimum expected IC value, it's not really zero


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


def download_file(url: str, file_name: str):
    """
    Download a file from a URL and save it to the specified location.

    Parameters:
    -----------
    url (str): URL to download the file from
    file_name (str): Name of the file to save
    """

    response = requests.get(url, stream=True)
    response.raise_for_status()

    if os.path.dirname(file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

    total_size = int(response.headers.get('content-length', 0))
    pbar = tqdm(total=total_size, desc=f"Downloading {file_name}", unit="B", unit_scale=True, unit_divisor=1024)

    # Write the content to a file
    with open(file_name, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            pbar.update(8192)

    pbar.close()
    print(f"Downloaded {file_name} successfully")


def extract_zip(data_zip, extract_path):
    """
    Extract a ZIP file to the specified location with progress bar.

    Parameters:
    -----------
    data_zip (str): Path to the ZIP file
    extract_path (str): Path to extract the files to
    """
    os.makedirs(extract_path, exist_ok=True)

    # Open the ZIP file once
    with zipfile.ZipFile(data_zip, "r") as zip_ref:
        # Get list of files to extract
        file_list = zip_ref.namelist()

        pbar = tqdm(total=len(file_list), desc=f"Extracting {data_zip}", unit="file")
        # Create progress bar
        # with tqdm(total=len(file_list), desc=f"Extracting {data_zip}", unit="file") as pbar:
        for file in file_list:
            zip_ref.extract(file, extract_path)
            pbar.update(1)

    print(f"Files extracted to {extract_path}")

    # Clean up
    os.remove(data_zip)

def load_yaml_config(config_path):
    """
    Load configuration from YAML file

    Parameters
    ----------
    :param config_path: Path to the configuration file
    :return: Configuration dictionary
    """

    if not validate_yaml(config_path):
        raise ValueError(f"Invalid YAML file: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_yaml(config_path):
    """
    Validate the YAML configuration file using schema validation.
    Parameters
    ----------
    :param config_path: Path to the configuration file
    :return: True if valid, False otherwise
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    schema = Schema({
        'seed': And(int, lambda n: n >= 0),
        'data': {
            'train_folder': And(str, len),
            'validation_folder': And(str, len)
        },
        'environment': {
            'actions': [int],
            'max_nb_cpts': And(int, lambda n: n > 0),
            'weight_reward_cpt': And(float, lambda n: 0 <= n <= 1),
            'image_width': And(int, lambda n: n > 0),
            'max_first_step': And(int, lambda n: n >= 0),
            'interpolation_method': Or("SchemaGAN", "InverseDistance"),
            'interpolation_method_params': Or(str, int)  # Allow str path or int
        },
        'agent': {
            'state_size': And(int, lambda n: n > 0),
            'learning_rate': And(float, lambda n: 0 < n < 1),
            'gamma': And(float, lambda n: 0 <= n <= 1),
            'epsilon_start': And(float, lambda n: 0 <= n <= 1),
            'epsilon_end': And(float, lambda n: 0 <= n <= 1),
            'epsilon_decay': And(float, lambda n: 0 <= n <= 1),
            'memory_size': And(int, lambda n: n > 0),
            'batch_size': And(int, lambda n: n > 0),
            'nb_steps_update': And(int, lambda n: n > 0),
            'hidden_layers': [int],
            'use_batch_norm': bool,
            'activation': Or("relu", "tanh", "sigmoid")
        },
        'training': {
            'nb_episodes': And(int, lambda n: n > 0),
            'make_plots': bool,
            'output_folder': And(str, len),
            'log_interval': And(int, lambda n: n >= 0)
        },
        'validation': {
            'make_plots': bool,
            'output_folder': And(str, len)
        }
    })

    try:
        schema.validate(config)
        return True
    except SchemaError as e:
        return False
