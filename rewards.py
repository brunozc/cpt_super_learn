from utils_SchemaGAN import IC_normalization, reverse_IC_normalization, preprocess_data
import numpy as np
import pandas as pd
from IDW import InverseDistance


def get_reward_rmse_based(interpolation_method: str, idx_current_position: int, idx_known_positions: list,
                          data: pd.DataFrame,  cost_cpt: float = 0, cost_rmse: float = 1,  SchemaGAN=None):
    r"""
    Get the reward based on the RMSE of the known cpt and the predicted cpt at the current position using SchemaGAN
    or the inverse distance interpolation method.

    Parameters:
        interpolation_method (str): interpolation method
        idx_current_position (int): index of current position
        idx_known_positions (list): list of known positions
        data (pd.DataFrame): data
        cost_cpt (float): cost of the cpt
        cost_rmse (float): cost of the RMSE
        SchemaGAN (object): SchemaGAN model

    Returns:
        reward (float): reward

    """

    # position of the known cpt
    total_list_of_positions = idx_known_positions + [idx_current_position]
    print(f"total_list_of_positions: {total_list_of_positions}")
    inputs_missing_field, known_field = preprocess_data(data, total_list_of_positions)
    no_rows = known_field.shape[1]
    no_cols = known_field.shape[0]
    no_samples = 1
    src_images = np.reshape(inputs_missing_field, (no_samples, no_rows, no_cols, 1))
    if interpolation_method == "InverseDistance":
        # IDW interpolation
        idw = InverseDistance(nb_points=6)
        cpt_position = np.array([[idx_current_position, 0]])
        points = np.array([[x, y] for x in range(len(known_field[:, 0])) for y in range(len(known_field[0, :]))])
        # reshape to only 2 dimensions
        points = np.reshape(points, (len(known_field[:, 0]) * len(known_field[0, :]), 2))
        # get all cpt points
        known_points = np.array([point for point in points if point[0] in cpt_position[0]])
        idw.interpolate(known_points, known_field[cpt_position[0], :].flatten())
        idw.predict(points)
        prediction = idw.prediction
        # reshape to the original shape
        prediction = np.reshape(prediction, (len(known_field[:, 0]), len(known_field[0, :])))
    else:
        # normalize the data
        src_norm = IC_normalization(src_images)
        # predict the missing data
        prediction = SchemaGAN.predict(src_norm, verbose=0)
        # reverse the normalization
        prediction = reverse_IC_normalization(np.squeeze(prediction.T))
    # compare at the entire field RMSE
    RMSE = np.sqrt(np.mean((known_field - prediction) ** 2))
    # cost of cpts
    reward = len(idx_known_positions) * cost_cpt
    # cost of the RMSE
    reward += RMSE * cost_rmse
    # cost of staying at the same position
    if idx_current_position == idx_known_positions[-1]:
        reward += -1
    return reward, prediction


def get_reward_accuracy_based(interpolation_method: str, idx_current_position: int, idx_known_positions: list,
                          data: pd.DataFrame,  cost_cpt: float = 0,  SchemaGAN=None):
    r"""
    Get the reward based on the RMSE of the known cpt and the predicted cpt at the current position using SchemaGAN
    or the inverse distance interpolation method.

    Parameters:
        interpolation_method (str): interpolation method
        idx_current_position (int): index of current position
        idx_known_positions (list): list of known positions
        data (pd.DataFrame): data
        cost_cpt (float): cost of the cpt
        SchemaGAN (object): SchemaGAN model

    Returns:
        reward (float): reward

    """

    # position of the known cpt
    total_list_of_positions = idx_known_positions + [idx_current_position]
    print(f"total_list_of_positions: {total_list_of_positions}")
    inputs_missing_field, known_field = preprocess_data(data, total_list_of_positions)
    no_rows = known_field.shape[1]
    no_cols = known_field.shape[0]
    no_samples = 1
    src_images = np.reshape(inputs_missing_field, (no_samples, no_rows, no_cols, 1))
    if interpolation_method == "InverseDistance":
        # IDW interpolation
        idw = InverseDistance(nb_points=6)
        cpt_position = np.array([[idx_current_position, 0]])
        points = np.array([[x, y] for x in range(len(known_field[:, 0])) for y in range(len(known_field[0, :]))])
        # reshape to only 2 dimensions
        points = np.reshape(points, (len(known_field[:, 0]) * len(known_field[0, :]), 2))
        # get all cpt points
        known_points = np.array([point for point in points if point[0] in cpt_position[0]])
        idw.interpolate(known_points, known_field[cpt_position[0], :].flatten())
        idw.predict(points)
        prediction = idw.prediction
        # reshape to the original shape
        prediction = np.reshape(prediction, (len(known_field[:, 0]), len(known_field[0, :])))
    else:
        # normalize the data
        src_norm = IC_normalization(src_images)
        # predict the missing data
        prediction = SchemaGAN.predict(src_norm, verbose=0)
        # reverse the normalization
        prediction = reverse_IC_normalization(np.squeeze(prediction.T))
    # compare at the entire field with accuracy
    # Calculate the residual sum of squares
    ss_res = np.sum((known_field - prediction) ** 2)
    # Calculate the total sum of squares
    ss_tot = np.sum((known_field - np.mean(known_field)) ** 2)
    # Calculate R-squared
    r_squared = 1 - (ss_res / ss_tot)
    # cost of cpts
    reward = len(idx_known_positions) * cost_cpt
    # cost of the RMSE
    reward += r_squared
    # cost of staying at the same position
    if idx_current_position == idx_known_positions[-1]:
        reward += -1
    return reward, prediction


def get_reward_variation_based(interpolation_method: str, idx_current_position: int, idx_known_positions: list,
                          data: pd.DataFrame,  cost_cpt: float = 0,  SchemaGAN=None):
    r"""
    Get the reward based on the RMSE of the known cpt and the predicted cpt at the current position using SchemaGAN
    or the inverse distance interpolation method.

    Parameters:
        interpolation_method (str): interpolation method
        idx_current_position (int): index of current position
        idx_known_positions (list): list of known positions
        data (pd.DataFrame): data
        cost_cpt (float): cost of the cpt
        SchemaGAN (object): SchemaGAN model

    Returns:
        reward (float): reward

    """

    # position of the known cpt
    total_list_of_positions = idx_known_positions + [idx_current_position]
    print(f"total_list_of_positions: {total_list_of_positions}")
    inputs_missing_field, known_field = preprocess_data(data, total_list_of_positions)
    no_rows = known_field.shape[1]
    no_cols = known_field.shape[0]
    no_samples = 1
    src_images = np.reshape(inputs_missing_field, (no_samples, no_rows, no_cols, 1))
    if interpolation_method == "InverseDistance":
        # IDW interpolation
        idw = InverseDistance(nb_points=6)
        cpt_position = np.array([[idx_current_position, 0]])
        points = np.array([[x, y] for x in range(len(known_field[:, 0])) for y in range(len(known_field[0, :]))])
        # reshape to only 2 dimensions
        points = np.reshape(points, (len(known_field[:, 0]) * len(known_field[0, :]), 2))
        # get all cpt points
        known_points = np.array([point for point in points if point[0] in cpt_position[0]])
        idw.interpolate(known_points, known_field[cpt_position[0], :].flatten())
        idw.predict(points)
        prediction = idw.prediction
        # reshape to the original shape
        prediction = np.reshape(prediction, (len(known_field[:, 0]), len(known_field[0, :])))
    else:
        # normalize the data
        src_norm = IC_normalization(src_images)
        # predict the missing data
        prediction = SchemaGAN.predict(src_norm, verbose=0)
        # reverse the normalization
        prediction = reverse_IC_normalization(np.squeeze(prediction.T))
    # compare at the entire field with accuracy
    # Calculate the residual sum of squares
    ss_res = np.sum((known_field - prediction) ** 2)
    # Calculate the total sum of squares
    ss_tot = np.sum((known_field - np.mean(known_field)) ** 2)
    # accuracy of the standard deviation
    ss_res_std = np.sum((np.std(known_field) - np.std(prediction)) ** 2)
    ss_tot_std = np.sum((np.std(known_field) - np.mean(np.std(known_field))) ** 2)
    accuracy_std = 1 - (ss_res_std / ss_tot_std)
    # Calculate R-squared
    r_squared = 1 - (ss_res / ss_tot)
    variation = (r_squared + accuracy_std) / 2
    # cost of cpts
    reward = len(idx_known_positions) * cost_cpt
    # cost of the RMSE
    reward += variation
    # cost of staying at the same position
    if idx_current_position == idx_known_positions[-1]:
        reward += -1
    return reward, prediction