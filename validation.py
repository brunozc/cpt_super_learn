import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
# NN
from neural_RL import SimpleNeuralNetwork
from states import get_state_divitions, get_state, get_state2
from rf_pytorch_SchemaGAN import get_next_position, get_reward, get_starting_position_index, preprocess_data, load_SchemaGAN_model, get_state_full_arrays

import matplotlib
matplotlib.use('Qt5Agg')


def read_input_data_file(training_data_folder: str) -> pd.DataFrame:

    # Read the .csv file into a pandas dataframe
    df = pd.read_csv(training_data_folder, delimiter=',')
    return df

data_dir = "P:/schemagan/synthetic_database/512x32_20k"
output_dir = "./results_schemaGAN_relu"
plots_dir = ("./validation_schemaGAN_4_cpts")

index = 900

# load the model
q_network = SimpleNeuralNetwork.load_model(os.path.join(output_dir, f"deep_{index}.pth"))

actions = [25, 57, 150]
max_nb_points = 512
max_starting_position = 30
schemaGAN_model_path = "D:/model_000036.h5"
SchemaGAN = load_SchemaGAN_model(schemaGAN_model_path)

files = os.listdir(os.path.join(data_dir, "test"))
states_all = []
q_all = []
for i, f in enumerate(files[10:30]):
    data = read_input_data_file(os.path.join(data_dir, "test", f))
    _, input_data_pre_processed = preprocess_data(data, [2])
    # transpose the data
    input_data_pre_processed = input_data_pre_processed.T

    done = False
    idx_position = get_starting_position_index(max_starting_position)
    idx_known_positions = [idx_position]
    # state = get_state(idx_position, data)
    state = get_state_full_arrays(idx_position, idx_known_positions, data, SchemaGAN)

    q_values_array = [0]
    last_index = []
    while not done:
        q_values = q_network(torch.Tensor(state))
        q_values_array_new = [float(i) for i in q_values]
        action_index = np.argmax(q_values_array_new)
        q_values_array = q_values_array_new
        idx_position = get_next_position(idx_position, actions[action_index], max_nb_points)
        last_index.append(idx_position)
        if (0 >= idx_position >= max_nb_points) or (last_index.count(last_index[-1]) > 5):
            done = True
        else:
            idx_known_positions.append(idx_position)
            state_new = get_state_full_arrays(idx_position, idx_known_positions, data, SchemaGAN)


            state = state_new
            states_all.append(state)
    print(f"{f}: {idx_known_positions} diff {np.diff(np.array(idx_known_positions))}")
# plot states for each file
fig, ax = plt.subplots(1, 1)
states_all = np.array(states_all).flatten()
ax.hist(states_all, bins=512)
plt.show()
# plot
get_reward(idx_position, idx_known_positions, data, f.split(".")[0],
               plots_dir, f, plots=True, SchemaGAN=SchemaGAN)
