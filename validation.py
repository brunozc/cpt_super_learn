import os
import numpy as np
import torch

# NN
from neural import SimpleNeuralNetwork
from rf_pytorch import get_next_position, get_state, get_state2, get_reward, get_starting_position_index

def read_data_file(file: str) -> np.ndarray:

    # read data
    with open(file, "r") as fi:
        data = fi.read().splitlines()
        data = [np.array(i.split(";")).astype(float) for i in data[1:]]

    # sort data
    data = sorted(data, key=lambda x: (x[0], x[1]))
    
    return np.array(data)

data_dir = "./data"
output_dir = "./output"
plots_dir = "./validation"

index = 100

# load the model
q_network = SimpleNeuralNetwork.load_model(os.path.join(output_dir, f"deep_{index}.pth"))


actions = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
max_nb_points = 51
max_starting_position = 5

files = os.listdir(os.path.join(data_dir, "validation"))
for i, f in enumerate(files):
    data = read_data_file(os.path.join(data_dir, "validation", f))

    done = False
    idx_position = get_starting_position_index(max_starting_position)
    idx_known_positions = [idx_position]
    # state = get_state(idx_position, data)
    state = get_state2(idx_position, idx_known_positions, data)


    last_index = []
    while not done:
        with torch.no_grad():
            q_values = q_network.forward(torch.Tensor(state))
        action_index = np.argmax(q_values.detach().numpy())
        idx_position = get_next_position(idx_position, actions[action_index], max_nb_points)
        last_index.append(idx_position)
        if (0 >= idx_position >= max_nb_points) or (last_index.count(last_index[-1]) > 5):
            done = True
        else:
            idx_known_positions.append(idx_position)
            state = get_state2(idx_position, idx_known_positions, data)
    print(f"{f}: {idx_known_positions} diff {np.diff(np.array(idx_known_positions))}")
    # plot
    get_reward(idx_position, idx_known_positions, data, f.split(".")[0],
               plots_dir, f, plots=True)
