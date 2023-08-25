import os
import shutil
import json
from collections import deque, namedtuple
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
import torch
# import torch.nn.functional as F

# interpolation
from IDW import InverseDistance
# NN
from neural_RL import SimpleNeuralNetwork


np.random.seed(14)
torch.manual_seed(14)

def make_plot(episode, state, cpt, depth, cpt_position, idw, new_data, unique_x, unique_y, output_folder, file_name):
    r"""
    Plot the results of the episode

    # to understand difference between imshow and scatter see:
    https://stackoverflow.com/questions/75235652/why-does-plt-imshow-flip-coordinates-compared-to-plt-scatter
    """
    # plot
    if not os.path.isdir(os.path.join(output_folder, f"episode_{episode}")):
        os.makedirs(os.path.join(output_folder, f"episode_{episode}"))

    vmin = 1
    vmax = 4
    fig, ax = plt.subplots(3, 1, figsize=(10, 5))
    ax[0].set_position([0.075, 0.70, 0.775, 0.175])
    ax[1].set_position([0.075, 0.40, 0.775, 0.25])
    ax[2].set_position([0.075, 0.10, 0.775, 0.25])
    for i, x in enumerate(cpt_position):
        ax[0].scatter(np.ones(len(depth[i])) * x, depth[i], c=cpt[i],
                      vmin=vmin, vmax=vmax, marker="s", s=3, cmap="viridis")

    x, y = np.meshgrid(unique_x, unique_y)
    ax[1].imshow(idw.prediction.T, vmin=vmin, vmax=vmax, cmap="viridis",
                    extent=[0, np.max(x), np.max(y), 0])#, aspect="auto")
    ax[1].invert_yaxis()

    ax[2].imshow(new_data.T, vmin=vmin, vmax=vmax, cmap="viridis",
                    extent=[0, np.max(x), np.max(y), 0])#, aspect="auto")
    ax[2].invert_yaxis()

    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[0].xaxis.set_ticklabels([])
    ax[1].xaxis.set_ticklabels([])
    ax[2].set_xlabel("Distance [m]")
    ax[0].set_ylabel("Known")
    ax[1].set_ylabel("Interpolation")
    ax[2].set_ylabel("True")
    ax[0].set_xlim([0, np.max(unique_x)])
    ax[1].set_xlim([0, np.max(unique_x)])
    ax[2].set_xlim([0, np.max(unique_x)])

    # add RMSE
    # ax[0].text(0, 14, f'Episode {episode:.0f}, position {state:.0f}, RMSE={RMSE:.3f}',
    #               horizontalalignment='left', verticalalignment='center')
    # Add a colorbar to a plot
    cax = ax[0].inset_axes([1.05, -2., 0.05, 2.5])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, cax=cax)
    cbar.set_label("IC")
    plt.savefig(os.path.join(os.path.join(output_folder, f"episode_{episode}", f"state_{state}_{file_name}.png")), dpi=600)
    plt.close()


def get_reward(idx_current_position: int, idx_known_positions: list, data: np.ndarray,
               episode: int, output_folder: str, file_name: str,
               cost_cpt: float = -10, cost_rmse: float = 5, plots: bool = True):
    r"""
    Custom reward function
    The interpolation is currently Inverse Distance Weighting (IDW)

    Parameters
    ----------
    idx_current_position (int): index of the current position of the agent
    idx_known_positions (list): list of indexes of known positions
    data (np.ndarray): path to the data
    episode (int): current episode
    output_folder (str): path to the output folder
    file_name (str): name of the file image
    cost_cpt (float): cost of a cpt
    cost_rmse (float): cost of a rmse
    plots (bool): if True, plots are generated
    """

    # unique coordinates CPT
    unique_x = np.unique(data[:, 0])
    unique_x.sort()
    unique_y = np.unique(data[:, 1])
    unique_y.sort()

    # read cpts at the known positions
    cpt = []
    depth = []
    cpt_position = []
    for k in idx_known_positions:
        idx = np.where(data[:, 0] == unique_x[k])[0]
        depth.append(data[idx, 1])
        cpt.append(data[idx, 2])
        cpt_position.append(unique_x[k])

    # read cpt at current position
    idx = np.where(data[:, 0] == unique_x[idx_current_position])[0]
    depth.append(data[idx, 1])
    cpt.append(data[idx, 2])
    cpt_position.append(unique_x[idx_current_position])

    # get unique cpt positions
    cpt_position, idx = np.unique(cpt_position, return_index=True)
    # interpolate at the entire field
    if len(cpt_position) < 2:
        return -10, 10

    # perform interpolation
    idw = InverseDistance(nb_points=6)
    idw.interpolate(cpt_position, np.array(cpt)[idx])
    idw.predict(unique_x)

    # reshape data
    new_data = data[:, 2].reshape(len(unique_x), len(unique_y))

    # compare at the entire field RMSE
    RMSE = np.sqrt(np.mean((new_data - idw.prediction) ** 2))

    # cost of cpts
    reward = len(idx_known_positions) * cost_cpt

    # cost of the RMSE
    reward += -1 * RMSE * cost_rmse

    # make plot
    if plots:
        make_plot(episode, unique_x[idx_current_position], cpt, depth, cpt_position, idw, new_data,
                  unique_x, unique_y, output_folder, file_name)
    return reward, RMSE



def get_starting_position_index(maximum_step: int) -> int:
    r"""
    Get the index of the starting position

    Parameters:
    -----------
    maximum_step (int): maximum step in pixels of the movement
    """
    return np.random.randint(maximum_step)


def get_next_action(q_values: np.ndarray, actions: list, epsilon: float) -> int:
    r"""
    Get the next action

    Parameters:
    -----------
    q_values (np.ndarray): q_values
    actions (np.ndarray): list of actions
    epsilon (float): probability of taking a random action
    """

    if np.random.random() > epsilon:
        return np.argmax(q_values)
    else:
        return np.random.randint(0, len(actions))


def get_next_position(current_position_idx: int, action: int, nb_points: int) -> int:
    r"""
    Get the next position index

    Parameters:
    -----------
    current_position (int): current position
    action (int): number of steps to take
    nb_points (int): maximum number of points
    """
    if (current_position_idx + action >= nb_points) or (current_position_idx + action < 0):
        return current_position_idx
    else:
        return current_position_idx + action


def read_input_data_file(training_data_folder: str) -> np.ndarray:

    # randomly initialise the field
    files = os.listdir(training_data_folder)
    files.sort()

    idx = np.random.randint(len(files))

    # read data
    with open(os.path.join(training_data_folder, files[idx]), "r") as fi:
        data = fi.read().splitlines()
        data = [np.array(i.split(";")).astype(float) for i in data[1:]]

    # sort data
    data = sorted(data, key=lambda x: (x[0], x[1]))

    return files[idx].split(".txt")[0], np.array(data)


def get_state(idx_current_position: int, data: np.ndarray) -> list:
    r"""
    Get the state: position, mean, std

    Parameters:
    -----------
    idx_current_position (int): index of current position
    data (np.ndarray): data
    """

    # unique coordinates CPT
    unique_x = np.unique(data[:, 0])
    unique_x.sort()
    unique_y = np.unique(data[:, 1])
    unique_y.sort()

    # idx of CPT at current position
    idx = np.where(data[:, 0] == unique_x[idx_current_position])[0]
    mean = np.mean(data[idx, 2])
    std = np.std(data[idx, 2])

    return [idx_current_position, mean, std]


def classify_cpt(ic_values: np.ndarray) -> int:
    r"""
    Classify the cpt values

    Parameters:
    -----------
    ic_values (np.ndarray): ic values

    Returns:
    --------
    label (int): number of soil layers in the cpt
    """

    label = np.zeros(len(ic_values))
    label[ic_values > 3.6] = 2
    label[(ic_values <= 3.6) & (ic_values > 2.95)] = 3
    label[(ic_values <= 2.95) & (ic_values > 2.6)] = 4
    label[(ic_values <= 2.6) & (ic_values > 2.05)] = 5
    label[(ic_values <= 2.05) & (ic_values > 1.31)] = 6
    label[ic_values <= 1.31] = 7

    return len(set(label))

def get_state2(idx_current_position: int, idx_known_position: list[int], data: np.ndarray) -> list:
    r"""
    Get the state: position, mean, std

    Parameters:
    -----------
    idx_current_position (int): index of current position
    data (np.ndarray): data
    """

    # unique coordinates CPT
    unique_x = np.unique(data[:, 0])
    unique_x.sort()
    unique_y = np.unique(data[:, 1])
    unique_y.sort()

    # idx of CPT at current position
    idx = np.where(data[:, 0] == unique_x[idx_current_position])[0]
    mean = np.mean(data[idx, 2])
    std = np.std(data[idx, 2])
    nb_layers = classify_cpt(data[idx, 2])


    # idx of CPT at known positions
    mean_k = []
    std_k = []
    mean_nb_layers = []
    for i in unique_x[idx_known_position]:
        idx = np.where(data[:, 0] == i)[0]
        mean_k.append(np.mean(data[idx, 2]))
        std_k.append(np.std(data[idx, 2]))
        mean_nb_layers.append(classify_cpt(data[idx, 2]))

    return [idx_current_position, mean, std, np.mean(mean_k), np.mean(std_k), nb_layers, np.mean(mean_nb_layers), np.std(mean_nb_layers)]


def check_update(num_steps, steps_update, length_data, minimum_batch_size):
    if (num_steps + 1) % steps_update == 0 and length_data >= minimum_batch_size:
        return True
    else:
        return False


def update_q_network(experiences: tuple, gamma: float,
                     q_network: torch.nn.Module, target_network: torch.nn.Module, tau=1e-3):
    """
    Updates the weights of the Q network

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"]
      gamma: (float) The discount factor.
      q_network: model for predicting the q_values
      tau: (float) The soft update factor
    target_network: model for predicting the targets
    """

    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences

    with torch.no_grad():
        # Compute max Q(s', a')
        max_qsa, _ = torch.max(target_network.forward(next_states), axis=1)

    # compute y values following Bellman Equation
    y_targets = rewards + (gamma * max_qsa.detach().numpy() * (1 - done_vals))

    # get q_values
    q_values = q_network.forward(states)
    q_values = q_values[torch.arange(q_values.shape[0]), actions]

    # # compute the loss
    q_network.run(q_values.view(-1, 1), torch.Tensor(y_targets).view(-1, 1))

    # update weights of target q_network : soft update
    for target_param, local_param in zip(target_network.parameters(), q_network.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    return q_network, target_network


def get_new_eps(epsilon):
    E_MIN = 0.01
    E_DECAY = 0.999
    return max(E_MIN, E_DECAY * epsilon)


def get_experiences(experiences_replay, minibatch_size):

    idx = np.random.choice(len(experiences_replay), size=minibatch_size, replace=False)

    states = [experiences_replay[i][0] for i in idx]
    actions = [experiences_replay[i][1] for i in idx]
    rewards = [experiences_replay[i][2] for i in idx]
    next_states = [experiences_replay[i][3] for i in idx]
    done = [experiences_replay[i][4] for i in idx]

    return (torch.Tensor(np.array(states)), np.array(actions), np.array(rewards), torch.Tensor(np.array(next_states)), np.array(done))


def dump_data(score, rmse, output_folder, file_name):
    summary = {"score": score, "rmse": rmse}
    # save the scores
    with open(os.path.join(output_folder, file_name), "w") as f:
        json.dump(summary, f, indent=2)


def main(training_data_folder, output_folder, plots=True):
    actions = [2, 4, 6, 8, 10, 12, 14, 15, 16]  # actions in number of pixels
    state = [1, 5.5, 3.2, 5.5, 3.2, 7, 5, 1]  # location, mean, std, mean mean, std mean, nb layers, mean nb layers, std nb layers
    nb_episodes = 101
    epsilon = 1.0
    gamma = 0.99
    max_nb_pixels = 51
    max_starting_index = 5
    # settings
    num_steps_update = 24
    batch_size = 64
    memory_size = 500

    # delete output folder if exists
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)

    experiences_replay = deque(maxlen=memory_size)
    iteration = 1

    scores = []
    rmses = []

    experience_tupple = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])

    # create Q-network
    DeepQNetwork = SimpleNeuralNetwork(len(state), len(actions), [10, 10, 10])
    # get weights of Q-network
    weights = [param.data for param in DeepQNetwork.parameters()]
    # create Target Q-network
    TargetQNetwork = SimpleNeuralNetwork(len(state), len(actions), [10, 10, 10])
    # set weights of Target Q-network
    for i, param in enumerate(TargetQNetwork.parameters()):
        param.data = weights[i]

    #run through training episodes
    for episode in range(nb_episodes):
        print(f"episode: {episode}")

        # get data
        file_name, input_data = read_input_data_file(training_data_folder)

        # get the starting location for this episode
        position_idx = get_starting_position_index(max_starting_index)
        known_positions_idx = [position_idx]
        # state = get_state(position, input_data)
        # alternatively
        state = get_state2(position_idx, known_positions_idx, input_data) #State = [location, mean, std, mean_known, std_known] (eventually nb points known)

        update = False
        score = 0
        rmse = 0
        terminal = False

        last_index = []
        while not terminal:
            # From the current state S choose an action A using an ε-greedy policy
            with torch.no_grad():
                q_values = DeepQNetwork.forward(torch.Tensor(state))

            #choose which action to take (i.e., where to move next)
            action_index = get_next_action(q_values.detach().numpy(), actions, epsilon)

            # perform the chosen action, and transition to the next state (i.e., move to the next location)
            position_idx = get_next_position(position_idx, actions[action_index], max_nb_pixels)
            # compute reward
            reward, rmse = get_reward(position_idx, known_positions_idx, input_data, episode,
                                      output_folder, file_name, plots=plots)
            score += reward
            last_index.append(position_idx)

            if (0 >= position_idx >= max_nb_pixels) or (last_index.count(last_index[-1]) > 5):
                terminal = True
            else:
                terminal = False

            # add to experience replay
            experiences_replay.append(experience_tupple(state, action_index, reward,
                                                        get_state2(position_idx, known_positions_idx, input_data), terminal))
            # check if it is time to update the Q-network
            update = check_update(iteration, num_steps_update, len(experiences_replay), batch_size)

            if update:
                print("### Performing update ###")
                # get experiences from experiences replay
                experiences = get_experiences(experiences_replay, batch_size)

                # update the Q-network
                DeepQNetwork, TargetQNetwork = update_q_network(experiences, gamma, DeepQNetwork, TargetQNetwork)

            # update the state
            state = get_state2(position_idx, known_positions_idx, input_data)
            # update known positions
            known_positions_idx.append(position_idx)
            # update counter
            iteration += 1

        # update epsilon
        epsilon = get_new_eps(epsilon)
        scores.append(score)
        rmses.append(rmse)

        if episode % 100 == 0:
            print(f"episode: {episode}, score: {score:.3f}, rmse: {rmse:.3f}")
            # save the model
            DeepQNetwork.save_model(output_folder, f"deep_{episode}.pth")
            TargetQNetwork.save_model(output_folder, f"target_{episode}.pth")
            # dump data
            dump_data(scores, rmses, output_folder, f"summary_{episode}.json")


if __name__ == "__main__":
    main("./data/train", "./output")
