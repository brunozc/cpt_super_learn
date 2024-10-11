import os
import shutil
import json
from collections import deque, namedtuple
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
import torch
import math
from keras.models import load_model
from utils_SchemaGAN import IC_normalization, reverse_IC_normalization, preprocess_data
import torch.nn.functional as F
from logger import Logger
import matplotlib
#matplotlib.use('Qt5Agg')
import pandas as pd
import pickle
# NN
from neural_RL import SimpleNeuralNetwork
from states import get_state2, get_state, get_state_no_position, get_state_divitions, get_state_full_arrays, get_states_size
from rewards import get_reward_rmse_based, get_reward_accuracy_based, get_reward_variation_based


np.random.seed(14)
torch.manual_seed(14)

def make_plot(episode, state, cpt, depth, cpt_position, prediction, new_data, unique_x, unique_y, output_folder, file_name, nb_episodes, reward_global, rmse_global, RMSE):
    r"""
    Plot the results of the episode

    # to understand difference between imshow and scatter see:
    https://stackoverflow.com/questions/75235652/why-does-plt-imshow-flip-coordinates-compared-to-plt-scatter
    """
    # plot
    if not os.path.isdir(os.path.join(output_folder, f"episode_{episode}")):
        os.makedirs(os.path.join(output_folder, f"episode_{episode}"))

    vmin = 0
    vmax = 4.5
    fig, ax = plt.subplots(3, 2, figsize=(10, 5))
    ax[0, 0].set_position([0.075, 0.70, 0.35, 0.25])
    ax[1, 0].set_position([0.075, 0.40, 0.35, 0.25])
    ax[2, 0].set_position([0.075, 0.10, 0.35, 0.25])
    ax[0, 1].set_position([0.60, 0.70, 0.35, 0.25])
    ax[1, 1].set_position([0.60, 0.40, 0.35, 0.25])
    ax[2, 1].set_position([0.60, 0.10, 0.35, 0.25])
    for i, x in enumerate(cpt_position):
        ax[0, 0].scatter(np.ones(len(depth[i])) * x, depth[i], c=np.flip(cpt[i]),
                      vmin=vmin, vmax=vmax, marker="s", s=3, cmap="viridis")

    ax[1,0].imshow(prediction.T, vmin=vmin, vmax=vmax, cmap="viridis")
    ax[2,0].imshow(new_data.T, vmin=vmin, vmax=vmax, cmap="viridis")#, aspect="auto")

    ax[0, 0].grid()
    ax[1, 0].grid()
    ax[2, 0].grid()
    ax[0, 0].xaxis.set_ticklabels([])
    ax[1, 0].xaxis.set_ticklabels([])
    ax[2, 0].set_xlabel("Distance [m]")
    ax[0, 0].set_ylabel("Known")
    ax[1, 0].set_ylabel("Interpolation")
    ax[2, 0].set_ylabel("True")
    ax[0, 0].set_xlim([0, np.max(unique_x)])
    ax[1, 0].set_xlim([0, np.max(unique_x)])
    ax[2, 0].set_xlim([0, np.max(unique_x)])

    ax[0, 1].set_axis_off()
    ax[1, 1].grid()
    ax[1, 1].plot(reward_global, color='b')
    ax[1, 1].set_ylabel("Reward")
    ax[1, 1].set_xlim((0, nb_episodes))
    ax[1, 1].set_ylim((-1, 10))
    ax[1, 1].xaxis.set_ticklabels([])

    ax[2, 1].plot(rmse_global, color='b')
    ax[2, 1].set_ylabel("Accuracy")
    ax[2, 1].set_xlabel("Number of episodes")
    ax[2, 1].grid()
    ax[2, 1].set_xlim((0, nb_episodes))
    ax[2, 1].set_ylim((0, 1))

    # add RMSE
    ax[0, 0].text(0, 36, f'Episode {episode}, state {state}, Accuracy={round(RMSE, 3)}',
                  horizontalalignment='left', verticalalignment='center')
    # Add a colorbar to a plot
    cax = ax[0,0].inset_axes([1.05, -2., 0.05, 2.5])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, cax=cax)
    cbar.set_label("IC")
    plt.savefig(os.path.join(os.path.join(output_folder, f"episode_{episode}", f"state_{state}_{file_name}.png")), dpi=600)
    plt.close()


def get_reward(settings:dict, idx_current_position: int, idx_known_positions: list, rmse_global:list,
               reward_global:list, data: pd.DataFrame, episode: int, output_folder: str, file_name: str,
               cost_cpt: float = 0, cost_rmse: float = 1, plots: bool = True, SchemaGAN=None,
               state_visitation_counts=None, beta: float = 0.1):
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

    # position of the known cpt
    total_list_of_positions = idx_known_positions + [idx_current_position]
    inputs_missing_field, known_field = preprocess_data(data, total_list_of_positions)
    if settings["reward_function"] == "get_reward_rmse_based":
        reward, prediction = get_reward_rmse_based(settings["interpolation_method"], idx_current_position, idx_known_positions,
                                       data, cost_cpt, cost_rmse, SchemaGAN)
    elif settings["reward_function"] == "get_reward_accuracy_based":
        reward, prediction = get_reward_accuracy_based(settings["interpolation_method"], idx_current_position, idx_known_positions,
                                       data, cost_cpt, cost_rmse, SchemaGAN)
    elif settings["reward_function"] == "get_reward_accuracy_based_with_state_visitation":
        reward, prediction = get_reward_variation_based(settings["interpolation_method"], idx_current_position,
                                                        idx_known_positions, data,  cost_cpt,  SchemaGAN)
    cpts = known_field[total_list_of_positions, :]
    unique_x = np.arange(0, known_field.shape[0])
    unique_y = np.arange(0, known_field.shape[1])
    depth = [unique_y.tolist() for i in range(len(total_list_of_positions))]
    # make plot
    if plots:
        make_plot(episode,
                  idx_current_position,
                  cpts,
                  depth,
                  total_list_of_positions,
                  prediction,
                  known_field,
                  unique_x,
                  unique_y,
                  output_folder,
                  file_name,
                  100,
                  reward_global,
                  rmse_global,
                  reward)
    return reward, np.sqrt(np.mean((known_field - prediction) ** 2))



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
        return np.argmax(q_values), "Exploit"
    else:
        return np.random.randint(0, len(actions)), "Explore"


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


def read_input_data_file(training_data_folder: str, suffix: str) -> pd.DataFrame:

    # randomly initialise the field
    files = os.listdir(training_data_folder)
    files.sort()

    idx = np.random.randint(len(files))
    idx = 50
    file = files[idx]
    # Read the .csv file into a pandas dataframe
    df = pd.read_csv(os.path.join(training_data_folder, file), delimiter=',')
    return file.split(".")[0], df


def check_update(num_steps, steps_update, length_data, minimum_batch_size):
    if (num_steps + 1) % steps_update == 0 and length_data >= minimum_batch_size:
        return True
    else:
        return False


def update_q_network(experiences: tuple,
                     gamma: float,
                     q_network: torch.nn.Module,
                     target_network: torch.nn.Module,
                     tau=1e-1,
                     entropy_coefficient=0.01,
                     weight_decay=1e-4):
    """
    Updates the weights of the Q network

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"]
      gamma: (float) The discount factor.
      q_network: model for predicting the q_values
      target_network: model for predicting the targets
      tau: (float) The soft update factor
      entropy_coefficient: (float) Coefficient for the entropy bonus
      weight_decay: (float) L2 regularization weight decay
    """

    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals, rsme = experiences

    with torch.no_grad():
        # Compute max Q(s', a')
        next_q_values = target_network.forward(next_states)
        max_qsa, indices = torch.max(next_q_values, axis=1)
        print(f"max indices: {indices}.")

    # Convert max_qsa to numpy if needed
    max_qsa_array = np.array([float(i) for i in max_qsa])
    # compute y values following Bellman Equation
    y_targets = rewards + (gamma * max_qsa_array * (1 - done_vals))

    # get q_values
    q_values = q_network.forward(states)
    actions = torch.Tensor(actions).long()
    q_values = q_values[torch.arange(q_values.shape[0]), actions]

    # Calculate entropy bonus (encourage exploration)
    q_distribution = F.softmax(q_values, dim=-1)
    entropy = -torch.sum(q_distribution * torch.log(q_distribution + 1e-8), dim=-1)
    entropy_bonus = entropy_coefficient * entropy

    # Compute loss with entropy bonus
    loss = F.mse_loss(q_values.view(-1, 1), torch.Tensor(y_targets).view(-1, 1)) #- entropy_bonus.mean()
    q_network.loss.append(loss.item())

    # Perform backward pass and optimization step
    q_network.optimizer.zero_grad()
    loss.backward()

    # Apply weight decay manually (PyTorch optimizer can handle this too)
    for param in q_network.parameters():
        param.grad.data.add_(weight_decay * param.data)

    q_network.optimizer.step()

    # Update weights of target q_network: soft update
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
    rsme = [experiences_replay[i][5] for i in idx]

    return (torch.Tensor(np.array(states)), np.array(actions), np.array(rewards), torch.Tensor(np.array(next_states)), np.array(done), np.array(rsme))


def dump_data(score, rmse, output_folder, file_name):
    summary = {"score": score, "rmse": rmse}
    # save the scores
    with open(os.path.join(output_folder, file_name), "w") as f:
        json.dump(summary, f, indent=2)


def define_default_settings_according_to_input(training_data_folder: str, suffix: str, state_function: str):
    # read a random file
    file_name, input_data = read_input_data_file(training_data_folder, suffix=suffix)
    # preprocess the data
    input_data, _ = preprocess_data(input_data, [2])
    # get number of pixels
    max_nb_pixels = input_data.shape[1]
    # get number of starting index
    starting_index = 20
    # define actions
    max_actions_step = int(max_nb_pixels // 3.2)
    actions = np.arange(25, max_actions_step, 25)
    # states are defined as [ mean and std for a 50 pixel window]
    state_len = get_states_size(state_function)

    return actions, max_nb_pixels, starting_index, state_len


def load_SchemaGAN_model(model_path: str):
    # load the model
    model = load_model(model_path)
    return model


def get_states_any_type(state_type:str, idx_current_position: int, idx_known_position: list[int], data: np.ndarray, SchemaGAN=None):
    if state_type == "get_state":
        return get_state(idx_current_position, data)
    elif state_type == "get_state2":
        return get_state2(idx_current_position, idx_known_position, data)
    elif state_type == "get_state_no_position":
        return get_state_no_position(idx_current_position, idx_known_position, data)
    elif state_type == "get_state_divitions":
        return get_state_divitions(idx_current_position, idx_known_position, data)
    elif state_type == "get_state_full_arrays":
        return get_state_full_arrays(idx_current_position, idx_known_position, data, SchemaGAN)
    else:
        raise ValueError(f"Unknown state type: {state_type}.")


def get_case(case_name: str, cases_file:str):
    # open json file
    with open(cases_file, "r") as f:
        cases = json.load(f)
    # get the case
    if case_name not in cases.keys():
        raise ValueError(f"Unknown case: {case_name}.")
    return cases[case_name]


def main(training_data_folder, plots=True):
    cases_file = "cases.json"
    case_name = "case_1"
    # get the case
    settings = get_case(case_name, cases_file)
    output_folder = os.path.join("CPTSuperLearnExperiment", case_name)
    if settings['interpolation_method'] == "SchemaGAN":
        schemaGAN_model_path = "P:/schemagan/model_000036.h5"
        # load the SchemaGAN model
        SchemaGAN = load_SchemaGAN_model(schemaGAN_model_path)
    else:
        SchemaGAN = None
    nb_episodes = 101
    epsilon = 1.0
    gamma = 0.99
    # settings
    num_steps_update = 2
    batch_size = 1
    memory_size = 500
    max_cpts = 7
    suffix = ".txt"
    actions, max_nb_pixels, max_starting_index, state_len = (
        define_default_settings_according_to_input(training_data_folder, suffix, settings["state_function"]))


    # initialise the logger
    logger = Logger("CPTSuperLearnExpirement",
                    f"results_{case_name}",
                    dict({"epsilon": epsilon,
                            "gamma": gamma,
                            "num_steps_update": num_steps_update,
                            "batch_size": batch_size,
                            "memory_size": memory_size,
                            "max_cpts": max_cpts,}, **settings))

    # delete output folder if exists
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)

    experiences_replay = deque(maxlen=memory_size)
    iteration = 1

    scores = []
    rmses = []

    experience_tupple = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done', 'rsme'])

    # create Q-network
    DeepQNetwork = SimpleNeuralNetwork(state_len, len(actions), [50, 50, 50, 40])
    # get weights of Q-network
    weights = [param.data for param in DeepQNetwork.parameters()]
    # create Target Q-network
    TargetQNetwork = SimpleNeuralNetwork(state_len, len(actions), [50, 50, 50, 40])
    # set weights of Target Q-network
    for i, param in enumerate(TargetQNetwork.parameters()):
        param.data = weights[i]
    state_visitation_counts = np.zeros(max_nb_pixels)
    # run through training episodes
    for episode in range(nb_episodes):
        print(f"episode: {episode}")

        # get data
        file_name, input_data = read_input_data_file(training_data_folder, suffix=suffix)
        # preprocess the data
        _, input_data_pre_processed = preprocess_data(input_data, [2])

        # get the starting location for this episode
        position_idx = get_starting_position_index(max_starting_index)
        known_positions_idx = [position_idx]
        # get the state
        state = get_states_any_type(settings["state_function"], position_idx, known_positions_idx, input_data_pre_processed, SchemaGAN)
        score = 0
        rmse = 0
        terminal = False

        last_index = []
        while not terminal:
            # From the current state S choose an action A using an ε-greedy policy
            with torch.no_grad():
                q_values = DeepQNetwork.forward(torch.Tensor(state))
            # avoid problem Numpy not Availble in Pytorch 1.9
            q_values_array = np.array([float(i) for i in q_values.detach()])
            #choose which action to take (i.e., where to move next)
            action_index, action_state = get_next_action(q_values_array, actions, epsilon)
            # perform the chosen action, and transition to the next state (i.e., move to the next location)
            position_idx = get_next_position(position_idx, actions[action_index], max_nb_pixels)
            state_visitation_counts[position_idx] += 1
            # compute reward
            reward, rmse = get_reward(settings=settings,
                                      idx_current_position=position_idx,
                                      idx_known_positions=known_positions_idx,
                                      data=input_data,
                                      episode=episode,
                                      output_folder=output_folder,
                                      file_name=file_name,
                                      rmse_global=rmses,
                                      reward_global=scores,
                                      plots=plots,
                                      SchemaGAN=SchemaGAN,
                                      state_visitation_counts=state_visitation_counts,
                                      beta=0.1,)
            score += reward
            last_index.append(position_idx)
            if (0 >= position_idx >= max_nb_pixels) or (len(known_positions_idx) + 1 > max_cpts) or len(last_index) != len(set(last_index)):
                terminal = True
            else:
                terminal = False

            # add to experience replay
            experiences_replay.append(experience_tupple(state, 
                                                        action_index, 
                                                        reward,
                                                        get_states_any_type(settings["state_function"], position_idx, known_positions_idx, input_data_pre_processed, SchemaGAN),
                                                        terminal,
                                                        rmse))
            # check if it is time to update the Q-network
            update = check_update(iteration, num_steps_update, len(experiences_replay), batch_size)

            if update:
                print("### Performing update ###")
                # get experiences from experiences replay
                experiences = get_experiences(experiences_replay, batch_size)

                # update the Q-network
                DeepQNetwork, TargetQNetwork = update_q_network(experiences, gamma, DeepQNetwork, TargetQNetwork)

            # update the state
            state = get_states_any_type(settings["state_function"], position_idx, known_positions_idx, input_data_pre_processed, SchemaGAN)
            # update known positions
            known_positions_idx.append(position_idx)
            # update counter
            iteration += 1

            if len(DeepQNetwork.loss) > 0:
                loss = DeepQNetwork.loss[-1]
            else:
                loss = 0
            # log the data
            logger.log({"score": score, 
                        "rmse": rmse, 
                        "action_idx": action_index,
                        "position": position_idx,
                        "reward": reward,
                        "loss": loss,
                        "epsilon": epsilon,
                        "episode": episode, 
                        "iteration": iteration,
                        "action": action_state,})
        # update epsilon
        epsilon = get_new_eps(epsilon)
        scores.append(score)
        rmses.append(rmse)

        if episode % 100 == 0:
            print(f"episode: {episode}, score: {score:.3f}, rmse: {rmse:.3f}")
            # save the model
            DeepQNetwork.save_model(output_folder, f"deep_{episode}.pth")
            TargetQNetwork.save_model(output_folder, f"target_{episode}.pth")
            # dump experience replay into a pickle file
            with open(os.path.join(output_folder, f"experience_replay_{episode}.pkl"), "wb") as f:
                # collect the experiences in list
                experiences_replay_list = [exp[0] for exp in experiences_replay]
                pickle.dump(experiences_replay_list, f)
            with open(os.path.join(output_folder, f"actions_{episode}.pkl"), "wb") as f:
                # collect the experiences in list
                experiences_replay_list = [exp[1] for exp in experiences_replay]
                pickle.dump(experiences_replay_list, f)
            # dump data
            dump_data(scores, rmses, output_folder, f"summary_{episode}.json")
    logger.finish()


if __name__ == "__main__":
    main("P:/schemagan/synthetic_database/512x32_20k/validation")
    print("Done")
