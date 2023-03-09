import os
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
# interpolation
from IDW import InverseDistance


def make_plot(episode, state, cpt, depth, cpt_position, idw, new_data, RMSE,
              nb_episodes, rmse_global, reward_global, unique_x, unique_y, output_folder):
    r"""
    Plot the results of the episode
    """
    # plot
    if not os.path.isdir(os.path.join(output_folder, f"episode_{episode}")):
        os.makedirs(os.path.join(output_folder, f"episode_{episode}"))

    vmin = 5
    vmax = 30
    fig, ax = plt.subplots(3, 2, figsize=(10, 5))
    ax[0, 0].set_position([0.075, 0.70, 0.35, 0.25])
    ax[1, 0].set_position([0.075, 0.40, 0.35, 0.25])
    ax[2, 0].set_position([0.075, 0.10, 0.35, 0.25])
    ax[0, 1].set_position([0.60, 0.70, 0.35, 0.25])
    ax[1, 1].set_position([0.60, 0.40, 0.35, 0.25])
    ax[2, 1].set_position([0.60, 0.10, 0.35, 0.25])
    for i, x in enumerate(cpt_position):
        ax[0, 0].scatter(np.ones(len(depth[i])) * x, depth[i], c=cpt[i],
                         vmin=vmin, vmax=vmax, marker="s", s=3,  cmap="viridis")

    x, y = np.meshgrid(unique_x, unique_y, indexing="ij")
    ax[1, 0].scatter(x, y, c=idw.prediction, vmin=vmin, vmax=vmax, marker="s", s=15,  cmap="viridis")
    ax[2, 0].scatter(x, y, c=new_data, vmin=vmin, vmax=vmax, marker="s", s=15,  cmap="viridis")
    ax[0, 0].grid()
    ax[1, 0].grid()
    ax[2, 0].grid()
    ax[0, 0].xaxis.set_ticklabels([])
    ax[1, 0].xaxis.set_ticklabels([])
    ax[2, 0].set_xlabel("Position")
    ax[0, 0].set_ylabel("Known")
    ax[1, 0].set_ylabel("Interpolation")
    ax[2, 0].set_ylabel("True")
    ax[0, 0].set_xlim([0, nb_points-1])
    ax[1, 0].set_xlim([0, nb_points-1])
    ax[2, 0].set_xlim([0, nb_points-1])

    ax[0, 1].set_axis_off()
    ax[1, 1].grid()
    ax[1, 1].plot(reward_global, color='b')
    ax[1, 1].set_ylabel("Reward")
    ax[1, 1].set_xlim((0, nb_episodes))
    ax[1, 1].set_ylim((0, -100))
    ax[1, 1].xaxis.set_ticklabels([])

    ax[2, 1].plot(rmse_global, color='b')
    ax[2, 1].set_ylabel("RMSE")
    ax[2, 1].set_xlabel("Number of episodes")
    ax[2, 1].grid()
    ax[2, 1].set_xlim((0, nb_episodes))
    ax[2, 1].set_ylim((0, 21))

    # add RMSE
    ax[0, 0].text(0, 8, f'Episode {episode}, state {state}, RMSE={round(RMSE, 3)}',
                  horizontalalignment='left', verticalalignment='center')
    # Add a colorbar to a plot
    cax = ax[0, 0].inset_axes([1.05, -2., 0.05, 2.5])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, cax=cax)
    cbar.set_label("Values")
    plt.savefig(os.path.join(os.path.join(output_folder, f"episode_{episode}", f"state_{state}.png")))
    plt.close()


def get_reward(current_position: int, known_positions: list, data: str,
               episode: int, state: int, nb_episodes: int,
               rmse_global: list, reward_global: list, output_folder,
               cost_cpt: float = -1, cost_rmse: float = 5, plots: bool = True):
    r"""
    Custom reward function
    The interpolation is currently Inversed Distance Weighting (IDW)

    Parameters
    ----------
    current_position (int): current position of the agent
    known_positions (list): list of known positions
    data (str): path to the data
    episode (int): current episode
    state (int): current state
    nb_episodes (int): total number of episodes
    rmse_global (list): list of RMSE for each episode
    reward_global (list): list of rewards for each episode
    cost_cpt (float): cost of a cpt
    cost_rmse (float): cost of a rmse
    output_folder (str): path to the output folder
    plots (bool): if True, plots are generated
    """

    # read data
    with open(data, "r") as fi:
        data = fi.read().splitlines()
        data = [np.array(i.split(";")).astype(float) for i in data[1:]]
    data = np.array(data)

    # unique coordinates CPT
    unique_x = np.unique(data[:, 0])
    unique_x.sort()
    unique_y = np.unique(data[:, 1])
    unique_y.sort()

    # check if len(unique_x) == nb_points
    if len(unique_x) != nb_points:
        raise ValueError("nb_points is not equal to the number of unique x coordinates")

    # read cpts at the known positions
    cpt = []
    depth = []
    cpt_position = []
    for k in known_positions:
        idx = np.where(data[:, 0] == k)[0]
        depth.append(data[idx, 1])
        cpt.append(data[idx, 2])
        cpt_position.append(k)

    # read cpt at current position
    idx = np.where(data[:, 0] == current_position)[0]
    depth.append(data[idx, 1])
    cpt.append(data[idx, 2])
    cpt_position.append(current_position)

    # get unique cpt positions
    cpt_position, idx = np.unique(cpt_position, return_index=True)
    # interpolate at the entire field
    if len(cpt_position) < 2:
        return -10, 10

    # perform interpolation
    idw = InverseDistance()
    idw.interpolate(cpt_position, np.array(cpt)[idx])
    idw.predict(unique_x)

    # reshape data
    new_data = []
    for i in unique_x:
        idx = np.where(data[:, 0] == i)[0]
        new_data.append(data[idx, 2])
    new_data = np.array(new_data)

    # compare at the entire field RMSE
    RMSE = np.sqrt(np.mean((new_data - idw.prediction) ** 2))

    # cost of cpts
    reward = len(known_positions) * cost_cpt

    # cost of the RMSE
    reward += -1 * RMSE * cost_rmse

    # make plot
    if plots:
        make_plot(episode, state, cpt, depth, cpt_position, idw, new_data, RMSE,
                  nb_episodes, rmse_global, reward_global, unique_x, unique_y, output_folder)
    return reward, RMSE


def is_terminal_state(current_position: int) -> bool:
    r""""
    Check if the current state is terminal

    Parameters:
    -----------
    current_position (int): current position
    """
    if (current_position >= nb_points - 1) or (current_position <= 0):
        return True
    else:
        return False

def get_starting_position(actions: list) -> int:
    r"""
    Get the starting position

    Parameters:
    -----------
    actions (list): list of actions
    """
    return actions[np.random.randint(len(actions))]


def get_next_action(current_position: int, epsilon: float) -> int:
    r"""
    Get the next action

    Parameters:
    -----------
    current_position (int): current position
    epsilon (float): probability of taking a random action
    """

    if np.random.random() < epsilon:
        return np.argmax(q_values[current_position, 0])
    else:
        return np.random.randint(0, len(actions))

def get_next_position(current_position: int, action: int) -> int:
    r"""
    Get the next position

    Parameters:
    -----------
    current_position (int): current position
    action (int): number of steps to take
    """
    if (current_position + action >= nb_points) or (current_position + action < 0):
        return current_position
    else:
        return current_position + action


def get_path(position: int) -> list:
    r"""
    Get the path from the starting position to the terminal state

    Parameters:
    -----------
    position (int): starting position
    """
    if is_terminal_state(position):
        print("invalid starting point")
        return []

    current_position = position
    path = [current_position]
    while not is_terminal_state(current_position):
        action_index = get_next_action(current_position, 1)
        current_position = get_next_position(current_position, actions[action_index])
        path.append(current_position)
    return path


def main(settings, input_data, output_folder="results", seed=14, plots=True):
    r"""
    Main function for the Q-learning algorithm

    Parameters:
    -----------
    settings (dict): dictionary with the settings
    input_data (str): path to the input data
    output_folder (str): path to the output folder
    seed (int): seed for the random number generator
    plots (bool): if True, make plots
    """

    # set seed
    np.random.seed(seed)


    epsilon = settings["epsilon"]  # the percentage of time when we should take the best action (instead of a random action)
    discount_factor = settings["discount_factor"]  # discount factor for future rewards
    learning_rate = settings["learning_rate"]  # the rate at which the AI agent should learn
    nb_episodes = settings["nb_episodes"]


    rmse_global = [10]
    reward_global = [-100]

    #run through training episodes
    for episode in range(nb_episodes):
        print(f"episode: {episode}")
        #get the starting location for this episode
        position = get_starting_position(actions)
        known_positions = [position]
        #continue taking actions (i.e., moving) until we reach a terminal state
        #(i.e., until we reach the item packaging area or crash into an item storage location)
        state = 0
        while not is_terminal_state(position):
            #choose which action to take (i.e., where to move next)
            action_index = get_next_action(position, epsilon)
            #perform the chosen action, and transition to the next state (i.e., move to the next location)
            position = get_next_position(position, actions[action_index])

            #receive the reward for moving to the new state, and calculate the temporal difference
            reward, rmse = get_reward(position, known_positions, input_data, episode, state,
                                      nb_episodes, rmse_global, reward_global, output_folder, plots=plots)

            # reward = rewards[position, 0]
            old_q_value = q_values[position, 0, action_index]
            temporal_difference = reward + (discount_factor * np.max(q_values[position, 0])) - old_q_value

            #update the Q-value for the previous state and action pair
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            q_values[position, 0, action_index] = new_q_value

            # update known positions
            known_positions.append(position)

            state += 1

        if state == 0:
            reward = reward_global[0]
            rmse = rmse_global[0]

        reward_global.append(reward)
        rmse_global.append(rmse)

    print('Training complete!')


if __name__ == "__main__":
    settings = {"epsilon": 0.9,  # the percentage of time when we should take the best action (instead of a random action)
                "discount_factor": 0.8,  # discount factor for future rewards
                "learning_rate": 0.8,  # the rate at which the AI agent should learn
                "nb_episodes": 100  # the number of episodes to run the training
                }

    # define environment
    # states
    nb_points = 51
    # actions
    actions = [1, 5, 10, 15]
    q_values = np.zeros((nb_points, 1, len(actions)))
    # rewards
    rewards = np.full((nb_points, 1), -1)
    main(settings, r"./data/slice.txt", output_folder="./results", plots=True)
    print(get_path(1))
    print(get_path(5))
