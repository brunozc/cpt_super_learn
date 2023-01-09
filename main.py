# https://www.youtube.com/watch?v=iKdlKYG78j4
# https://colab.research.google.com/drive/1E2RViy7xmor0mhqskZV14_NUj2jMpJz3#scrollTo=3N5BB0m0JHIn

import os
import numpy as np
from IDW import InverseDistance


# define environment
## states
nb_points = 51
## actions
actions = [1, 2, 5, 10]
q_values = np.zeros((nb_points, 1, len(actions)))
## rewards
rewards = np.full((nb_points, 1), -1)


def _get_reward(current_position, known_positions, data, episode, state, cost_cpt=-1):

    with open(data, "r") as fi:
        data = fi.read().splitlines()
        data = [np.array(i.split(";")).astype(float) for i in data[1:]]
    data = np.array(data)

    # unique coordinates CPT
    unique_x = np.unique(data[:, 0])

    # check if len(unique_x) == nb_points
    if len(unique_x) != nb_points:
        raise ValueError("nb_points is not equal to the number of unique x coordinates")

    # read cpts at the known positions
    cpt = []
    cpt_position = []
    for k in known_positions:
        idx = np.where(data[:, 0] == k)[0]
        cpt.append(data[idx, 2])
        cpt_position.append(k)

    # read cpt at current position
    idx = np.where(data[:, 0] == current_position)[0]
    cpt.append(data[idx, 2])
    cpt_position.append(current_position)

    # interpolate at the entire field
    if len(cpt) < 2:
        return -10

    idw = InverseDistance()
    idw.interpolate(cpt_position, cpt)
    idw.predict(unique_x)

    # show plot
    import matplotlib.pylab as plt
    import matplotlib as mpl
    if not os.path.isdir(os.path.join("./results", f"episode_{episode}")):
        os.makedirs(os.path.join("./results", f"episode_{episode}"))

    vmin = 5
    vmax = 30
    fig, ax = plt.subplots(2, 1)
    ax[0].set_position([0.1, 0.1, 0.7, 0.35])
    ax[1].set_position([0.1, 0.5, 0.7, 0.35])
    ax[0].scatter(data[:, 0], data[:, 1], c=data[:, 2], vmin=vmin, vmax=vmax)
    ax[1].scatter(data[:, 0], data[:, 1], c=idw.prediction.T.ravel(), vmin=vmin, vmax=vmax)
    ax[0].grid()
    ax[1].grid()
    ax[0].set_xlabel("Position")
    ax[0].set_ylabel("Depth")
    ax[1].set_ylabel("Depth")
    # Add a colorbar to a plot
    cax = ax[0].inset_axes([1.015, 0., 0.05, 2.15])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, cax=cax)
    cbar.set_label("Values")
    plt.savefig(os.path.join(os.path.join("./results", f"episode_{episode}", f"state_{state}.png")))
    plt.close()

    # compare at the entire field RMSE
    RMSE = np.sqrt(np.mean((data[:, 2] - idw.prediction.T.ravel()) ** 2))


    # cost of cpts
    reward = len(known_positions) * cost_cpt

    # cost of the RMSE
    reward += -1 * RMSE

    return reward



def _is_terminal_state(current_position):
    if current_position >= nb_points - 1:
        return True
    else:
        return False

def _get_starting_position():
    # return np.random.randint(0, nb_points-1)
    # always starts at 0
    return 0

def _get_next_action(current_position, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_position, 0])
    else:
        return np.random.randint(0, len(actions))

def _get_next_position(current_position, action):
    if current_position + action >= nb_points:
        return current_position
    else:
        return current_position + action


def _get_path(position):
    if _is_terminal_state(position):
        print("invalid starting point")
        return []

    current_position = position
    path = [current_position]
    while _is_terminal_state(current_position) != True:
        action = _get_next_action(current_position, 1)
        current_position = _get_next_position(current_position, action)
        path.append(current_position)
    return path


def main():
    #define training parameters
    epsilon = 0.9 #the percentage of time when we should take the best action (instead of a random action)
    discount_factor = 0.9 #discount factor for future rewards
    learning_rate = 0.9 #the rate at which the AI agent should learn

    #run through 1000 training episodes
    for episode in range(1000):
        print(f"episode: {episode}")
        #get the starting location for this episode
        position = _get_starting_position()
        known_positions = []
        #continue taking actions (i.e., moving) until we reach a terminal state
        #(i.e., until we reach the item packaging area or crash into an item storage location)
        state = 0
        while _is_terminal_state(position) != True:
            #choose which action to take (i.e., where to move next)
            action_index = _get_next_action(position, epsilon)
            #perform the chosen action, and transition to the next state (i.e., move to the next location)
            old_position = position  #store the old row and column indexes
            position = _get_next_position(position, action_index)

            #receive the reward for moving to the new state, and calculate the temporal difference
            reward = _get_reward(position, known_positions, "./data/slice.txt", episode, state)

            # reward = rewards[position, 0]
            old_q_value = q_values[position, 0, action_index]
            temporal_difference = reward + (discount_factor * np.max(q_values[position])) - old_q_value

            #update the Q-value for the previous state and action pair
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            q_values[position, 0, action_index] = new_q_value

            # update known positions
            known_positions.append(position)

            state += 1

    print('Training complete!')
    print(_get_path(1))


if __name__ == "__main__":
    main()
