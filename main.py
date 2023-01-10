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

    # interpolate at the entire field
    if len(cpt) < 2:
        return -10

    idw = InverseDistance()
    cpt_position, idx = np.unique(cpt_position, return_index=True)
    idw.interpolate(cpt_position, np.array(cpt)[idx])
    idw.predict(unique_x)

    # import matplotlib.pylab as plt
    # for i in range(len(cpt)):
    #     plt.plot(cpt[i], label="cpt")
    #     plt.plot(idw.prediction.T[:, i], label="cpt")
    #     plt.show()

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
    reward += -1 * RMSE

    # plot
    import matplotlib.pylab as plt
    import matplotlib as mpl
    if not os.path.isdir(os.path.join("./results", f"episode_{episode}")):
        os.makedirs(os.path.join("./results", f"episode_{episode}"))

    vmin = 5
    vmax = 30
    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
    ax[0].set_position([0.1, 0.70, 0.75, 0.25])
    ax[1].set_position([0.1, 0.40, 0.75, 0.25])
    ax[2].set_position([0.1, 0.10, 0.75, 0.25])
    for i, x in enumerate(cpt_position):
        ax[0].scatter(np.ones(len(depth[i])) * x, depth[i], c=cpt[i],
                      vmin=vmin, vmax=vmax, marker="s", s=15,  cmap="viridis")

    x, y = np.meshgrid(unique_x, unique_y, indexing="ij")
    ax[1].scatter(x, y, c=idw.prediction, vmin=vmin, vmax=vmax, marker="s", s=30,  cmap="viridis")
    ax[2].scatter(x, y, c=new_data, vmin=vmin, vmax=vmax, marker="s", s=30,  cmap="viridis")
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[2].set_xlabel("Position")
    ax[0].set_ylabel("Known")
    ax[1].set_ylabel("Interpolation")
    ax[2].set_ylabel("True")
    ax[0].set_xlim([0, nb_points-1])
    # add RMSE
    ax[0].text(0, 8, f'Episode {episode}, state {state}, RMSE={round(RMSE, 3)}', horizontalalignment='left', verticalalignment='center')
    # Add a colorbar to a plot
    cax = ax[0].inset_axes([1.05, -2., 0.05, 2.5])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, cax=cax)
    cbar.set_label("Values")
    plt.savefig(os.path.join(os.path.join("./results", f"episode_{episode}", f"state_{state}.png")))
    plt.close()

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
