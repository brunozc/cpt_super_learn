import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch

from CPTSuperLearn.interpolator import InverseDistance


class CPTEnvironment:
    def __init__(self, action_list: List[int], max_nb_cpts: int, cpt_cost: float, image_width: int,
                 max_first_step: int, interpolator_points: int):
        """
        Initialize the CPT environment

        Parameters:
        -----------
        :param action_list: list of possible actions
        :param max_nb_cpts: maximum number of CPTs
        :param cpt_cost: cost of a CPT
        :param image_width: width of the image
        :param max_first_step: maximum first step
        :param interpolator_points: number of points for the interpolator
        """

        self.action_list = action_list
        self.max_nb_cpts = max_nb_cpts
        self.cpt_cost = cpt_cost
        self.image_width = image_width
        self.maximum_first_step = max_first_step

        self.reward_out_of_bounds = 0

        self.interpolator = InverseDistance(nb_points=interpolator_points)

        self.current_image = None
        self.current_image_id = None
        self.sampled_positions = []
        self.sampled_values = []

    def reset(self, image_id: str, image_data: np.ndarray):
        """
        Reset environment with new image

        Parameters:
        -----------
        :param image_id: image id
        :param image_data: image data
        """
        self.current_image_id = image_id
        self.current_image = image_data
        self.sampled_positions = []
        self.sampled_values = []

        # Sample first cpt
        first_cpt_index = self._get_starting_position_index(self.maximum_first_step)
        self.sampled_positions.append(first_cpt_index)
        image_data = image_data[image_data[:, 0] == first_cpt_index]
        self.sampled_values.append(image_data[:, 2])

        # get the state
        state = self._get_state()

        return state

    def _get_state(self) -> torch.Tensor:
        """
        Get the current state

        At the moment the state consists of:
        - Mean of the current CPT
        - Standard deviation of the current CPT
        - Mean cosine similarity with the previous CPTs
        - Standard deviation of the cosine similarity with the previous CPTs
        - Normalized position of the current CPT
        - Fraction of the CPTs sampled

        Returns:
        --------
        :return: current state
        """

        # If only one CPT exists
        if len(self.sampled_positions) == 1:
            current_mean = np.mean(self.sampled_values[0])
            current_std = np.std(self.sampled_values[0])
            similarity = 0.0
        else:
            current_values = self.sampled_values[-1]
            previous_values = self.sampled_values[:-1]

            # Calculate statistics
            current_mean = np.mean(current_values)
            current_std = np.std(current_values)

            # Compute cosine similarity
            similarity = [np.dot(current_values, p) / (np.linalg.norm(current_values) * np.linalg.norm(p)) for  p in previous_values]

        # Add normalized position information
        normalized_position = self.sampled_positions[-1] / self.image_width

        state = torch.FloatTensor([current_mean, current_std, np.mean(similarity), np.std(similarity),
                                   normalized_position, len(self.sampled_positions) / self.max_nb_cpts])

        return state

    def step(self, action_index: int) -> Tuple[torch.Tensor, float, bool]:
        """
        Take a step in the environment

        Parameters:
        -----------
        :param action_index: index of the action to take

        Returns:
        --------
        :return: state, reward, terminal
        """
        next_position = self.sampled_positions[-1] + self.action_list[action_index]

        # Check if the next position is within the image
        if (next_position >= self.image_width) or (next_position < 0):
            reward = self.reward_out_of_bounds
            terminal = True
            return self._get_state(), reward, terminal

        reward = self._get_reward(next_position)
        terminal = False

        return self._get_state(), reward, terminal

    def _get_reward(self, next_position: int) -> float:
        """
        Get the reward for the next position

        Parameters:
        -----------
        :param next_position: next position

        Returns:
        --------
        :return: reward
        """

        new_cpt = self.current_image[self.current_image[:, 0] == next_position]

        self.sampled_positions.append(next_position)
        self.sampled_values.append(new_cpt[:, 2])

        # Perform interpolation
        self.interpolator.interpolate(np.array(self.sampled_positions), np.array(self.sampled_values))

        # Predict for all x positions
        all_x = np.unique(self.current_image[:, 0])
        all_y = np.unique(self.current_image[:, 1])
        self.interpolator.predict(all_x)
        predicted_values = self.interpolator.prediction

        true_data = self.current_image[:, 2].reshape(len(all_x), len(all_y))

        # compare at the entire field RMSE
        RMSE = np.sqrt(np.mean((true_data - predicted_values) ** 2))

        # Calculate reward
        cpt_penalty = self.cpt_cost * len(self.sampled_positions)
        reward = -RMSE - cpt_penalty

        return reward

    @staticmethod
    def _get_starting_position_index(maximum_step: int) -> int:
        """
        Get the index of the starting position

        Parameters:
        -----------
        maximum_step (int): maximum step in pixels of the movement
        """
        return np.random.randint(maximum_step)

    def plot_environment(self, filename: str):
        """
        Plot the environment

        # to understand difference between imshow and scatter see:
        https://stackoverflow.com/questions/75235652/why-does-plt-imshow-flip-coordinates-compared-to-plt-scatter

        Parameters:
        -----------
        :param filename: filename to save the plot
        """

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        vmin = 1
        vmax = 4
        fig, ax = plt.subplots(3, 1, figsize=(10, 5), sharex=True, sharey=True)
        ax[0].set_position([0.075, 0.70, 0.775, 0.175])
        ax[1].set_position([0.075, 0.40, 0.775, 0.25])
        ax[2].set_position([0.075, 0.10, 0.775, 0.25])

        all_x = np.unique(self.current_image[:, 0])
        all_y = np.unique(self.current_image[:, 1])
        predicted_values = self.interpolator.prediction

        true_data = self.current_image[:, 2].reshape(len(all_x), len(all_y))

        for pos, dat in zip(self.sampled_positions, self.sampled_values):
            ax[0].scatter(np.ones(len(all_y)) * pos, all_y, c=dat,
                          vmin=vmin, vmax=vmax, marker="s", s=3, cmap="viridis")

        x, y = np.meshgrid(all_x, all_y)
        ax[1].imshow(predicted_values.T, vmin=vmin, vmax=vmax, cmap="viridis",
                     extent=[0, np.max(x), np.max(y), 0])#, aspect="auto")
        ax[1].invert_yaxis()

        ax[2].imshow(true_data.T, vmin=vmin, vmax=vmax, cmap="viridis",
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
        ax[0].set_xlim([0, np.max(all_x)])

        # add RMSE
        # ax[0].text(0, 14, f'Episode {episode:.0f}, position {state:.0f}, RMSE={RMSE:.3f}',
                    #   horizontalalignment='left', verticalalignment='center')
        # Add a colorbar to a plot
        cax = ax[0].inset_axes([1.05, -2., 0.05, 2.5])
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, cax=cax)
        cbar.set_label("IC")
        plt.savefig(f"{filename}.png", dpi=600)
        plt.close()