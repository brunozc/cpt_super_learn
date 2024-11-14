import os
import random
import numpy as np
import torch

from CPTSuperLearn.utils import input_random_data_file, write_rmse
from CPTSuperLearn.environment import CPTEnvironment
from CPTSuperLearn.agent import DQLAgent


# fix all the seeds
random.seed(14)
np.random.seed(14)
torch.manual_seed(14)

def main(nb_episodes: int, cpt_env: CPTEnvironment, agent: DQLAgent, training_data_folder: str, output_folder: str,
         make_plots=False):
    """
    Train the DRL model

    Parameters
    ----------
    :param nb_episodes: number of episodes
    :param cpt_env: environment
    :param agent: agent
    :param training_data_folder: folder with the training data
    :param output_folder: output folder
    :param make_plots: make plots
    """

    total_score = []

    for episode in range(nb_episodes):
        file_name, image_data = input_random_data_file(training_data_folder)

        state = cpt_env.reset(file_name, image_data)
        score = 0
        terminal = False

        while not terminal:
            action_index = agent.get_next_action(state)
            next_state, reward, terminal = cpt_env.step(action_index)
            agent.step(state, action_index, reward, next_state, terminal)
            state = next_state
            score += reward

            if terminal:
                break

        total_score.append(score)
        average_score = np.mean(total_score)

        if make_plots:
            cpt_env.plot_environment(os.path.join(output_folder, "training", f"{file_name}_episode_{episode}.png"))

        if episode % 10 == 0:
            print(f"Episode {episode} Average Score: {average_score:.2f} Epsilon: {agent.epsilon:.2f}")

    agent.save_model(os.path.join(output_folder, "cpt_model.pth"))
    write_rmse(range(nb_episodes), total_score, os.path.join(output_folder, "cpt_score.txt"))


if __name__ == "__main__":
    training_data_folder = "./data_fabian/train"
    num_episodes = 100
    actions = [10, 25, 50, 100, 150]  # actions in number of pixels
    output_folder = "results"

    cpt_env = CPTEnvironment(actions,
                             max_nb_cpts=50,
                             cpt_cost=0.1,
                             image_width=512,
                             max_first_step=20,
                             interpolator_points=6,
                             )
    cpt_agent = DQLAgent(state_size=6,
                         action_size=len(actions),
                         learning_rate=1e-4,
                         gamma=0.99,
                         epsilon_start=1.,
                         epsilon_end=0.01,
                         epsilon_decay=0.995,
                         memory_size=10000,
                         batch_size=64,
                         nb_steps_update=10)

    main(num_episodes, cpt_env, cpt_agent, training_data_folder, output_folder)