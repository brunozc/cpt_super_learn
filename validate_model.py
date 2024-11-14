import os
from tqdm import tqdm
import torch
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

from CPTSuperLearn.utils import read_data_file, write_score
from CPTSuperLearn.environment import CPTEnvironment
from CPTSuperLearn.agent import DQLAgent


def evaluate_model(cpt_env: CPTEnvironment, agent: DQLAgent, validation_data_folder: str, output_folder: str,
                   make_plots=False):
    """
    Evaluate the model on the validation dataset

    Parameters:
    -----------
    :param cpt_env: CPT environment
    :param agent: DQL agent
    :param validation_data_folder: folder with the validation data
    :param output_folder: output folder
    :param make_plots: make plots
    """

    # Load the saved model
    agent.load_model()
    agent.qnetwork_local.eval()  # Set the model to evaluation mode

    # Test on validation dataset
    validation_scores = []
    rmse_scores = []

    val_files = os.listdir(validation_data_folder)

    for fil in tqdm(val_files):
        file_name, image_data = read_data_file(os.path.join(validation_data_folder, fil))
        state = cpt_env.reset(file_name, image_data)
        score = 0
        terminal = False

        while not terminal:
            action_index = agent.get_next_action(state, training=False)
            next_state, reward, terminal = cpt_env.step(action_index)
            state = next_state
            score += reward

            if terminal:
                break

        validation_scores.append(score)
        rmse = np.sqrt(np.mean((cpt_env.true_data - cpt_env.predicted_data) ** 2))
        rmse_scores.append(rmse)

        if make_plots:
            cpt_env.plot_environment(os.path.join(output_folder, "images", f"file_{file_name}"))


    val_files = [os.path.splitext(os.path.basename(f))[0] for f in val_files]
    write_score(val_files, validation_scores,  os.path.join(output_folder, "validation_score.txt"))
    write_score(val_files, rmse_scores,  os.path.join(output_folder, "validation_rmse.txt"))



# Example usage
if __name__ == "__main__":
    validation_data_folder = "./data_fabian/validation"
    actions = [10, 25, 50, 100, 150]  # actions in number of pixels
    output_folder = "results/validation"

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
                         nb_steps_update=10,
                         model_path="results/cpt_model.pth")

    evaluate_model(cpt_env, cpt_agent, validation_data_folder, output_folder, make_plots=True)