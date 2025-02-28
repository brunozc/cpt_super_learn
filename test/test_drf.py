import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random
import shutil
import pytest
import numpy as np
import torch
import tensorflow as tf

from CPTSuperLearn.utils import input_random_data_file, write_score
from CPTSuperLearn.environment import CPTEnvironment
from CPTSuperLearn.agent import DQLAgent
from CPTSuperLearn.interpolator import InverseDistance, SchemaGANInterpolator
from CPTSuperLearn.utils import download_file

# fix all the seeds
random.seed(14)
np.random.seed(14)
torch.manual_seed(14)
tf.random.set_seed(14)
tf.keras.utils.set_random_seed(14)



def main(nb_episodes: int, cpt_env: CPTEnvironment, agent: DQLAgent, training_data_folder: str, output_folder: str):
    """
    Train the DRL model

    Parameters
    ----------
    :param nb_episodes: number of episodes
    :param cpt_env: environment
    :param agent: agent
    :param training_data_folder: folder with the training data
    :param output_folder: output folder
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

    agent.save_model(os.path.join(output_folder, "cpt_model.pth"))
    write_score(range(nb_episodes), total_score, os.path.join(output_folder, "cpt_score.txt"))


def test_inv_dist():
    """
    Test the model with inverse distance interpolation
    """
    training_data_folder = "test/data/train"
    num_episodes = 10
    actions = [5, 10, 25, 50]  # actions in number of pixels
    output_folder = "./results_test"

    cpt_env = CPTEnvironment(actions,
                             max_nb_cpts=20,
                             weight_reward_cpt=0.5,
                             image_width=512,
                             max_first_step=1,
                             interpolation_method=InverseDistance(nb_points=6),
                             )

    cpt_agent = DQLAgent(state_size=6,
                         action_size=len(actions),
                         learning_rate=1e-4,
                         gamma=0.99,
                         epsilon_start=0.95,
                         epsilon_end=0.05,
                         epsilon_decay=0.90,
                         memory_size=10000,
                         batch_size=64,
                         nb_steps_update=2)

    main(num_episodes, cpt_env, cpt_agent, training_data_folder, output_folder)

    with open(os.path.join(output_folder, "cpt_score.txt"), "r") as f:
        scores = f.read().splitlines()
    scores = [list(map(float, i.split(";"))) for i in scores]

    with open("test/data/cpt_score_id.txt", "r") as f:
        scores_test = f.read().splitlines()
    scores_test = [list(map(float, i.split(";"))) for i in scores_test]

    np.testing.assert_almost_equal(np.array(scores), np.array(scores_test), decimal=2)

    shutil.rmtree(output_folder)

@pytest.mark.skip(reason="SchemaGAN model test is WIP")
def test_inv_schemaGAN():
    """
    Test the model with schemaGAN interpolation
    """

    schemaGAN_path = "./test/schemaGAN/schemaGAN.h5"
    # download the data
    if not os.path.isfile(schemaGAN_path):
        download_file("https://zenodo.org/records/13143431/files/schemaGAN.h5", schemaGAN_path)

    training_data_folder = "test/data/train"
    num_episodes = 10
    actions = [5, 10, 25, 50]  # actions in number of pixels
    output_folder = "./results_test"

    cpt_env = CPTEnvironment(actions,
                             max_nb_cpts=20,
                             weight_reward_cpt=0.5,
                             image_width=512,
                             max_first_step=1,
                             interpolation_method=SchemaGANInterpolator(schemaGAN_path),
                             )

    cpt_agent = DQLAgent(state_size=6,
                         action_size=len(actions),
                         learning_rate=1e-4,
                         gamma=0.99,
                         epsilon_start=0.95,
                         epsilon_end=0.05,
                         epsilon_decay=0.90,
                         memory_size=10000,
                         batch_size=64,
                         nb_steps_update=2)

    main(num_episodes, cpt_env, cpt_agent, training_data_folder, output_folder)

    with open(os.path.join(output_folder, "cpt_score.txt"), "r") as f:
        scores = f.read().splitlines()
    scores = [list(map(float, i.split(";"))) for i in scores]

    with open("test/data/cpt_score_sg.txt", "r") as f:
        scores_test = f.read().splitlines()
    scores_test = [list(map(float, i.split(";"))) for i in scores_test]

    np.testing.assert_almost_equal(np.array(scores), np.array(scores_test), decimal=2)

    shutil.rmtree(output_folder)
