import os
import random
import numpy as np
import torch

from CPTSuperLearn.utils import input_random_data_file
from CPTSuperLearn.environment import CPTEnvironment
from CPTSuperLearn.agent import DQLAgent
from CPTSuperLearn.interpolator import InverseDistance, SchemaGANInterpolator
from CPTSuperLearn.metric import MetricsTracker

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
    metrics = MetricsTracker(output_folder)

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
        metrics.update(episode, score, agent, cpt_env, action_index)

        if make_plots:
            cpt_env.plot_environment(os.path.join(output_folder, "training_2", f"episode_{episode}_file_{file_name}"))

        if episode % 10 == 0:
            print(f"Episode {episode} / {nb_episodes} | Average score: {average_score:.2f} Epsilon: {agent.epsilon:.2f}")

    # save the model, agent, environment and score
    cpt_env.save_environment(output_folder)
    agent.save_model(output_folder)
    metrics.visualize_metrics()
    metrics.save_metrics()


if __name__ == "__main__":
    training_data_folder = "data/train"
    num_episodes = 10000
    actions = [25, 50, 100, 150]  # More evenly spaced actions

    # Experiment with the improved reward function
    output_folder = "results2"

    cpt_env = CPTEnvironment(
        actions,
        max_nb_cpts=20,
        weight_reward_cpt=0.6,
        image_width=512,
        max_first_step=1,
        interpolation_method=SchemaGANInterpolator(model_path="./schemaGAN/schemaGAN.h5"),
    )

    cpt_agent = DQLAgent(
        state_size=6,
        action_size=len(actions),
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=0.95,
        epsilon_end=0.05,
        epsilon_decay=0.9996,
        memory_size=10000,
        batch_size=64,
        nb_steps_update=100,
        hidden_layers=[32, 64, 64, 32],
        use_batch_norm=True,
        activation='relu'
    )

    main(num_episodes, cpt_env, cpt_agent, training_data_folder, output_folder, make_plots=False)
