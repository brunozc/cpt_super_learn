import os
import random
import numpy as np
import torch
import yaml
import argparse

from CPTSuperLearn.utils import input_random_data_file
from CPTSuperLearn.environment import CPTEnvironment
from CPTSuperLearn.agent import DQLAgent
from CPTSuperLearn.interpolator import InverseDistance, SchemaGANInterpolator
from CPTSuperLearn.metric import MetricsTracker


def load_config(config_path):
    """
    Load configuration from YAML file

    Parameters
    ----------
    :param config_path: Path to the configuration file
    :return: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path: str):
    """
    Main function to train the CPTSuperLearn model.

    Parameters
    ----------
    :param config_path: Path to the yml configuration file.
    """

    # Load configuration
    config = load_config(config_path)

    # fix all the seeds
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    data_config = config['data']
    agent_config = config['agent']
    environment_config = config['environment']
    training_config = config['training']

    # Set up agent
    cpt_agent = DQLAgent(
        state_size=agent_config['state_size'],
        action_size=len(environment_config['actions']),
        learning_rate=agent_config['learning_rate'],
        gamma=agent_config['gamma'],
        epsilon_start=agent_config['epsilon_start'],
        epsilon_end=agent_config['epsilon_end'],
        epsilon_decay=agent_config['epsilon_decay'],
        memory_size=agent_config['memory_size'],
        batch_size=agent_config['batch_size'],
        nb_steps_update=agent_config['nb_steps_update'],
        hidden_layers=agent_config['hidden_layers'],
        use_batch_norm=agent_config['use_batch_norm'],
        activation=agent_config['activation']
    )

    # Set up interpolation method
    if environment_config['interpolation_method'] == 'SchemaGAN':
        interpolator = SchemaGANInterpolator(model_path=environment_config['interpolation_method_params'])
    elif environment_config['interpolation_method'] == 'InverseDistance':
        interpolator = InverseDistance(nb_points=int(environment_config['interpolation_method_params']))

    # Set up environment
    cpt_env = CPTEnvironment(
        action_list=environment_config['actions'],
        max_nb_cpts=environment_config['max_nb_cpts'],
        weight_reward_cpt=environment_config['weight_reward_cpt'],
        image_width=environment_config['image_width'],
        max_first_step=environment_config['max_first_step'],
        interpolation_method=interpolator,
    )

    train(data_config, training_config, cpt_env, cpt_agent)


def train(config_data: dict, config_training: dict, cpt_env: dict, agent: dict):
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

    output_folder = config_training['output_folder']
    nb_episodes = config_training['nb_episodes']
    training_data_folder = config_data['train_folder']

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

        if config_training["make_plots"]:
            cpt_env.plot_environment(os.path.join(output_folder, "training_2", f"episode_{episode}_file_{file_name}"))

        if episode % config_training['log_interval'] == 0:
            print(f"Episode {episode} / {nb_episodes} | Average score: {average_score:.2f} Epsilon: {agent.epsilon:.2f}")

    # save the model, agent, environment and score
    cpt_env.save_environment(output_folder)
    agent.save_model(output_folder)
    metrics.visualize_metrics()
    metrics.save_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CPTSuperLearn model.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file (required)")
    args = parser.parse_args()
    main(args.config)
