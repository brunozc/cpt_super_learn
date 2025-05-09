import os
import random
import numpy as np
import torch
from tqdm import tqdm

from CPTSuperLearn.utils import input_random_data_file, read_data_file, write_score, load_yaml_config
from CPTSuperLearn.environment import CPTEnvironment
from CPTSuperLearn.agent import DQLAgent
from CPTSuperLearn.interpolator import InverseDistance, SchemaGANInterpolator
from CPTSuperLearn.metric import MetricsTracker


def train_model(config_path: str):
    """
    Main function to train the CPTSuperLearn model.

    Parameters
    ----------
    :param config_path: Path to the yml configuration file.
    """
    # Load configuration
    config = load_yaml_config(config_path)

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

    # Train the model
    __train(data_config, training_config, cpt_env, cpt_agent)


def __train(config_data: dict, config_training: dict, cpt_env: CPTEnvironment, agent: DQLAgent):
    """
    Train the DRL model

    Parameters
    ----------
    :param config_data: Data configuration
    :param config_training: Training configuration
    :param cpt_env: CPT environment
    :param agent: DQL agent
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
            cpt_env.plot_environment(os.path.join(output_folder, "training", f"episode_{episode}_file_{file_name}"))

        if episode % config_training['log_interval'] == 0:
            print(f"Episode {episode} / {nb_episodes} | Average score: {average_score:.2f} Epsilon: {agent.epsilon:.2f}")

    # save the model, agent, environment and score
    cpt_env.save_environment(output_folder)
    agent.save_model(output_folder)
    metrics.visualize_metrics()
    metrics.save_metrics()


def validate_model(config_path: str):
    """
    Main function to validate the CPTSuperLearn model.

    Parameters
    ----------
    :param config_path: Path to the yml configuration file.
    """
    # Load configuration
    config = load_yaml_config(config_path)

    # validation settings
    data_config = config['data']
    validation_config = config['validation']
    results_path = config['training']['output_folder']

    # Create validation output folder
    output_folder = validation_config['output_folder']
    os.makedirs(output_folder, exist_ok=True)

    # Load environment and agent from trained model
    cpt_env = CPTEnvironment.load_environment(results_path)
    cpt_agent = DQLAgent.load_model(results_path)

    # Run validation
    __evaluate_model(
        cpt_env,
        cpt_agent,
        data_config['validation_folder'],
        output_folder,
        make_plots=validation_config['make_plots']
    )


def __evaluate_model(cpt_env: CPTEnvironment, agent: DQLAgent, validation_data_folder: str, output_folder: str,
                     make_plots=False):
    """
    Evaluate the model on the validation dataset

    Parameters:
    -----------
    :param cpt_env: CPT environment
    :param agent: DQL agent
    :param validation_data_folder: folder with the validation data
    :param output_folder: output folder
    :param make_plots: make plots (default: False)
    """
    # Set the model to evaluation mode
    agent.qnetwork_local.eval()

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

        # compute the RMSE and make plot for the DRL model
        validation_scores.append(score)
        rmse = np.sqrt(np.mean((cpt_env.true_data - cpt_env.predicted_data) ** 2))
        if make_plots:
            cpt_env.plot_environment(os.path.join(output_folder, "images", f"file_{file_name}"))

        # compute the RMSE and make plot for the uniform distribution
        # the number of CPT remains the same
        nb_cpts = len(cpt_env.sampled_positions)
        idx_cpts = np.linspace(0, cpt_env.image_width - 1, nb_cpts, dtype=int)

        cpts = [image_data[image_data[:, 0] == i, 2] for i in idx_cpts]
        interpolator = cpt_env.interpolator
        interpolator.interpolate(idx_cpts, np.array(cpts))
        interpolator.predict(np.arange(cpt_env.image_width))
        rmse_2 = np.sqrt(np.mean((cpt_env.true_data - interpolator.prediction) ** 2))
        if make_plots:
            cpt_env.sampled_positions = idx_cpts
            cpt_env.sampled_values = cpts
            cpt_env.predicted_data = interpolator.prediction
            cpt_env.plot_environment(os.path.join(output_folder, "images", f"file_{file_name}_uniform"))

        # combine RMSEs
        rmse_scores.append(";".join([str(rmse), str(rmse_2)]))

    val_files = [os.path.splitext(os.path.basename(f))[0] for f in val_files]
    write_score(val_files, validation_scores, os.path.join(output_folder, "validation_score.txt"))
    write_score(val_files, rmse_scores, os.path.join(output_folder, "validation_rmse.txt"))
