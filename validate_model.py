import os
import numpy as np
import argparse
import yaml
from tqdm import tqdm

from CPTSuperLearn.utils import read_data_file, write_score
from CPTSuperLearn.environment import CPTEnvironment
from CPTSuperLearn.agent import DQLAgent


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
    write_score(val_files, validation_scores,  os.path.join(output_folder, "validation_score.txt"))
    write_score(val_files, rmse_scores,  os.path.join(output_folder, "validation_rmse.txt"))


def main(config_path: str):
    """
    Main function to validate the CPTSuperLearn model.

    Parameters
    ----------
    :param config_path: Path to the yml configuration file.
    """
    # Load configuration
    config = load_config(config_path)

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
    evaluate_model(
        cpt_env,
        cpt_agent,
        data_config['validation_folder'],
        output_folder,
        make_plots=validation_config['make_plots']
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the CPTSuperLearn model.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file (required)")
    args = parser.parse_args()
    main(args.config)
