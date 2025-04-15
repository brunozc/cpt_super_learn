import os
import numpy as np
from tqdm import tqdm

from CPTSuperLearn.utils import read_data_file, write_score
from CPTSuperLearn.environment import CPTEnvironment
from CPTSuperLearn.agent import DQLAgent
from CPTSuperLearn.interpolator import SchemaGANInterpolator


def evaluate_model(cpt_env: CPTEnvironment, agent: DQLAgent, validation_data_folder: str, output_folder: str,
                   nb_cpts: int, make_plots=False):
    """
    Evaluate the model on the validation dataset

    Parameters:
    -----------
    :param cpt_env: CPT environment
    :param agent: DQL agent
    :param validation_data_folder: folder with the validation data
    :param output_folder: output folder
    :param nb_cpts: number of CPTs for the uniform distribution
    :param make_plots: make plots
    """

    # Set the model to evaluation mode
    agent.qnetwork_local.eval()

    # Test on validation dataset
    validation_scores = []
    cpt_number = []
    rmse_scores = []

    val_files = os.listdir(validation_data_folder)[:10]

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
        cpt_number.append(len(cpt_env.sampled_positions))
        # combine RMSEs
        rmse_scores.append(";".join([str(rmse), str(rmse_2)]))

    val_files = [os.path.splitext(os.path.basename(f))[0] for f in val_files]
    write_score(val_files, validation_scores, cpt_number,  os.path.join(output_folder, "validation_score.txt"))
    write_score(val_files, rmse_scores, cpt_number,  os.path.join(output_folder, "validation_rmse.txt"))


# Example usage
if __name__ == "__main__":
    training_data_folder = "data/validation"

    results_path = "results_4"
    output_folder = "results_4/validation_1"

    cpt_env = CPTEnvironment.load_environment(results_path)
    if cpt_env.interpolator == "SchemaGANInterpolator" or cpt_env.interpolator is None:
        cpt_env.interpolator = SchemaGANInterpolator("./schemaGAN/schemaGAN.h5")
    cpt_agent = DQLAgent.load_model(results_path)

    nb_cpts = 4
    evaluate_model(cpt_env, cpt_agent, training_data_folder, output_folder, nb_cpts, make_plots=True)
