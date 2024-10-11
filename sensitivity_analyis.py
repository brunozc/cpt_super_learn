from neural_RL import SimpleNeuralNetwork
from SALib.sample import saltelli
from SALib.analyze import sobol

import torch
from sklearn.metrics import accuracy_score
from collections import Counter
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
import shap


if __name__ == "__main__":
    model_path = "./results_schemaGAN_actions_restricted_3/deep_100.pth"
    model = SimpleNeuralNetwork.load_model(model_path)
    # define the problem values
    problem = {
        'num_vars': 4,
        'names': ['Position', 'Mean_division', 'Std_division', 'Nb_layers_div'],
        'bounds': [[0, 1], [0, 1], [0, 1], [0, 1]]
    }
    # generate the samples
    param_values = saltelli.sample(problem, 100)
    # get the predictions
    predictions = []
    for i in range(param_values.shape[0]):
        predictions.append(model(torch.Tensor(param_values[i])).detach().numpy())
    # perform the sobol analysis per output value
    results = []
    for i in range(len(predictions[0])):
        Si = sobol.analyze(problem, np.array(predictions)[:, i], print_to_console=True)
        results.append(Si)
    fig, ax = plt.subplots(1, len(predictions[0]), figsize=(12, 5))
    for i in range(len(predictions[0])):
        ax[i].bar(problem['names'], results[i]['S1'], yerr=results[i]['S1_conf'], capsize=5)
        # set min and max values
        ax[i].set_ylim([0, 1])
        ax[i].set_title(f"Output {i}")
        ax[i].set_ylabel("Sobol index")
        ax[i].set_xlabel("Parameters")
    plt.tight_layout()
    plt.show()




