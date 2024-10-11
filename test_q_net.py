
from neural_RL import SimpleNeuralNetwork
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



# Load the pre-trained model (replace 'model.pth' with your model file)
model_path = "./results_schemaGAN_relu_explore_in_mot_b/target_10.pth"
model = SimpleNeuralNetwork.load_model(model_path)
#model.eval()  # Set the model to evaluation mode

# Load the validation dataset (replace with your data loading logic)
experiences = pickle.load(open("./results_schemaGAN_relu_explore_in_mot_b/experience_replay_10.pkl", "rb"))
experiences = np.array(experiences)
X_test = torch.tensor(experiences, dtype=torch.float32)
# Get the predictions
predictions = model(X_test).detach().numpy()
# get the index of the maximum value
y_pred = np.argmax(predictions, axis=1)
# histogram of the experiences
fig, ax = plt.subplots(1, 1)
# histogram of the experiences with the color of the prediction
ax.hist(experiences, bins=512, color=y_pred, alpha=0.7)
plt.title("Histogram of the experiences")
plt.show()

# create positioning heatmap for the experience replay first 1 column
fig, ax = plt.subplots(1, 1)
# experience signifies the position in a 1D grid group by the value of the experience to get the count
experience_location = Counter(experiences[0] * 512)
# get the position and the count
position, count = zip(*experience_location.items())
heatmap = np.zeros(512)
for i in range(512):
    if i not in position:
        heatmap[i] = 0
    else:
        heatmap[i] = count[position.index(i)]
# repeat 32 times the heatmap to get a 32x512 heatmap
heatmap = np.array([heatmap for _ in range(32)])
# create a heatmap
ax.imshow(heatmap, cmap='Blues')
ax.set_xlabel("X axis")
ax.set_ylabel("Depth")
# add colorbar
plt.colorbar(ax.imshow(heatmap, cmap='Blues'), ax=ax, orientation='horizontal')
# add title
plt.title("Most visited positions in the experience replay")

# to tensor


# get y values





