import os
import json
import numpy as np
import matplotlib.pylab as plt

from smooth import smooth


index = 100
output_folder = "./results_schemaGAN_actions_restricted_2"

with open(os.path.join(output_folder, f"summary_{index}.json"), "r") as f:
    summary = json.load(f)

fig, ax = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
ax[0].set_position([0.15, 0.55, 0.8, 0.35])
ax[1].set_position([0.15, 0.12, 0.8, 0.35])
ax[0].plot(summary["score"])
ax[0].plot(smooth(np.array(summary["score"]), 50), label="smoothed")
ax[0].set_ylabel('Loss')
ax[0].grid()
ax[0].legend()
ax[1].plot(summary["rmse"])
ax[1].plot(smooth(np.array(summary["rmse"]), 50), label="smoothed")
ax[1].set(xlabel='Episodes', ylabel='RMSE')
ax[1].set_ylim(bottom=0)
ax[1].set_xlim(left=0)
ax[1].grid()
ax[1].legend()
plt.show()
