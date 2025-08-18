import os
import numpy as np
import matplotlib.pylab as plt

path_to_results =  "results"

with open(os.path.join(path_to_results, "validation/validation_metrics.txt"), "r") as fi:
    data = fi.read().splitlines()

data = [dat.split(";") for dat in data[1:]]

data = np.array(data)

class1_rmse = data[:, 1].astype(float)
class2_rmse = data[:, 2].astype(float)
class1_bias = data[:, 3].astype(float)
class2_bias = data[:, 4].astype(float)
class1_pia = data[:, 5].astype(float)
class2_pia = data[:, 6].astype(float)

# Calculate mean and standard deviation for each class
class1_mean = np.mean(class1_rmse)
class1_std = np.std(class1_rmse)
class2_mean = np.mean(class2_rmse)
class2_std = np.std(class2_rmse)

# Create a figure and axis
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
ax[0].set_position([0.04, 0.1, 0.28, 0.8])
ax[1].set_position([0.37, 0.1, 0.28, 0.8])
ax[2].set_position([0.70, 0.1, 0.28, 0.8])

ax[0].scatter(np.ones(len(class1_rmse)), class1_rmse, marker="x", color='b', s=3)
ax[0].scatter(np.ones(len(class2_rmse))+1, class2_rmse, marker="x", color='b', s=4)
violin_parts = ax[0].violinplot([class1_rmse, class2_rmse], showmeans=False, showextrema=False)
color=['lightblue', 'lightcoral']
for i, body in enumerate(violin_parts['bodies']):
    body.set_facecolor(color[i])
    # body.set_edgecolor('black')
    body.set_alpha(0.7)

ax[0].errorbar([1, 2], [class1_mean, class2_mean],
            yerr=[class1_std, class2_std],
            fmt='o', color="b", markersize=6, linewidth=3, capsize=5, label=['DRL', 'Standard'])

ax[1].bar([1, 2], [np.mean(class1_bias), np.mean(class2_bias)], #yerr=[np.std(class1_bias), np.std(class2_bias)],
          color=['lightblue', 'lightcoral'], alpha=0.7, capsize=5, label=['DRL', 'Standard'])

ax[1].errorbar([1, 2], [np.mean(class1_bias), np.mean(class2_bias)], yerr=[np.std(class1_bias), np.std(class2_bias)],
            fmt='o', color="b", markersize=6, linewidth=3, capsize=5, label=['DRL', 'Standard'])

ax[2].bar([1, 2], [np.mean(class1_pia), np.mean(class2_pia)], #yerr=[np.std(class1_pia), np.std(class2_pia)],
          color=['lightblue', 'lightcoral'], alpha=0.7, capsize=5, label=['DRL', 'Standard'])

ax[2].errorbar([1, 2], [np.mean(class1_pia), np.mean(class2_pia)], yerr=[np.std(class1_pia), np.std(class2_pia)],
            fmt='o', color="b", markersize=6, linewidth=3, capsize=5, label=['DRL', 'Standard'])

# Set labels and title
ax[0].set_xticks([1, 2])
ax[0].set_xticklabels(['DRL', 'Standard'], fontsize=12)
ax[0].set_ylabel('RMSE', fontsize=12)
ax[0].set_ylim(bottom=0)
ax[0].grid()

ax[1].set_xticks([1, 2])
ax[1].set_xticklabels(['DRL', 'Standard'], fontsize=12)
ax[1].set_ylabel('Bias', fontsize=12)
ax[1].set_ylim(0, 0.25)
ax[1].grid()

ax[2].set_xticks([1, 2])
ax[2].set_xticklabels(['DRL', 'Standard'], fontsize=12)
ax[2].set_ylabel('PIA', fontsize=12)
ax[2].set_ylim(0, 1)
ax[2].grid()

# Show the plot
plt.savefig(os.path.join(path_to_results, "metrics_comparison.png"))
plt.savefig(os.path.join(path_to_results, "metrics_comparison.pdf"))
plt.close()


print(f"Class 1 RMSE: {class1_mean:.4f} ± {class1_std:.4f}")
print(f"Class 2 RMSE: {class2_mean:.4f} ± {class2_std:.4f}")
print(f"Class 1 Bias: {np.mean(class1_bias):.4f} ± {np.std(class1_bias):.4f}")
print(f"Class 2 Bias: {np.mean(class2_bias):.4f} ± {np.std(class2_bias):.4f}")
print(f"Class 1 PIA: {np.mean(class1_pia):.4f} ± {np.std(class1_pia):.4f}")
print(f"Class 2 PIA: {np.mean(class2_pia):.4f} ± {np.std(class2_pia):.4f}")
