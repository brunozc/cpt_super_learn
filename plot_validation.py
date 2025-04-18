import os
import numpy as np
import matplotlib.pylab as plt

path_to_results =  "results"

with open(os.path.join(path_to_results, "validation/validation_rmse.txt"), "r") as fi:
    data = fi.read().splitlines()

data = [dat.split(";") for dat in data]

data = np.array(data)

class1 = data[:, 1].astype(float)
class2 = data[:, 2].astype(float)

# Calculate mean and standard deviation for each class
class1_mean = np.mean(class1)
class1_std = np.std(class1)
class2_mean = np.mean(class2)
class2_std = np.std(class2)

# Create a figure and axis
fig, ax = plt.subplots()

ax.scatter(np.ones(len(class1)), class1, marker="x", color='b', s=3)
ax.scatter(np.ones(len(class2))+1, class2, marker="x", color='b', s=4)
ax.violinplot([class1, class2], showmeans=False, showextrema=False)
ax.errorbar([1, 2], [class1_mean, class2_mean],
            yerr=[class1_std, class2_std],
            fmt='o', color="r", markersize=6, linewidth=3, capsize=5, label=['DRL', 'Standard'])

# Set labels and title
ax.set_xticks([1, 2])
ax.set_xticklabels(['DRL', 'Standard'])
ax.set_ylabel('RMSE')
ax.set_ylim(bottom=0)
ax.grid()
# Show the plot
plt.savefig(os.path.join(path_to_results, "rmse_comparison.png"))
plt.close()



