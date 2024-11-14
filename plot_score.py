import numpy as np
import matplotlib.pyplot as plt

from CPTSuperLearn.utils import moving_average


with open('./results/cpt_score.txt', 'r') as f:
    lines = f.read().splitlines()

data = np.array([line.split(";") for line in lines[1:]]).astype(float)


fig, ax = plt.subplots()
ax.plot(data[:, 0], data[:, 1], color='b')
ax.plot(data[:, 0], moving_average(data[:, 1], 10), color='r')
ax.set(xlabel='Episode', ylabel='Score')
ax.grid()
plt.show()
