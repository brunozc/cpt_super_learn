import matplotlib.pyplot as plt

with open('./results/cpt_score.txt', 'r') as f:
    lines = f.read().splitlines()

data = [line.split(";") for line in lines[1:]]

fig, ax = plt.subplots()
ax.plot([int(x[0]) for x in data], [float(x[1]) for x in data])
ax.set(xlabel='Episode', ylabel='Score')
ax.grid()
plt.show()

