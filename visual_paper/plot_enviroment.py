from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Recreate the image with modifications: flip the rectangle and place the trucks outside with wheels touching the main rectangle

# Create a new figure and axis
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_xlim(0, 17)
ax.set_ylim(0, 20)
ax.axis('off')

# Draw the main flipped rectangular box
main_rect = patches.Rectangle((2, 4), 12, 6, linewidth=1, edgecolor='black', facecolor='none')
ax.add_patch(main_rect)

# Draw the two small rectangles (vehicles) outside the main rectangle, with wheels just touching
vehicle_1 = patches.Rectangle((3.8, 10.5), 2.5, 1.5, linewidth=1, edgecolor='black', facecolor='none')
ax.add_patch(vehicle_1)
vehicle_2 = patches.Rectangle((9.8, 10.5), 2.5, 1.5, linewidth=1, edgecolor='black', facecolor='none')
ax.add_patch(vehicle_2)

# Draw the circles to the right of each vehicle (representing wheels), touching the main rectangle's left side
circle_1a = patches.Circle((4, 10.25), 0.25, linewidth=1, edgecolor='black', facecolor='none')
circle_1b = patches.Circle((6, 10.25), 0.25, linewidth=1, edgecolor='black', facecolor='none')
circle_2a = patches.Circle((10, 10.25), 0.25, linewidth=1, edgecolor='black', facecolor='none')
circle_2b = patches.Circle((12, 10.25), 0.25, linewidth=1, edgecolor='black', facecolor='none')
ax.add_patch(circle_1a)
ax.add_patch(circle_1b)
ax.add_patch(circle_2a)
ax.add_patch(circle_2b)

# Draw zigzag lines inside the main rectangle near the right side
y_vals = np.linspace(5, 10, 100)
x_vals_1 = 5 + 0.25 * np.sin(8 * np.pi * y_vals)
x_vals_2 = 10.9 + 0.25 * np.sin(8 * np.pi * y_vals)
# add some noise to the x_vals
x_vals_1 += np.random.normal(0, 0.05, x_vals_1.shape)
x_vals_2 += np.random.normal(0, 0.05, x_vals_2.shape)
ax.plot(x_vals_1, y_vals, color='black')
ax.plot(x_vals_2, y_vals, color='black')


# Draw action arrow with annotation
ax.annotate('', xy=(4.5, 13), xytext=(11, 13), arrowprops=dict(arrowstyle='<->', lw=1))
ax.text(6, 13, 'Action =\nnext cpt in x meters', va='center', fontsize=10)

plt.show()
