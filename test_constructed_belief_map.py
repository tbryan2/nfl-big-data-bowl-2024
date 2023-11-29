from src.pso.mpso import construct_belief_map
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

# Example usage
field_size = (120, 53.3)
num_grids = (50, 100)  # Specify the number of grids along the x and y axes
target_location_x = 60  # X coordinate of the target, in field units
target_location_y = 26.65  # Y coordinate of the target, in field units
sigma = 1.0

belief_map = construct_belief_map(field_size, num_grids, target_location_x, target_location_y, sigma)

plt.imshow(belief_map, cmap=cm.Reds)
fig = plt.gcf()
fig.set_size_inches(10, 5)
plt.show()