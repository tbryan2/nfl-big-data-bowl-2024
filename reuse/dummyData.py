import numpy as np
import pandas as pd

# Grid dimensions
x_length = 120.0
y_length = 53.3

# Number of frames
num_frames = 50

# Adjusting the resolution to the 1s place
x_resolution = 1.0
y_resolution = 1.0

# Recreating x and y coordinates with the new resolution
x_coords = np.arange(0, x_length + x_resolution, x_resolution)
y_coords = np.arange(0, y_length + y_resolution, y_resolution)

# Recreating the dataset with the adjusted resolution
dataset = []
for frame in range(num_frames):
    for x in x_coords:
        for y in y_coords:
            probability = np.random.uniform(0, 100)  # Random probability for each cell
            dataset.append(
                {'frameId': frame,
                 'x': round(x, 2), 
                 'y': round(y, 2), 
                 'SWARM': round(probability, 2)})

# Convert to DataFrame for better visualization
df = pd.DataFrame(dataset)

df.to_csv('data/dummyData.csv', index=False)
