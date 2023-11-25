import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde, multivariate_normal
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# Load data
df = pd.read_csv('example_play.csv')

# Function to apply color based on the club
def apply_coloring(x):
    if x == 'football':
        return 'orange'
    elif x == 'SF':
        return 'red'
    else:
        return 'green'

df['color'] = df['club'].apply(apply_coloring)

# Set bounds for the grid
xmin, xmax = df.x.min(), df.x.max()
ymin, ymax = df.y.min(), df.y.max()
xx, yy = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]

# Update function for animation
def update(frame_number):
    ax.clear()
    frame = df.loc[df['frameId'] == frame_number]
    
    offense = frame.loc[(frame['club'] == 'CHI') & (frame['nflId'] != 53646)]
    
    x, y = offense['x'].values, offense['y'].values
    data = np.vstack((x, y))
    kde = gaussian_kde(data)
    
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kde(positions).T, xx.shape)

    ball_carrier = frame.loc[(frame['club'] == 'CHI') & (frame['nflId'] == 53646)]
    ball_carrier_mean = ball_carrier[['x', 'y']].values[0]

    orientation = np.radians(ball_carrier['o'].values[0])

    # Standard deviations (long in the direction of movement, short perpendicular)
    long_std_dev = 2  # Example value, adjust as needed
    short_std_dev = 0.3  # Example value, adjust as needed

    # Initial covariance matrix (before rotation)
    covariance = np.diag([long_std_dev**2, short_std_dev**2])

    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(orientation), -np.sin(orientation)],
        [np.sin(orientation), np.cos(orientation)]
    ])

    # Rotated covariance matrix
    ball_carrier_covariance = rotation_matrix @ covariance @ rotation_matrix.T

    pos = np.dstack((xx, yy))
    rv = multivariate_normal(ball_carrier_mean, ball_carrier_covariance)
    g = rv.pdf(pos)

    combined_distribution = f + g*-0.2

    ax.imshow(np.rot90(combined_distribution), cmap="RdBu_r", extent=[xmin, xmax, ymin, ymax], aspect='auto')
    ax.scatter(frame['x'], frame['y'], c=frame['color'], edgecolors='purple')
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"Frame {frame_number}")

# Create initial plot
fig, ax = plt.subplots(figsize=(10, 6))

# Create animation
ani = FuncAnimation(fig, update, frames=df['frameId'].unique(), interval=1)

# Show the animation
plt.show()

# Optionally, save the animation
# ani.save('animation.mp4', writer='ffmpeg')
