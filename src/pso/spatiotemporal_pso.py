import numpy as np
import pandas as pd

# PSO algorithim for optimizing the route a defender takes to intercept a
# ball carrier while avoiding other offensive players. The dimensions of
# the search space are x, y, and t.

# Build objective function
# Objective function takes a 3D array of x, y, and t values and returns
# a 2D array of the same shape with the objective function value at each
# point in the search space.

def spatiotemporal_objective_function(defender_position: np.ndarray,
                                ball_carrier_position: np.ndarray,
                                defender_velocity: np.ndarray,
                                obstacle_positions: np.ndarray,
                                penalty_factor: float) -> float:
    """
    Objective function that minimizes the time-adjusted 
    distance to the ball carrier while avoiding obstacles.
    """
    # Calculate defender speed
    defender_speed = np.linalg.norm(defender_velocity)

    # Time-adjusted distance to the ball carrier
    distance_to_ball_carrier = np.linalg.norm(defender_position - ball_carrier_position)
    time_to_reach = distance_to_ball_carrier / defender_speed if defender_speed != 0 else np.inf

    # Penalty for being close to obstacles
    penalty = np.sum(
                    np.exp(-np.linalg.norm(defender_position - obstacle_positions, axis=1)
                           * penalty_factor)
                    )

    # Combine time-adjusted distance and penalty
    return time_to_reach + penalty

# Testing the objective function
df = pd.read_csv("data/example_play.csv")

actual_ball_carrier_position = np.array(
                                        [df])