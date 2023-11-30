import numpy as np

# format logger with method name, line number, and message
def minimize_distance_to_ball_carrier(positions: np.ndarray, ball_carrier_position: np.ndarray) -> np.ndarray:
    """Objective function that minimizes the distance between the ball carrier and the defense

    Args:
        positions (np.ndarray): Array of positions of the defense
        ball_carrier_position (np.ndarray): Position of the ball carrier

    Returns:
        np.ndarray: Array of distances between the ball carrier and the defense
    """
    return np.linalg.norm(positions - ball_carrier_position, axis=1)
    