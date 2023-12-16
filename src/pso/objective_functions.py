import numpy as np

# format logger with method name, line number, and message
def minimize_distance_to_ball_carrier(
        particle_position: np.ndarray, # positions[i]
        target_positions: np.ndarray,
        time_weighting_factor: int = 3
    ) -> np.ndarray:

    cumulative_distance = 0
    total_frames = len(target_positions)
    for frame_index, ball_carrier_pos in enumerate(target_positions):
        # Increasing weight for later frames (e.g., linearly increasing)
        time_weight = (frame_index**time_weighting_factor + 1) / total_frames
        cumulative_distance += time_weight * np.linalg.norm(particle_position - ball_carrier_pos)

    return cumulative_distance    

def minimize_distance_to_ball_carrier_with_obstacle_avoidance(
        particle_position: np.ndarray,
        particle_velocity: np.ndarray,
        target_positions: np.ndarray,
        obstacle_positions: np.ndarray,
        time_weighting_factor: int=3,
        obstacle_avoidance_factor: float=1.0
    ) -> float:

    target_distances = []
    obstacle_weights = []
    total_frames = len(target_positions)

    # Calculate distances to target and obstacles for each frame
    for frame_index in range(total_frames):
        # Distance to target
        target_distance = np.linalg.norm(particle_position - target_positions[frame_index])
        target_distances.append(target_distance)

        # Weight for each obstacle
        future_particle_position = particle_position + particle_velocity * (frame_index + 1)

        current_frame_obstacle_weights = []
        for obstacle in obstacle_positions[frame_index]:
            obstacle_distance = np.linalg.norm(future_particle_position - obstacle)
            obstacle_weight = np.exp(-obstacle_distance * obstacle_avoidance_factor)
            current_frame_obstacle_weights = np.append(current_frame_obstacle_weights, obstacle_weight).reshape(1, -1)

        obstacle_weights.append(np.array(current_frame_obstacle_weights))
    
    target_distances = np.array(target_distances).reshape(-1, 1)
    obstacle_weights = np.array(obstacle_weights).reshape(total_frames, -1)

    # normalize the target distances
    normalized_target_distances = target_distances / np.max(target_distances)
    # normalize the obstacle weights
    normalized_obstacle_weights = obstacle_weights / np.max(obstacle_weights)

    log_weighting = (np.log(np.flip(np.arange(1, total_frames + 1))) * time_weighting_factor).reshape(-1, 1)

    # Apply the log weighting to the normalized obstacle weights using element-wise multiplication
    weighted_obstacle_weights = normalized_obstacle_weights * log_weighting
    penalty = np.sum(weighted_obstacle_weights)

    exp_weighting = np.power(np.arange(1, total_frames + 1), time_weighting_factor+1).reshape(-1, 1) / total_frames
    weighted_target_importance = normalized_target_distances * exp_weighting
    reward = np.sum(weighted_target_importance)
    print('Reward: ', reward, 'Penalty: ', penalty, 'Total: ', reward - penalty)
    return reward - penalty