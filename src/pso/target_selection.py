import numpy as np

def select_common_target(particle_positions, candidate_targets, theta_max=30, w_d=1.0, w_theta=1.0):
    best_score = np.inf
    best_target = None
    best_target_index = -1  # Variable to store the index of the best target
    horizontal_axis = np.array([1, 0])  # Horizontal axis vector

    for index, target in enumerate(candidate_targets):
        total_score = 0

        for particle_position in particle_positions:
            distance = np.linalg.norm(target - particle_position)
            target_vector = target - particle_position
            
            # Calculate the angle using arctan2
            angle = np.arctan2(target_vector[1], target_vector[0])

            # Check if the angle is within the allowed range
            if angle < -theta_max or angle > theta_max:
                # If the angle is not within the allowed range, give a high penalty score
                # total_score += 1e6  # A large number to represent an infeasible option
                pass
            else:
                score = w_d * distance + w_theta * abs(angle)  # Use absolute value of angle for scoring
                total_score += score

        if total_score < best_score:
            best_score = total_score
            best_target = target
            best_target_index = index  # Update the index of the best target

    return best_target, best_target_index
