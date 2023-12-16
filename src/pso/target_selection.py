import numpy as np

def select_common_target(particle_positions, candidate_targets, theta_max, w_d=1.0, w_theta=1.0):
    best_score = np.inf
    best_target = None

    for target in candidate_targets:
        total_score = 0

        for particle_position in particle_positions:
            distance = np.linalg.norm(target - particle_position)
            target_vector = target - particle_position
            
            # Calculate the angle using arctan2
            angle = np.arctan2(target_vector[1], target_vector[0])

            angle = abs(angle) - np.pi / 2

            # print(
            #     f'Particle position: {particle_position}, Target: {target}, Distance: {distance}, Angle: {np.degrees(angle)}'
            # )

            # Check if the angle is within the allowed range
            if angle < theta_max:
                # If the angle is not within the allowed range, give a high penalty score
                # total_score += 1e6  # A large number to represent an infeasible option
                pass
            else:
                score = w_d * distance + w_theta * abs(angle)  # Use absolute value of angle for scoring
                total_score += score

        if total_score < best_score:
            best_score = total_score
            best_target = target

    return best_target