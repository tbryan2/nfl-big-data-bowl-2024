def penalty_function(distance_to_obstacle, max_penalty=100, safe_distance=1):
    """
    Calculate a penalty based on the distance to an obstacle.

    Args:
        distance_to_obstacle (float): The distance between the agent and the obstacle.
        max_penalty (float): The maximum penalty to be applied for zero distance. Default is 100.
        safe_distance (float): The distance at which no penalty is applied. Default is 1.

    Returns:
        float: The calculated penalty.
    """
    if distance_to_obstacle >= safe_distance:
        return 0
    else:
        # Inverse square law for penalty calculation
        return max_penalty / (distance_to_obstacle ** 2)