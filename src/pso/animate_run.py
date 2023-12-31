import pandas as pd
from src.pso.pso import PSODefense
from src.pso.objective_functions import (
    minimize_distance_to_ball_carrier_with_obstacle_avoidance
)
from src.pso.data_preprocessing import get_preprocessed_tracking_data


specific_plays = [
    {'week_num': 1, 'game_id': 2022091103, 'play_id': 1126},  # Highest Frechet for is_screen_pass
    {'week_num': 1, 'game_id': 2022091104, 'play_id': 580},   # Lowest Frechet for is_screen_pass
    {'week_num': 1, 'game_id': 2022091111, 'play_id': 1946},  # Highest Frechet for is_contested_ball
    {'week_num': 1, 'game_id': 2022091200, 'play_id': 156},   # Lowest Frechet for is_contested_ball
    {'week_num': 1, 'game_id': 2022091103, 'play_id': 1126},  # Highest Frechet for is_play_action
    {'week_num': 1, 'game_id': 2022091103, 'play_id': 1249},  # Lowest Frechet for is_play_action
    {'week_num': 1, 'game_id': 2022091101, 'play_id': 1744},  # Top YAC Diff
    {'week_num': 1, 'game_id': 2022091100, 'play_id': 3554}   # Lowest YAC Diff
]


# Step 2: Process each play individually
def process_specific_plays(specific_plays):
    all_paths = []

    for play in specific_plays:
        week_num = play['week_num']
        game_id = play['game_id']
        play_id = play['play_id']

        # Step 3: Call the PSO optimization for the specified play
        # Assuming get_preprocessed_tracking_data, PSODefense, and related functions are defined
        data = get_preprocessed_tracking_data(week_num, game_id, play_id)

        df = data['df']
        ball_carrier_id = data['ball_carrier_id']
        off_abbr = data['off_abbr']
        def_abbr = data['def_abbr']

        if data is not None:
            pso = PSODefense(
                play=df,
                objective_function=minimize_distance_to_ball_carrier_with_obstacle_avoidance,
                def_abbr=def_abbr,
                off_abbr=off_abbr,
                ball_carrier_id=ball_carrier_id,
                positional_group='secondary',
                w=.729,
                c1=1.49445,
                c2=1.49445,
                max_iterations=1_000,
                time_weighting_factor=3,
                obstacle_avoidance_factor=1.0,
                stop_threshold=0.0001

            )  # Initialize PSODefense with appropriate arguments
            pso.optimize()
            pso.smooth_paths()
            frechet_distances_df, paths_df = pso.calculate_frechet_distances()

            # Add the game and play IDs to the paths DataFrame
            paths_df['gameId'] = game_id
            paths_df['playId'] = play_id

            all_paths.append(paths_df)

    # Step 4: Concatenate all the path DataFrames and export to a CSV file
    final_paths_df = pd.concat(all_paths)
    final_paths_df.to_csv('/Users/nick/nfl-big-data-bowl-2024/data/specific_plays_paths.csv', index=False)

    print("All specified plays have been processed. Paths data saved to 'specific_plays_paths.csv'.")

# Run the script for the specific plays
process_specific_plays(specific_plays)
