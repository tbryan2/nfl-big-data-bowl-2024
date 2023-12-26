from src.pso.pso import PSODefense
from src.pso.objective_functions import minimize_distance_to_ball_carrier_with_obstacle_avoidance
from src.pso.data_preprocessing import get_preprocessed_tracking_data
from config import TRACKING_DATA_URL, PLAYS_URL, PLAYERS_URL
import pandas as pd


def run_pso_pipeline(week_num, game_id):
    '''
    Loop through all the plays within a game and run the PSO algorithm on each play,
    calculate the Frechet distance between the best paths and the actual paths, and
    save the distances in a CSV with identifying information.
    '''

    df = pd.read_csv(TRACKING_DATA_URL.format(week=week_num))
    df = df.loc[df['gameId'] == game_id]

    if df.empty:
        raise ValueError(f"No tracking data found for game ID {game_id}")

    all_frechet_distances = []

    play_ids = df.playId.unique()
    if not play_ids.size:
        raise ValueError(f"No plays found for game ID {game_id}")

    for play_id in play_ids:
        data = get_preprocessed_tracking_data(week_num, game_id, play_id)

        if data is None:  # Skip the play if no handoff or pass caught
            print(f"Skipping play ID {play_id}: No handoff or pass outcome caught.")
            continue

        if not data or data['df'].empty:
            print(f"Warning: No data for play ID {play_id}")
            continue

        play_df = data['df']
        ball_carrier_id = data['ball_carrier_id']
        off_abbr = data['off_abbr']
        def_abbr = data['def_abbr']

        pso = PSODefense(
            play=play_df,
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
        )
        pso.optimize()
        pso.smooth_paths()
        frechet_distances_df = pso.calculate_frechet_distances()

        if not frechet_distances_df.empty:
            frechet_distances_df['play_id'] = play_id
            frechet_distances_df['game_id'] = game_id
            frechet_distances_df['week_num'] = week_num
            all_frechet_distances.append(frechet_distances_df)

    if not all_frechet_distances:
        raise ValueError("No valid Frechet distances calculated for any play.")

    # Concatenate all Frechet distances data
    concatenated_df = pd.concat(all_frechet_distances)

    # Read the complete plays and players data
    plays = pd.read_csv(PLAYS_URL)
    players = pd.read_csv(PLAYERS_URL)

    # Merge with player and play data
    concatenated_df = concatenated_df.merge(players[['nflId', 'displayName', 'position']], on='nflId', how='left')
    concatenated_df = concatenated_df.merge(plays[['gameId', 'playId', 'passResult', 'expectedPointsAdded']],
                                            left_on=['game_id', 'play_id'], right_on=['gameId', 'playId'], how='left')

    return concatenated_df


# Example usage of the function
result = run_pso_pipeline(1, 2022091102)
result.to_csv('run_pso_pipeline_test.csv')  # Save the result to a CSV file
