from src.pso.pso import PSODefense
from src.pso.objective_functions import minimize_distance_to_ball_carrier
from src.pso.data_preprocessing import get_preprocessed_tracking_data
from config import TRACKING_DATA_URL
import pandas as pd


# Step 1: Loop through all the plays within a game and prepreprocess the data
# This will return a dictionary consisting of the following:
# week number, game id, play id, ball carrier id, offensive team abbreviation, 
# defensive team abbreviation, and the play dataframe {week}_{game_id}_{play_id}.csv
# 
# Step 2: For each play, we'd like to run the PSO algorithm
# First we need to identify the agents - to start, we'll just select any players that are S, FS, or SS
# in other words, just the safeties (verify that there's no other safety abbreviations)
# From this, we need to return the best paths of the agents as well as the actual paths of the players
#
# Step 3: Calculate the Frechet distance between the best paths and the actual paths
# Step 4: Save these the distances in a CSV with identifying information
# (game id, play id, week number, nflIds, etc.)

def run_pso_pipeline(week_num, game_id):
    '''
    Looop through all the plays within a game and run the PSO algorithm on each play,
    calculate the Frechet distance between the best paths and the actual paths, and
    save the distances in a CSV with identifying information.
    '''

    df = pd.read_csv(TRACKING_DATA_URL.format(week=week_num))

    df = df.loc[df['gameId'] == game_id]

    for play_id in df.playId.unique():
        data = get_preprocessed_tracking_data(
            week_num=week_num, 
            game_id=game_id, 
            play_id=play_id
        )

        df = data['df']
        ball_carrier_id = data['ball_carrier_id']
        off_abbr = data['off_abbr']
        def_abbr = data['def_abbr']

        pso = PSODefense(
            play=df, 
            objective_function=minimize_distance_to_ball_carrier, 
            def_abbr=def_abbr, 
            off_abbr=off_abbr, 
            ball_carrier_id=ball_carrier_id,
            # HAVE TO FIND A WAY TO PROGRAMMATICALLY SELECT THE AGENTS HERE
            agents=[44925, 54513],
            w=1,
            c1=1,
            c2=2,
            num_iterations=1000, 
            min_velocity=-0.6,
            max_velocity=0.6,
            time_weighting_factor=100
        )

        pso.optimize() # HAVE TO FIND A WAY TO HAVE THIS RETURN THE BEST PATHS AND ACTUAL PATHS, AND THE FRECHET DISTANCE
        # HAVE TO FIND A WAY TO SAVE THE DATA
        