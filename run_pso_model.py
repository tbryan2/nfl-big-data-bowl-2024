from src.pso.pso import PSODefense
from src.pso.objective_functions import minimize_distance_to_ball_carrier
from src.pso.data_preprocessing import get_preprocessed_tracking_data
import pandas as pd
import numpy as np

data = get_preprocessed_tracking_data(
    week_num=1, 
    game_id=2022091102, 
    play_id=921
)

df = data['df']
ball_carrier_id = data['ball_carrier_id']
off_abbr = data['off_abbr']
def_abbr = data['def_abbr']
ball_arrived_frame = data['df']['ball_arrived'].iloc[0]
pso = PSODefense(
    play=df,
    objective_function=minimize_distance_to_ball_carrier,
    def_abbr=def_abbr,
    off_abbr=off_abbr,
    ball_carrier_id=ball_carrier_id,
    ball_arrived_frame=ball_arrived_frame,
    agents=[44925, 54513],
    w=1,
    c1=1,
    c2=2,
    num_iterations=1000,
    min_velocity=-0.6,
    max_velocity=0.6,
    time_weighting_factor=100
)

pso.optimize()
pso.animate_play()