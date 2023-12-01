from src.pso.pso import PSODefense
from src.pso.objective_functions import minimize_distance_to_ball_carrier
from src.pso.data_preprocessing import get_preprocessed_tracking_data
import pandas as pd

data = get_preprocessed_tracking_data(
    week_num=1, 
    game_id=2022091100, 
    play_id=870
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
    move_particles=[1],
    w=1,
    c1=0.3,
    c2=2,
    num_iterations=10_000,
    min_velocity=-0.3,
    max_velocity=0.3
)

pso.optimize_play()
pso.animate_play(save_fig=True)