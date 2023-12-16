from src.pso.pso import PSODefense
from src.pso.objective_functions import (
    minimize_distance_to_ball_carrier_with_obstacle_avoidance
)
from src.pso.data_preprocessing import get_preprocessed_tracking_data

data = get_preprocessed_tracking_data(
    week_num=1, 
    game_id=2022091102, 
    play_id=2065
)

df = data['df']
ball_carrier_id = data['ball_carrier_id']
off_abbr = data['off_abbr']
def_abbr = data['def_abbr']

pso = PSODefense(
    play=df, 
    objective_function=minimize_distance_to_ball_carrier_with_obstacle_avoidance, 
    def_abbr=def_abbr, 
    off_abbr=off_abbr, 
    ball_carrier_id=ball_carrier_id,
    positional_group='secondary',
    w=1,
    c1=1,
    c2=2,
    num_iterations=1_000,
    time_weighting_factor=3,
    obstacle_avoidance_factor=1.0
)
pso.optimize()
pso.animate_play()