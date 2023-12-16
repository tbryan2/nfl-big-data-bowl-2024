import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from typing import Callable
import logging
from matplotlib.animation import FuncAnimation
import random
from src.pso.target_selection import select_common_target

class PSODefense:
    def __init__(
        self, 
        play: pd.DataFrame, 
        objective_function: Callable,
        def_abbr: str, 
        off_abbr: str,
        ball_carrier_id: int,
        positional_group: str, # options: 'safties' = 'FS', 'SS', 'linebackers' = 'MLB', 'OLB', 'ILB', 'cornerbacks' = 'CB', 'secondary' = 'FS', 'SS', 'CB'
        w: float = 0.1,
        c1: float = 2,
        c2: float = 0.2,
        num_iterations: int = 10_000,
        time_weighting_factor: int = 3,
        obstacle_avoidance_factor: float = 1.0
    ):

        # DataFrame of the play
        self.play = play.sort_values(by=['frameId', 'nflId']) 

        # metadata
        self.num_frames = len(play.frameId.unique())
        self.def_abbr = def_abbr
        self.off_abbr = off_abbr
        self.ball_carrier_id = int(ball_carrier_id)

        # define the ball carrier velocity
        self.ball_carrier_velocity = self.play.loc[self.play['nflId'] == self.ball_carrier_id][['xy_velocity']].values
        
        # positional group
        if positional_group == 'safeties':
            self.positional_group = ['FS', 'SS']
        elif positional_group == 'linebackers':
            self.positional_group = ['MLB', 'OLB', 'ILB']
        elif positional_group == 'cornerbacks':
            self.positional_group = ['CB']
        elif positional_group == 'secondary':
            self.positional_group = ['FS', 'SS', 'CB']

        # agents
        self.agents = self.play.loc[self.play['position'].isin(self.positional_group)]['nflId'].unique()

        # constraints
        self.xmin = 0
        self.xmax = 120  # Including endzones
        self.ymax = 53.3  # Standard width of a football field
        self.ymin = 0

        self.num_particles = len(self.agents)
        self.num_dimensions = 2

        self.time_weighting_factor = time_weighting_factor
        self.obstacle_avoidance_factor = obstacle_avoidance_factor

        # this represents the actual x and y positions of the particles
        # not the positions that will be updated by method self.optimize_frame
        # group by frame and then convert 
        particle_frames = self.play.loc[self.play.nflId.isin(self.agents)].groupby('frameId')
        self.actual_particle_positions = np.array([frame[['x', 'y']].to_numpy() for _, frame in particle_frames])
        self.actual_velocities = np.array([frame[['x_velocity', 'y_velocity']].to_numpy() for _, frame in particle_frames])

        self.actual_obstacle_positions = np.array([self.play.loc[(self.play.frameId == frame_id) & \
            (self.play.club == self.off_abbr)][['x', 'y']].to_numpy() for frame_id in self.play.frameId.unique()])

        ball_carrier_frames = self.play.loc[self.play['nflId'] == self.ball_carrier_id].groupby('frameId')
        self.target_positions = np.array([frame.loc[frame['nflId'] == self.ball_carrier_id][['x', 'y']].values[0] for _, frame in ball_carrier_frames])

        # calculate the target location
        self.best_target, self.best_target_idx = select_common_target(self.actual_particle_positions[0], self.target_positions, w_theta=1)
        # mask for the ball carrier positions
        self.best_targets = np.full((self.num_frames, 2), np.nan)
        self.best_targets[self.best_target_idx] = self.best_target

        # objective function and param setting
        self.objective_function = objective_function
        self.objective_function_params = {
            'obstacle_positions': self.actual_obstacle_positions,
            'target_positions': self.best_targets,
            'time_weighting_factor': self.time_weighting_factor,
            'obstacle_avoidance_factor': self.obstacle_avoidance_factor
        }

        # initialize the positions and velocities of the particles
        # these are the positions and particles that will be updated by method self.optimize_frame
        self.positions = self.play.loc[(self.play.nflId.isin(self.agents)) & (self.play['frameId'] == 1)][['x', 'y']].values
        
        self.velocities = np.zeros((self.num_particles, self.num_dimensions))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(self.num_particles, np.inf)
        self.global_best_score = np.inf

        # note: not sure why np.nan did not work here, but np.zeros did
        # check back later
        self.global_best_position = np.zeros(self.num_dimensions)

        # history of the positions optimized by PSO
        self.positions_history = np.array([self.positions.copy()])
        self.velocities_history = np.array([self.velocities.copy()])
        self.personal_best_positions_history = np.array([self.personal_best_positions.copy()])
        self.personal_best_scores_history = np.array([self.personal_best_scores.copy()])
        self.global_best_score_history = np.array([self.global_best_score])
        self.global_best_position_history = np.array([self.global_best_position.copy()])

        # hyper parameters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.num_iterations = num_iterations
        
        # logging
        logging.basicConfig(
            format='%(levelname)s [%(name)s.%(funcName)s]: %(message)s',
            level=logging.INFO,
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def exponential_smoothing(series, alpha):
        """Apply exponential smoothing to a series."""
        smoothed_series = np.zeros_like(series)
        smoothed_series[0] = series[0]
        for i in range(1, len(series)):
            smoothed_series[i] = alpha * series[i] + (1 - alpha) * smoothed_series[i-1]
        return smoothed_series

    def optimize(self):
        for _ in range(self.num_iterations):
            for i in range(self.num_particles):
                # Update velocity for each particle
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * random.random() * (self.personal_best_positions[i] - self.positions[i]) +
                                      self.c2 * random.random() * (self.global_best_position - self.positions[i]))

                # Clip velocity to its bounds
                self.velocities[i] = np.clip(self.velocities[i], -self.ball_carrier_velocity.max(), self.ball_carrier_velocity.max())

                # Update position for each particle
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], [self.xmin, self.ymin], [self.xmax, self.ymax])

                cost = self.objective_function(self.positions[i], self.velocities[i], **self.objective_function_params)

                # Update personal and global bests using cumulative distance
                if cost < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = cost
                    self.personal_best_positions[i] = self.positions[i].copy()

                if cost < self.global_best_score:
                    self.global_best_score = cost
                    self.global_best_position = self.positions[i].copy()

            # History update for each iteration
            self.positions_history = np.append(self.positions_history, [self.positions.copy()], axis=0)
            self.velocities_history = np.append(self.velocities_history, [self.velocities.copy()], axis=0)
            self.personal_best_positions_history = np.append(self.personal_best_positions_history, [self.personal_best_positions.copy()], axis=0)
            self.personal_best_scores_history = np.append(self.personal_best_scores_history, [self.personal_best_scores.copy()], axis=0)
            self.global_best_score_history = np.append(self.global_best_score_history, [self.global_best_score])
            self.global_best_position_history = np.append(self.global_best_position_history, [self.global_best_position.copy()], axis=0)
        
        return self.global_best_position, self.global_best_score

    def animate_play(self):
        fig, ax = plt.subplots(figsize=(12, 7))

        # Set the limits of the football field
        ax.set_xlim(0, self.xmax)
        ax.set_ylim(0, self.ymax)

        # Lines for the agents and ball carrier
        lines = [ax.plot([], [], '-', label=f'Agent {i}')[0] for i in range(self.num_particles)]
        ball_carrier_line, = ax.plot([], [], '-', color='red', label='Ball Carrier')

        # Setting up legend
        ax.legend()

        def init():
            # Initialize empty lines
            for line in lines:
                line.set_data([], [])
            ball_carrier_line.set_data([], [])
            return lines + [ball_carrier_line]

        def animate(frame):
            # Ensure there is data to plot
            if frame == 0:
                return lines + [ball_carrier_line]

            # Update the positions for each agent with smoothing
            for i, line in enumerate(lines):
                if frame < len(self.positions_history):
                    # Apply smoothing only if there are enough data points
                    smoothed_positions = self.exponential_smoothing(self.positions_history[:frame+1, i], alpha=0.04)
                    line.set_data(smoothed_positions[:, 0], smoothed_positions[:, 1])

            # Update the position for the ball carrier
            ball_carrier_line.set_data(self.target_positions[:frame+1, 0], self.target_positions[:frame+1, 1])

            return lines + [ball_carrier_line]

        # ... [rest of the animate_play method]

        anim = FuncAnimation(fig, animate, init_func=init, frames=len(self.positions_history), interval=100, blit=True)
        plt.show()
        return anim



