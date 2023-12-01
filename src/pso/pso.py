import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde, multivariate_normal
from matplotlib import pyplot as plt
from typing import Callable
import logging
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

class PSODefense:
    def __init__(
        self, 
        play: pd.DataFrame, 
        objective_function: Callable,
        def_abbr: str, 
        off_abbr: str,
        ball_carrier_id: int,
        move_particles: list[int] = [],
        w: float = 0.1,
        c1: float = 2,
        c2: float = 0.2,
        num_iterations: int = 10_000,
        min_velocity: float = -0.3,
        max_velocity: float = 0.3
    ):

        # objective function and param setting
        self.objective_function = objective_function
        initial_ball_carrier_position = play.loc[(play['frameId'] == 1) & (play['nflId'] == ball_carrier_id)][['x', 'y']].values[0]
        self.objective_function_params = {
            'ball_carrier_position': initial_ball_carrier_position
        }

        # DataFrame of the play
        self.play = play.sort_values(by=['frameId', 'nflId']) 

        # metadata
        self.num_frames = len(play.frameId.unique())
        self.def_abbr = def_abbr
        self.off_abbr = off_abbr
        self.ball_carrier_id = int(ball_carrier_id)
        
        # if move_particles is None, then all particles will be able to move
        # if there is a certain set of particles that should be able to move, then specify them here
        self.move_particles = move_particles

        # constraints
        self.xmin = 0
        self.xmax = 120  # Including endzones
        self.ymax = 53.3  # Standard width of a football field
        self.ymin = 0
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity

        self.num_particles = len(self.play.loc[self.play['club'] == def_abbr].nflId.unique())
        self.num_dimensions = 2

        # this represents the actual x and y positions of the particles
        # not the positions that will be updated by method self.optimize_frame
        # group by frame and then convert 
        frames = self.play.loc[self.play['club'] == def_abbr].groupby('frameId')
        self.actual_particle_positions = np.array([frame[['x', 'y']].to_numpy() for _, frame in frames])
        self.actual_velocities = np.array([frame[['x_velocity', 'y_velocity']].to_numpy() for _, frame in frames])

        # initialize the positions and velocities of the particles
        # these are the positions and particles that will be updated by method self.optimize_frame
        self.positions = self.play.loc[(self.play['club'] == def_abbr) & (self.play['frameId'] == 1)][['x', 'y']].values
        self.velocities = np.zeros((self.num_particles, self.num_dimensions))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(self.num_particles, np.inf)
        self.global_best_score = np.inf
        self.global_best_position = np.full(self.num_dimensions, np.nan)

        # history of the positions optimized by PSO
        self.positions_history = [self.positions.copy()]
        self.velocities_history = [self.velocities.copy()]
        self.personal_best_positions_history = [self.personal_best_positions.copy()]
        self.personal_best_scores_history = [self.personal_best_scores.copy()]
        self.global_best_score_history = [self.global_best_score]
        self.global_best_position_history = [self.global_best_position.copy()]

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

    def optimize_frame(self) -> (float, np.ndarray):
        """Optimize the positions of the particles for a single frame

        Args:
            self: Instance of the PSODefense class

        Returns:
            (float, np.ndarray): The global best score and the global best position
        """

        for _ in range(self.num_iterations):
            fitness = self.objective_function(self.positions, **self.objective_function_params)

            for i in range(self.num_particles):
                if fitness[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness[i]
                    self.personal_best_positions[i] = self.positions[i]
                
                if fitness[i] < self.global_best_score:
                    self.global_best_score = fitness[i]
                    self.global_best_position = self.positions[i]

        # After completing all iterations, update the velocities
        for i in range(self.num_particles):
            # if the particle is selected as one that should be able to adjust with the defense, then update the velocity
            if i in self.move_particles:
                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + self.c1 * np.random.rand() * (self.personal_best_positions[i] - self.positions[i])
                    + self.c2 * np.random.rand() * (self.global_best_position - self.positions[i])
                )

                self.velocities[i] = np.clip(self.velocities[i], self.min_velocity, self.max_velocity)

                self.positions[i] += self.velocities[i]
                self.positions = np.clip(self.positions, [self.xmin, self.ymin], [self.xmax, self.ymax])
            else:
                # if the particle is not selected as one that should be allowed to move,
                # then set the velocity and position to be the same as the actual position
                self.velocities[i] = self.actual_velocities[self.frame_id-1][i]
                self.positions[i] = self.actual_particle_positions[self.frame_id-1][i]

        return self.global_best_score, self.global_best_position

    def optimize_play(self):
        """Optimize the positions of the particles for the entire play

        Args:
            self: Instance of the PSODefense class

        Returns:
            None
        """
        for frame_id in range(1, self.num_frames+1):
            
            self.frame_id = frame_id

            frame = self.play.loc[self.play['frameId'] == self.frame_id]
            self.frame = frame

            self.logger.info(f'Optimizing frame {self.frame_id}')

            self.objective_function_params['ball_carrier_position'] = frame.loc[frame['nflId'] == self.ball_carrier_id][['x', 'y']].values[0]
            self.logger.info(f'Ball carrier position: {self.objective_function_params["ball_carrier_position"]}')

            # reset global best score and position
            self.global_best_score = np.inf
            self.global_best_position = np.full(self.num_dimensions, np.nan)

            self.optimize_frame()

            self.positions_history.append(self.positions.copy())
            self.velocities_history.append(self.velocities.copy())
            self.personal_best_positions_history.append(self.personal_best_positions.copy())
            self.personal_best_scores_history.append(self.personal_best_scores.copy())
            self.global_best_score_history.append(self.global_best_score)
            self.global_best_position_history.append(self.global_best_position.copy())

    def animate_play(self):
        fig, ax = plt.subplots()
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)

        # Initialize colors for each particle
        colors = ['red' if i in self.move_particles else 'blue' for i in range(self.num_particles)]
        
        # Initialize scatter plot with dummy data matching the length of colors array
        scatter = ax.scatter(np.zeros(self.num_particles), np.zeros(self.num_particles), s=30, c=colors)

        # Initialize lines for actual and PSO suggested paths
        actual_path, = ax.plot([], [], 'k-', linewidth=2, label='Actual Path')
        pso_path, = ax.plot([], [], 'g-', linewidth=2, label='PSO Path')

        # Initialize ball carrier's path and position
        ball_carrier_path, = ax.plot([], [], 'b--', linewidth=1, label='Ball Carrier Path')
        ball_carrier_pos = ax.scatter([], [], c='yellow', s=50, label='Ball Carrier')

        # Update function for the animation
        def update(frame_number):
            # Update scatter plot
            positions = self.positions_history[frame_number]
            scatter.set_offsets(positions)

            # Update the actual path and PSO path
            if frame_number > 0:
                actual_path.set_data(self.actual_particle_positions[:frame_number + 1, self.move_particles[0], 0], 
                                     self.actual_particle_positions[:frame_number + 1, self.move_particles[0], 1])
                pso_path.set_data([pos[self.move_particles[0], 0] for pos in self.positions_history[:frame_number + 1]],
                                  [pos[self.move_particles[0], 1] for pos in self.positions_history[:frame_number + 1]])

            # Update ball carrier's path and position
            ball_carrier_positions = self.play.loc[self.play['nflId'] == self.ball_carrier_id, ['frameId', 'x', 'y']].sort_values('frameId')
            ball_carrier_path.set_data(ball_carrier_positions['x'][:frame_number + 1], 
                                       ball_carrier_positions['y'][:frame_number + 1])
            try:
                ball_carrier_current_pos = ball_carrier_positions.iloc[frame_number]
                ball_carrier_pos.set_offsets([[ball_carrier_current_pos['x'], ball_carrier_current_pos['y']]])

                return scatter, actual_path, pso_path, ball_carrier_path, ball_carrier_pos
            except:
                return scatter, actual_path, pso_path, ball_carrier_path

        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(self.positions_history), interval=100, blit=True)

        # Legend
        ax.legend()

        # Show the animation
        plt.show()

        if save_fig:
            ani.save('img/animation.gif', writer='imagemagick')    


