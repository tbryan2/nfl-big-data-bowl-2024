import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from typing import Callable
import logging
from matplotlib.animation import FuncAnimation
import random
from src.pso.target_selection import select_common_target
from IPython.display import HTML
from frechetdist import frdist
from scipy.interpolate import interp1d
import plotly.graph_objs as go

class PSODefense:
    """
        A class to represent Particle Swarm Obstacle avoidance algorithm on a play level using data from the 2024 NFL Big Data Bowl.

        Attributes
        ----------
        play : pd.DataFrame
            Raw DataFrame representing the play, containing positional data for players.
        objective_function : Callable
            The objective function used to evaluate particle positions in the optimization process.
        def_abbr : str
            Abbreviation for the defensive team.
        off_abbr : str
            Abbreviation for the offensive team.
        ball_carrier_id : int
            Unique identifier for the ball carrier player.
        positional_group : str
            Positional group for the defensive players ('safeties', 'linebackers', 'cornerbacks', 'secondary').
        w : float, optional
            Inertia weight in the Particle Swarm Optimization algorithm, default is 0.1.
        c1 : float, optional
            Cognitive parameter in the Particle Swarm Optimization algorithm, default is 2.
        c2 : float, optional
            Social parameter in the Particle Swarm Optimization algorithm, default is 0.2.
        num_iterations : int, optional
            Number of iterations for the optimization process, default is 10,000.
        time_weighting_factor : int, optional
            Weighting factor for time in the objective function, default is 3.
        obstacle_avoidance_factor : float, optional
            Factor influencing obstacle avoidance in the objective function, default is 1.0.

        Methods
        -------
        optimize()
            Executes the Particle Swarm Optimization algorithm to find optimal positions for defensive players.
        
        animate_play()
            Animates the play, displaying the movement of defensive players and the ball carrier over frames.
        
        exponential_smoothing(series, alpha)
            Applies exponential smoothing to a series.

        """
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
        time_weighting_factor: int = 3,
        obstacle_avoidance_factor: float = 1.0,
        max_iterations: int = 10_000,
        stop_threshold: float = 0.1
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
        #else: 
        #   add exception handling

        # agents
        self.agents = self.play.loc[self.play['position'].isin(self.positional_group)]['nflId'].unique()

        # constraints
        self.xmin = 0
        self.xmax = 120  # Including endzones
        self.ymax = 53.3  # Standard width of a football field
        self.ymin = 0

        self.stop_threshold = stop_threshold
        self.max_iterations = self.num_frames

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
        self.smoothed_positions_history = None


        # hyper parameters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
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
        """Executes the Particle Swarm Optimization algorithm to find optimal positions for defensive players."""
        iteration = 0
        while iteration < self.max_iterations:
            any_within_threshold = False  # Changed from all_within_threshold to any_within_threshold
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

                # Check if the particle is within the stop threshold from the target
                distance_to_target = np.linalg.norm(self.positions[i] - self.best_target)
                if distance_to_target <= self.stop_threshold:
                    any_within_threshold = True
                    break  # Stop the inner loop if any agent is within threshold distance from the target

            # History update for each iteration
            self.positions_history = np.append(self.positions_history, [self.positions.copy()], axis=0)
            self.velocities_history = np.append(self.velocities_history, [self.velocities.copy()], axis=0)
            self.personal_best_positions_history = np.append(self.personal_best_positions_history, [self.personal_best_positions.copy()], axis=0)
            self.personal_best_scores_history = np.append(self.personal_best_scores_history, [self.personal_best_scores.copy()], axis=0)
            self.global_best_score_history = np.append(self.global_best_score_history, [self.global_best_score])
            self.global_best_position_history = np.append(self.global_best_position_history, [self.global_best_position.copy()], axis=0)

            if any_within_threshold:
                break  # Stop the outer loop if any agent is within threshold distance from the target

            iteration += 1

        return self.global_best_position, self.global_best_score


    
    def dynamic_alpha(self, distance, max_alpha=0.03, min_alpha=0.001, scale_factor=10):
        # Adjust alpha based on distance to the target
        alpha = max_alpha / (1 + distance / scale_factor)
        # Clamp alpha within a range
        return max(min_alpha, min(max_alpha, alpha))

    def smooth_paths(self):
        self.smoothed_positions_history = []
        for i in range(self.num_particles):
            smoothed_path = [self.positions_history[0, i, :]]
            for frame in range(1, len(self.positions_history)):
                current_position = self.positions_history[frame, i, :]
                target_position = self.best_targets[min(frame, len(self.best_targets) - 1)]
                distance = np.linalg.norm(current_position - target_position)
                alpha = self.dynamic_alpha(distance)
                new_smoothed_position = alpha * current_position + (1 - alpha) * smoothed_path[-1]
                smoothed_path.append(new_smoothed_position)
            self.smoothed_positions_history.append(np.array(smoothed_path))

    def calculate_frechet_distances(self):
        """Calculates the Fréchet distance between actual and smooth paths for each agent."""
        frechet_distances_data = []
        paths_data = []

        for i in range(self.num_particles):
            nfl_id = self.agents[i]
            actual_path = self.actual_particle_positions[:, i, :]
            smooth_path = self.smoothed_positions_history[i]

            # Check if the smooth_path is longer by one element
            if len(smooth_path) == len(actual_path) + 1:
                # Trim the last element from the smooth_path
                smooth_path = smooth_path[:-1]

            # Calculate the Fréchet distance
            distance = frdist(actual_path, smooth_path)

            for frame in range(len(actual_path)):
                # Assuming 'gameId' and 'playId' are columns in self.play DataFrame
                # and that 'frameId' is indexed starting from 1
                frame_data = self.play[(self.play['frameId'] == frame + 1) & (self.play['nflId'] == nfl_id)]
                if not frame_data.empty:
                    game_id = frame_data['gameId'].iloc[0]
                    play_id = frame_data['playId'].iloc[0]

                    paths_data.append({
                        'nflId': nfl_id,
                        'gameId': game_id,
                        'playId': play_id,
                        'frameId': frame + 1,
                        'actual_x': actual_path[frame, 0],
                        'actual_y': actual_path[frame, 1],
                        'smooth_x': smooth_path[frame, 0],
                        'smooth_y': smooth_path[frame, 1],
                    })

            # Append distance once per nflId
            frechet_distances_data.append({
                'nflId': nfl_id,
                'frechet_distance': distance
            })

        # Create DataFrames
        frechet_distances_df = pd.DataFrame(frechet_distances_data)
        paths_df = pd.DataFrame(paths_data)

        return frechet_distances_df, paths_df

    def animate_play(self):
        """Animates the play, displaying the movement of defensive players and the ball carrier over frames."""
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_xlim(0, self.xmax)
        ax.set_ylim(0, self.ymax)

        lines = [ax.plot([], [], '-', label=f'Agent {i}')[0] for i in range(self.num_particles)]
        ball_carrier_line, = ax.plot([], [], '-', color='red', label='Ball Carrier')
        ax.legend()

        def init():
            for line in lines:
                line.set_data([], [])
            ball_carrier_line.set_data([], [])
            return lines + [ball_carrier_line]

        def animate(frame):
            for i, line in enumerate(lines):
                if frame < len(self.smoothed_positions_history[i]):
                    x_data, y_data = self.smoothed_positions_history[i][:frame+1, 0], self.smoothed_positions_history[i][:frame+1, 1]
                    line.set_data(x_data, y_data)

            if frame < len(self.target_positions):
                ball_carrier_line.set_data(self.target_positions[:frame + 1, 0], self.target_positions[:frame + 1, 1])

            return lines + [ball_carrier_line]

                # Create the animation
        anim = FuncAnimation(fig, animate, init_func=init, frames=self.num_frames, interval=100, blit=True)

        # Convert the animation to HTML and display it in the notebook
        return HTML(anim.to_html5_video())

    def visualize_paths_with_plotly(self, frechet_distances_df):
        """Creates an interactive Plotly visualization of the player paths."""
        fig = go.Figure()

        # Find the agent with the lowest Fréchet distance
        min_frechet_nfl_id = frechet_distances_df.idxmin()['frechet_distance']

        # Define the color for the optimal paths
        color_optimal = 'blue'  # Blue for optimal paths

        # Add actual and optimal paths for each agent to the figure
        for i, nfl_id in enumerate(self.agents):
            actual_path = self.actual_particle_positions[:, i, :]
            smooth_path = self.smoothed_positions_history[i]
            frechet_distance = frechet_distances_df.loc[nfl_id, 'frechet_distance']

            # Add actual path with hover text for Fréchet distance
            fig.add_trace(go.Scatter(
                x=actual_path[:, 0], y=actual_path[:, 1],
                mode='lines+markers',
                name=f'NFLID {nfl_id}' + (" Best" if nfl_id == min_frechet_nfl_id else ""),
                line=dict(color='orange' if nfl_id == min_frechet_nfl_id else 'black'),
                text=f"Frechet Distance: {frechet_distance:.2f}",
                hoverinfo='text+name'
            ))

            # Add optimal (SWARM) path
            fig.add_trace(go.Scatter(
                x=smooth_path[:, 0], y=smooth_path[:, 1],
                mode='lines',
                name='SWARM Optimal',
                line=dict(color=color_optimal, dash='dash'),
                showlegend=i == 0  # Only show legend entry for SWARM once
            ))

        # Add ball carrier's path
        ball_carrier_path = self.target_positions
        fig.add_trace(go.Scatter(
            x=ball_carrier_path[:, 0], y=ball_carrier_path[:, 1],
            mode='lines',
            name='Ball Carrier',
            line=dict(color='red', width=2)
        ))

        # Update layout
        fig.update_layout(
            title='Player Paths',
            xaxis_title='X Position',
            yaxis_title='Y Position',
            legend_title='Legend',
            showlegend=True
        )

        # Show the figure
        fig.show()





