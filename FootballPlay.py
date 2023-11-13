import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import nfl_data_py as nfl

class FootballPlay(gym.Env):
    '''
    Custom gym environment for a football play based on
    NFL Big Data Bowl tracking data.
    '''

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, df, agent_nflid, render_mode=None, max_frames=100):
        super().__init__()

        self.df = df
        self.agent_nflid = agent_nflid  # The nflId of the player controlled by the agent

        # Get initial positions of players and football,
        # and also get the initial position of the agent
        self.players_positions_df, self.df, self._initial_agent_location = self._get_initial_positions(df, agent_nflid)
        self.football_position = self._get_football_initial_position()


        # TODO: all dimensions, observation spaces, and action spaces
        # need to be adjusted; the granularity of the tracking data
        # is to the hundreths place (e.g. 0.01), so the dimensions
        # of the field and the agent's position need to be more fine

        # Dimensions of a football field in yards
        self.xmin = 0
        self.xmax = 120  # Including endzones
        self.ymax = 53.3  # Standard width of a football field
        self.ymin = 0

        self.size_x = int(self.xmax)
        self.size_y = int(self.ymax)

        # Window size for pygame rendering
        self.window_size = 512

        # TODO: Understand what "observation space means"
        # also, the target right now is based on the example custom space
        # in the gym documentation; eventually this should be the football or ball carrier
        # see: https://gymnasium.farama.org/api/spaces/

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(np.array([self.xmin, self.ymin]), 
                                    np.array([self.size_x - 1, self.size_y - 1]), dtype=np.int32),
                "target": spaces.Box(np.array([self.xmin, self.ymin]), 
                                     np.array([self.size_x - 1, self.size_y - 1]), dtype=np.int32),
            }
        )

        # Continuous action space: each action is a 2D vector with components in the range [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Set the maximum number of frames per episode based on the tracking data
        self.max_frames = max_frames
        self.current_frame = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_initial_positions(self, df, agent_nflid):
        '''
        Get the initial positions of the players, and remove
        the specified nflId's movements so the agent can assume
        control from that player's starting position.
        Returns the players_positions_df, the modified df, and the _agent_location.
        '''

        initial_frame_df = df[df['frameId'] == 1]

        # Create a new DataFrame for player positions and team colors
        player_positions_df = initial_frame_df[initial_frame_df['nflId'] != agent_nflid][
            ['nflId', 'x', 'y', 'team_color', 'team_color2', 'jerseyNumber']
        ]

        # Extract the agent's initial location
        agent_initial_df = initial_frame_df[initial_frame_df['nflId'] == agent_nflid]
        agent_location = np.array([0, 0], dtype=np.int32)  # Default agent location if not found

        if not agent_initial_df.empty:
            agent_location = np.array([
                agent_initial_df.iloc[0]['x'],
                agent_initial_df.iloc[0]['y']
            ], dtype=np.int32)

        # Remove the agent player's movements from the original dataframe
        df = df[df['nflId'] != agent_nflid]

        return player_positions_df, df, agent_location


    def _get_football_initial_position(self):
        '''
        Initialize the starting point of the football.
        '''

        initial_frame_df = self.df[(self.df['frameId'] == 1) &\
                                    (self.df['displayName'] == 'football')]

        if not initial_frame_df.empty:
            row = initial_frame_df.iloc[0]
            return np.array([row['x'], row['y']])
        return np.array([0, 0])  # Default position if not found

    def _get_obs(self):
        '''
        The _get_obs method is responsible for generating the observation that the agent receives 
        after taking an action in the environment. This observation is a representation of the 
        environment's current state and is crucial for the agent to make informed decisions 
        about future actions.
        '''

        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        '''
        The _get_info method is designed to provide supplementary information 
        about the environment's state that is not necessarily part of the observation used 
        for making decisions. This information can be useful for debugging, training diagnostics, or 
        understanding the environment's dynamics better.
        '''

        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self):
        '''
        Reset the environment to its initial state and return an initial observation.
        '''
        # TODO: Priority #1, reset other player's to their initail positions as well
        # Right now, the non-agent players are the football are just
        # animations, not actually part of the environment.
        # These will all also have to be part of the the _get_obs method

        # Reset the agent to its initial position
        self._agent_location = self._initial_agent_location.copy()

        # The agent's location is already set in __init__, so we just use it directly
        self._target_location = self._agent_location

        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, [self.size_x, self.size_y], 
                                                            dtype=np.int32)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        # Reset the current frame for each iteration
        self.current_frame = 1

        return observation, info

    def step(self, action):
        '''
        Step through each frame of the play, and return the observation, reward,
        termination signal, and info for the current frame.
        '''

        # Normalize the action to make sure it's within the allowed range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Move the agent based on the action
        new_location = self._agent_location + action

        # Update agent's location
        self._agent_location = np.clip(
            new_location, [self.xmin, self.ymin], [self.size_x - 1, self.size_y - 1]
        ).astype(np.int32)

        # Update the positions of the other players and the football for the current frame
        current_frame_df = self.df[self.df['frameId'] == self.current_frame]
        for _, row in current_frame_df.iterrows():
            if pd.notna(row['nflId']):  # Check if it's a player and not the football
                # Update positions in the DataFrame
                self.players_positions_df.loc[self.players_positions_df['nflId'] == row['nflId'], ['x', 'y']] = row['x'], row['y']
            elif row['displayName'] == 'football':
                self.football_position = np.array([row['x'], row['y']])

        self.current_frame += 1

        # TODO: Termination function! How do we know when the play is over?
        # What if a non-agent tackles the ball carrier?

        # Terminate if the agent has tackled the target 
        # or if the max number of frames has been reached
        terminated = np.array_equal(self._agent_location, self._target_location) or\
                                                 self.current_frame >= self.max_frames

        # TODO: Reward function! How do we reward the agent?
        # Obviously, a tackle - but what is a tackle? 
        # How do we know when a tackle occur empirically?
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        '''
        Render the play in a pygame window
        for humans.
        '''

        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        '''
        Render the current frame of the play in a pygame window
        for humans.
        '''

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # Fill with white

        # Drawing logic

        # Calculate pixel size based on the window size and field dimensions
        pix_square_size_x = self.window_size / self.size_x
        pix_square_size_y = self.window_size / self.size_y

        # Draw the field
        field_color = (0, 128, 0)  # Dark green for the field
        pygame.draw.rect(
            canvas,
            field_color,
            pygame.Rect(0, 0, self.window_size, self.window_size)
        )

        # Draw yard lines
        for x in range(self.size_x):
            line_color = (255, 255, 255)  # White for yard lines
            if x % 10 == 0:  # Every 10 yards
                pygame.draw.line(
                    canvas,
                    line_color,
                    (x * pix_square_size_x, 0),
                    (x * pix_square_size_x, self.window_size)
                )

        for _, row in self.players_positions_df.iterrows():
            if pd.notna(row['nflId']):  # Check if it's a player and not the football
                team_color = row.get('team_color', (0, 0, 255))  # Default color if not found
                team_color2 = row.get('team_color2', (0, 0, 0))  # Default border color

                # Draw each player using their updated position and team colors
                pygame.draw.circle(
                    canvas,
                    team_color2,
                    (int(row['x'] * pix_square_size_x), int(row['y'] * pix_square_size_y)),
                    int(pix_square_size_x / 2) + 4  # Add 4 to make the circle bigger
                )
                # Outer circle (team_color2) as border
                pygame.draw.circle(
                    canvas,
                    team_color,
                    (int(row['x'] * pix_square_size_x), int(row['y'] * pix_square_size_y)),
                    int(pix_square_size_x / 2) + 5,
                    width=3  # Border thickness
                )

        # Draw the football using its updated position
        pygame.draw.circle(
            canvas,
            (139, 69, 19),
            (
                int(self.football_position[0] * pix_square_size_x),
                int(self.football_position[1] * pix_square_size_y),
            ),
            int(pix_square_size_x / 4) + 3,
        )

        # Draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 0),  # Black color for the agent
            (
                int(self._agent_location[0] * pix_square_size_x + pix_square_size_x / 2),
                int(self._agent_location[1] * pix_square_size_y + pix_square_size_y / 2),
            ),
            int(pix_square_size_x / 2) + 4,
        )

        # Outer circle for the agent
        pygame.draw.circle(
            canvas,
            (255, 255, 255),  # White color for the agent
            (
                int(self._agent_location[0] * pix_square_size_x + pix_square_size_x / 2),
                int(self._agent_location[1] * pix_square_size_y + pix_square_size_y / 2),
            ),
            int(pix_square_size_x / 2) + 5,
            width=3,  # Border thickness
        )

        # Draw the target
        pygame.draw.circle(
            canvas,
            (255, 0, 0),  # Red color for the target
            (
                int(self._target_location[0] * pix_square_size_x + pix_square_size_x / 2),
                int(self._target_location[1] * pix_square_size_y + pix_square_size_y / 2),
            ),
            int(pix_square_size_x / 2),
        )

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        '''
        Close the pygame window.
        '''
        # TODO: Need a more graceful way to exit pygame

        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
