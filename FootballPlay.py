import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import nfl_data_py as nfl

class FootballPlay(gym.Env):
    '''
    Custom gym environment for a football play.
    '''
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, df, render_mode=None, max_frames=100):
        super().__init__()

        self.df = df
        self.players_positions = self._get_initial_positions()
        self.football_position = self._get_football_initial_position()

        # Dimensions of a football field in yards
        self.xmin = 0
        self.xmax = 120  # Including endzones
        self.ymax = 53.3  # Standard width of a football field
        self.ymin = 0

        self.size_x = int(self.xmax)
        self.size_y = int(self.ymax)

        # Window size for rendering
        self.window_size = 512

        # Observations are dictionaries with the agent's and the target's location.
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(np.array([self.xmin, self.ymin]), np.array([self.size_x - 1, self.size_y - 1]), dtype=np.int32),
                "target": spaces.Box(np.array([self.xmin, self.ymin]), np.array([self.size_x - 1, self.size_y - 1]), dtype=np.int32),
            }
        )

        # Continuous action space: each action is a 2D vector with components in the range [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Define the obstacle location
        self.obstacle_location = np.array([88.37, 27.27])

        # Set the maximum number of frames per episode based on the tracking data
        self.max_frames = max_frames
        self.current_frame = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_initial_positions(self):
        # Filter the dataframe to get the initial positions of the players
        initial_frame_df = self.df[self.df['frameId'] == 1]
        player_positions = {}
        for _, row in initial_frame_df.iterrows():
            if pd.notna(row['nflId']):  # Check if it's a player and not the football
                player_positions[row['nflId']] = np.array([row['x'], row['y']])
        return player_positions

    def _get_football_initial_position(self):
        initial_frame_df = self.df[(self.df['frameId'] == 1) & (self.df['displayName'] == 'football')]
        if not initial_frame_df.empty:
            row = initial_frame_df.iloc[0]
            return np.array([row['x'], row['y']])
        return np.array([0, 0])  # Default position if not found

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, [self.size_x, self.size_y], dtype=np.int32)
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, [self.size_x, self.size_y], dtype=np.int32)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        # Reset the current frame for each iteration
        self.current_frame = 0

        return observation, info
    
    def step(self, action):
        # Normalize the action to make sure it's within the allowed range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Check for collision with the obstacle
        new_location = self._agent_location + action
        if np.array_equal(new_location, self.obstacle_location):
            # Prevent movement if it results in a collision
            new_location = self._agent_location

        # Update agent's location
        self._agent_location = np.clip(
            new_location, [self.xmin, self.xmin], [self.size_x - 1, self.size_y - 1]
        ).astype(np.int32)

        # Update the positions of the players and the football for the current frame
        current_frame_df = self.df[self.df['frameId'] == self.current_frame]
        for _, row in current_frame_df.iterrows():
            if pd.notna(row['nflId']):  # Check if it's a player and not the football
                self.players_positions[row['nflId']] = np.array([row['x'], row['y']])
            elif row['displayName'] == 'football':
                self.football_position = np.array([row['x'], row['y']])

        self.current_frame += 1

        terminated = np.array_equal(self._agent_location, self._target_location) or self.current_frame >= self.max_frames
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        

        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
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

        # Draw the updated positions of the players
        for nflId, position in self.players_positions.items():
            # Draw each player using their updated position
            pygame.draw.circle(
                canvas,
                (0, 0, 255),  # Blue color for players
                (
                    int(position[0] * pix_square_size_x),
                    int(position[1] * pix_square_size_y),
                ),
                int(pix_square_size_x / 2),
            )

        # Draw the football using its updated position
        pygame.draw.circle(
            canvas,
            (255, 165, 0),  # Orange color for the football
            (
                int(self.football_position[0] * pix_square_size_x),
                int(self.football_position[1] * pix_square_size_y),
            ),
            int(pix_square_size_x / 4),
        )

        # Draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),  # Blue color for the agent
            (
                int(self._agent_location[0] * pix_square_size_x + pix_square_size_x / 2),
                int(self._agent_location[1] * pix_square_size_y + pix_square_size_y / 2),
            ),
            int(pix_square_size_x / 2),
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

        # Draw the obstacle
        pygame.draw.rect(
            canvas,
            (0, 255, 0),  # Green color for the obstacle
            pygame.Rect(
                self.obstacle_location[0] * pix_square_size_x,
                self.obstacle_location[1] * pix_square_size_y,
                pix_square_size_x,
                pix_square_size_y,
            ),
        )

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
