from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import sys
import numpy as np


# Function to create a football field
def create_football_field(fig, ax, line_color='white', field_color='darkgreen'):
    plt.xlim(0, 120)
    plt.ylim(0, 53.3)
    field_color_dec = field_color

    # Draw hash lines
    x_values = np.arange(11, 110, 1)
    ymin_value = 20.2 
    ymax_value = 20.8 
    ax.vlines(x=x_values, ymin=ymin_value, ymax=ymax_value, color=line_color, linewidth=.5)

    x_values = np.arange(11, 110, 1)
    ymin_value = 33.2 
    ymax_value = 32.6 
    ax.vlines(x=x_values, ymin=ymin_value, ymax=ymax_value, color=line_color,linewidth=.5)

    for i in range(24):
        rect = patches.Rectangle((5 * i, 0), 10, 53.3, linewidth=1, edgecolor=line_color, facecolor=field_color)
        ax.add_patch(rect)
    
    # for patch in ax.patches[-2:2]:
    #     patch.set_facecolor('black')
    #     patch.set_edgecolor('black')

    ax.patches[0].set_facecolor('black')
    ax.patches[0].set_edgecolor('black')
    ax.patches[1].set_facecolor('black')
    ax.patches[1].set_edgecolor('black')
    ax.patches[-1].set_facecolor('black')
    ax.patches[-1].set_edgecolor('black')
    ax.patches[-2].set_facecolor('black')
    ax.patches[-2].set_edgecolor('black')

    ax.tick_params(
        axis='both',
        which='both',
        direction='in',
        pad=-60,
        length=5,
        bottom=True,
        top=True,
        labeltop=True,
        labelbottom=True,
        left=False, right=False,
        labelleft=False,
        labelright=False,
        color=line_color
    )
    ax.set_xticks([i for i in range(10, 111)])
    label_set = []
    for i in range(1, 10):
            label_set += [" " for j in range(9)] + [str(i * 10) if i < 5 else str((10 - i) * 10)]
    #label_set[49] = '50'
    label_set = [" "] + label_set + [" " for j in range(10)]
    ax.set_xticklabels(label_set, fontsize=15, color=line_color)


    image_path = "/Users/benwolbransky/nfl-big-data-bowl-2024/images/bdb_logo.png"
    img = plt.imread(image_path)
    img_height = 53.3 / 5  # Adjust to the height of the field
    img_width = img_height * img.shape[1] / img.shape[0]
    ax.imshow(img, extent=[60 - img_width / 2, 60 + img_width / 2, 26.65 - img_height / 2, 26.65 + img_height / 2], aspect='auto', zorder=2)

    return fig, ax


class NFLPlayApp(tk.Tk):
    def __init__(self, df):
        super().__init__()
        self.df = self.every_other_agent_frame(df)
        self.title("NFL Play Animation")
        self.geometry("800x600")

        # Matchup Selection: Acts as Game ID
        self.matchup_label = ttk.Label(self, text="Select Matchup:")
        self.matchup_label.pack()
        self.matchup_combobox = ttk.Combobox(self, values=df['matchup'].unique().tolist(), state='readonly', width=50)
        self.matchup_combobox.pack()
        self.matchup_combobox.bind("<<ComboboxSelected>>", self.update_play_desc_combobox)

        # Play Desc selection
        self.play_desc_label = ttk.Label(self, text="Select Play Description:")
        self.play_desc_label.pack()
        self.play_desc_combobox = ttk.Combobox(self, state='readonly', width=50)
        self.play_desc_combobox.pack()

        self.animate_button = ttk.Button(self, text="Animate Play", command=self.animate_play)
        self.animate_button.pack()

        control_frame = ttk.Frame(self)
        control_frame.pack()

        # Checkbox for Secondary players
        self.secondary_checkbox_var = tk.BooleanVar(value=True)
        self.secondary_checkbox = ttk.Checkbutton(self, text="Visualize Secondary?",
                                                  variable=self.secondary_checkbox_var)
        self.secondary_checkbox.pack()

        # Toggle box for Agent Path
        self.agent_checkbox_var = tk.BooleanVar(value=False)
        self.agent_checkbox = ttk.Checkbutton(self, text="Visualize Agent?", variable=self.agent_checkbox_var)
        self.agent_checkbox.pack()

        # Toggle box for Entire Agent Path
        self.entire_agent_path_var = tk.BooleanVar(value=False)
        self.entire_agent_path_checkbox = ttk.Checkbutton(self, text="Visualize Agent Path?", variable=self.entire_agent_path_var)
        self.entire_agent_path_checkbox.pack()

        # Toggle box for Entire Agent Path
        self.save_animation = tk.BooleanVar(value=False)
        self.save_animation_checkbox = ttk.Checkbutton(self, text="Save Animation?", variable=self.save_animation)
        self.save_animation_checkbox.pack()

        # Playback control buttons packed in a row within the control frame
        self.play_button = ttk.Button(control_frame, text="Play", command=self.play_animation)
        self.play_button.pack(side=tk.LEFT)

        self.pause_button = ttk.Button(control_frame, text="Pause", command=self.pause_animation)
        self.pause_button.pack(side=tk.LEFT)

        self.canvas = None

        # Animation control variables
        self.animation = None
        self.animation_running = False
        self.animation_speed = 1.0  # 1x speed
    
    def every_other_agent_frame(self, df):
            unchanged_columns = df.columns.difference(['smooth_x','smooth_y'])
            unchanged_df = df[unchanged_columns]
            unchanged_df = unchanged_df.set_index(['displayName','frameId'])

            smooth_df = df.set_index(['displayName','frameId'])[['smooth_x','smooth_y']]
            
            result_df = pd.DataFrame()

            for display_name in df['displayName'].unique():
                display_name_data = df[df['displayName'] == display_name]
                display_name_data = display_name_data.groupby('frameId').first().reset_index()
                # Find the halfway point
                halfway_point = len(display_name_data) // 2

                # Create a new DataFrame with every other row skipped after the halfway point
                new_dataframe = pd.concat([display_name_data.iloc[:halfway_point, :], display_name_data.iloc[halfway_point + 1::2, :]])
                
                import numpy as np
                new_dataframe['frameId'] = np.arange(1, len(new_dataframe) + 1)
                    
                # Append to the result DataFrame
                result_df = pd.concat([result_df,new_dataframe])

            result_df = result_df.set_index(['displayName','frameId'])[['smooth_x','smooth_y']]

            return unchanged_df.merge(result_df, left_index = True, right_index = True, how = 'left').reset_index()

    def animate_play_func(self, df, visualize_secondary=True):

        fig, ax = plt.subplots(figsize =(120/12,53.30/12))
        fig, ax = create_football_field(fig, ax)
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 53.3)

        ax.set_title(
            label='Game ID: ' + str(df['gameId'].values[0]) + ' Play ID: ' + str(df['playId'].values[0]),
            fontsize=12,
        )
        fig.suptitle(
            t=df['playDescription'].values[0],
            fontsize=12,
        )

        if self.entire_agent_path_var.get():
            agent_df = df.dropna(subset = ['smooth_x'])
            for agent in agent_df['displayName'].unique():
                plot_df = agent_df[agent_df['displayName'] == agent].groupby(['nflId','gameId','playId','frameId'])[['smooth_x','smooth_y']].mean()
                ax.plot(plot_df['smooth_x'], plot_df['smooth_y'], label=agent,  linestyle='dotted', color='gray', alpha=0.8)

        # # line of scrimmage
        # ax.vlines(
        #     100 - df['yardlineNumber'].values[0] if df['yardlineSide'].values[0] != df['possessionTeam'].values[0] else df['yardlineNumber'].values[0],
        #     0,
        #     53.3,
        #     colors='black',
        #     linestyles='dashed',
        #     linewidth=1.5
        # )
        # Initialize player and football markers
        player_dots = {
            displayName: 
            ax.plot([], [], 'o', color=player_colors.get(displayName), alpha=.8, label=displayName)[0]
            if displayName != 'football' else
            ax.plot([], [], '^', color='brown', alpha=1, markeredgecolor='black', label=displayName)[0]
            # Use '^' (triangle) for players without team_color
            for displayName in df['displayName'].unique()
        }

        if self.agent_enabled:
            agent_df = df.dropna(subset=['smooth_x'])
            for displayName in agent_df['displayName'].unique():
                player_dots[f'{displayName}_agent'] = \
                ax.plot([], [], 's', color=player_colors.get(displayName), alpha=1, markeredgecolor='black', label=displayName)[0]

        def update(frame_id):
            for dot in player_dots.values():
                dot.set_data([], [])

            frame_data = df[df['frameId'] == frame_id]
            for displayName, dot in player_dots.items():
                is_agent = "agent" in displayName
                
                if is_agent:
                    player_data = frame_data[frame_data['displayName'] == displayName.replace('_agent','')]
                else:
                    player_data = frame_data[frame_data['displayName'] == displayName]

                player_secondary = player_data['position'].iloc[0] in ['SS', 'CB', 'FS']
 
                if self.agent_enabled and is_agent:
                    dot.set_data(player_data['smooth_x'].iloc[0], player_data['smooth_y'].iloc[0])
                elif player_secondary and not is_agent:
                    if visualize_secondary:
                        dot.set_data(player_data['x'].iloc[0], player_data['y'].iloc[0])
                    else:
                        continue
                else:
                    dot.set_data(player_data['x'].iloc[0], player_data['y'].iloc[0])

            return list(player_dots.values())

        ani = animation.FuncAnimation(fig, update, frames=df['frameId'].unique(), interval=100, blit=True, repeat=True)
        
        if self.save_animation.get():
            ani.save('animation.gif')

        # Display a message or perform any other necessary actions after saving
        return fig, ani

    def update_play_desc_combobox(self, event):
        matchup = self.matchup_combobox.get()
        play_desc = self.df[self.df['matchup'] == matchup]['playDescription'].unique().tolist()
        self.play_desc_combobox['values'] = play_desc

    def animate_play(self):
        matchup = self.matchup_combobox.get()
        play_desc = self.play_desc_combobox.get()
        visualize_secondary = self.secondary_checkbox_var.get()
        self.agent_enabled = self.agent_checkbox_var.get()

        filtered_df = self.df[(self.df['matchup'] == matchup) & (self.df['playDescription'] == play_desc)]
        fig, ani = self.animate_play_func(filtered_df, visualize_secondary)
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.animation = ani  # Store the animation object
        self.play_animation()  # Automatically start the animation

    def play_animation(self):
        if self.animation and not self.animation_running:
            self.animation.event_source.start()
            self.animation_running = True

    def pause_animation(self):
        if self.animation:
            self.animation.event_source.stop()
            self.animation_running = False

    def on_close(self):
        """ Close the application. """
        sys.exit(0)


if __name__ == '__main__':
    df = pd.read_csv('https://bigdatabowl2023.nyc3.cdn.digitaloceanspaces.com/raw/tracking_data/tracking_week_1.csv')
    colors = pd.read_csv('https://bigdatabowl2023.nyc3.cdn.digitaloceanspaces.com/raw/colors.csv')
    players = pd.read_csv('https://bigdatabowl2023.nyc3.cdn.digitaloceanspaces.com/raw/players.csv')
    play_desc = pd.read_csv('https://bigdatabowl2023.nyc3.cdn.digitaloceanspaces.com/raw/plays.csv')
    games = pd.read_csv('https://bigdatabowl2023.nyc3.cdn.digitaloceanspaces.com/raw/games.csv')
    paths = pd.read_csv('/Users/benwolbransky/nfl-big-data-bowl-2024/data/specific_plays_paths.csv')
    games['matchup'] = games['visitorTeamAbbr'] + ' @ ' + games['homeTeamAbbr'] + ' week ' + games['week'].astype(
        str) + ' of the ' + games['season'].astype(str) + ' season'
    colors = colors.rename({
        'team_abbr': 'club'
    }, axis=1)
    df = df.merge(colors, on='club', how='left')
    df = df.merge(play_desc, on=['gameId', 'playId'], how='left')
    df = df.merge(games, on=['gameId'], how='left')
    df = df.merge(players[['nflId', 'position']], on=['nflId'], how='left')
    # Ensure data types are consistent
    paths['gameId'] = paths['gameId'].astype(int)
    paths['playId'] = paths['playId'].astype(int)
    paths['nflId'] = paths['nflId'].astype(int)  # Convert to int if it doesn't have fractional values
    print(df.shape)
    # Merge paths DataFrame with the main DataFrame
    df = df.merge(paths, on=['gameId', 'playId', 'nflId', 'frameId'], how='left')

    # Filter out plays that are not in the paths DataFrame
    plays_with_paths = paths[['gameId', 'playId']].drop_duplicates()
    df = pd.merge(df, plays_with_paths, on=['gameId', 'playId'], how='inner')

    print(df.shape)
    app = NFLPlayApp(df)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    player_colors = dict(
        df.drop_duplicates(subset=['displayName', 'team_color'], keep='first')[['displayName', 'team_color']].fillna(
            {'team_color': 'brown'}).to_records(index=False))
    app.mainloop()