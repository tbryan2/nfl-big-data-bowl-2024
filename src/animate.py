import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import sys

# Function to create a football field
def create_football_field(fig, ax, line_color='black', field_color='white'):
    plt.xlim(0, 120)
    plt.ylim(0, 53.3)

    for i in range(12):
        rect = patches.Rectangle((10*i, 0), 10, 53.3, linewidth=1, edgecolor=line_color, facecolor=field_color)
        ax.add_patch(rect)

    ax.tick_params(
        axis='both', 
        which='both', 
        direction='in', 
        pad=-40, 
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
        label_set += [" " for j in range(9)] + [str(i*10) if i <= 5 else str((10-i)*10)]
    label_set = [" "] + label_set + [" " for j in range(10)]
    ax.set_xticklabels(label_set, fontsize=20, color=line_color)
    return fig, ax

def animate_play(df):
    
    fig, ax = plt.subplots()
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

    # # line of scrimmage
    # ax.vlines(
    #     100 - df['yardlineNumber'].values[0] if df['yardlineSide'].values[0] != df['possessionTeam'].values[0] else df['yardlineNumber'].values[0], 
    #     0, 
    #     53.3, 
    #     colors='black', 
    #     linestyles='dashed', 
    #     linewidth=1.5
    # )

    player_colors = dict(df.drop_duplicates(subset=['displayName', 'team_color'], keep='first')[['displayName', 'team_color']].fillna({'team_color': 'brown'}).to_records(index=False))
    # Initialize player markers
    player_dots = {
        displayName: ax.plot([], [], 'o', color=player_colors.get(displayName, 'brown'), label=displayName)[0] for displayName in df['displayName'].unique()
    }

    # Get team colors

    def update(frame_id):
        # Clear only the player markers, not the entire field
        for dot in player_dots.values():
            dot.set_data([], [])

        frame_data = df[df['frameId'] == frame_id]
        for displayName, dot in player_dots.items():
            player_data = frame_data[frame_data['displayName'] == displayName]
            if not player_data.empty:
                dot.set_data(player_data['x'].iloc[0], player_data['y'].iloc[0])
        
        return list(player_dots.values())

    ani = animation.FuncAnimation(fig, update, frames=df['frameId'].unique(), interval=100, blit=True, repeat=True)
    return fig, ani

class NFLPlayApp(tk.Tk):
    def __init__(self, df):
        super().__init__()
        self.df = df
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

    def update_play_desc_combobox(self, event):
        matchup = self.matchup_combobox.get()
        play_desc = self.df[self.df['matchup'] == matchup]['playDescription'].unique().tolist()
        self.play_desc_combobox['values'] = play_desc

    def animate_play(self):
        matchup = self.matchup_combobox.get()
        play_desc = self.play_desc_combobox.get()
        filtered_df = self.df[(self.df['matchup'] == matchup) & (self.df['playDescription'] == play_desc)]
        fig, ani = animate_play(filtered_df)
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
    play_desc = pd.read_csv('https://bigdatabowl2023.nyc3.cdn.digitaloceanspaces.com/raw/plays.csv')
    games = pd.read_csv('https://bigdatabowl2023.nyc3.cdn.digitaloceanspaces.com/raw/games.csv')
    games['matchup'] = games['visitorTeamAbbr'] + ' @ ' + games['homeTeamAbbr'] + ' week ' + games['week'].astype(str) + ' of the ' + games['season'].astype(str) + ' season'
    colors = colors.rename({
        'team_abbr': 'club'
    }, axis=1)
    df = df.merge(colors, on='club', how='left')
    df = df.merge(play_desc, on=['gameId', 'playId'], how='left')
    df = df.merge(games, on=['gameId'], how='left')
    app = NFLPlayApp(df)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()