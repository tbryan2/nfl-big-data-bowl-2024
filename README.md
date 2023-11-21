# Situationally Weighted Agression Reinforcement Map (SWARM)

## Working Concept
SWARM is a composite score of two elements:

- The influence of a tackle on the outcome of a game; WPA for a tackle made at a given x-coordinate
- The probability of first impact occuring at a given point on the field; this probability is based on a multivariate Gaussian regression model

Here are some visual breakdowns of the components of SWARM: 
<img width="688" alt="image" src="https://github.com/tbryan2/nfl-big-data-bowl-2024/assets/29851231/1a208e21-4674-44b2-ba57-f2f6aa3f0871">
<img width="688" alt="image" src="https://github.com/tbryan2/nfl-big-data-bowl-2024/assets/29851231/b8bafe24-9041-458f-8df1-7ceb90443dd7">
<img width="687" alt="image" src="https://github.com/tbryan2/nfl-big-data-bowl-2024/assets/29851231/8e9aeeb8-9029-4caa-9cd1-9e317466a617">

This concept can be visualized as a three-dimensional, multivariate, Gaussian distribution where x and y are the coordinates and the z-axis represents the composite SWARM:

![Worth-distribution-as-a-Gaussian-mixture-model](https://github.com/tbryan2/nfl-big-data-bowl-2024/assets/29851231/cf6122cf-67ac-46b8-8dbf-c24429a33759)

Football is a complex system - each player's actions cascade effects through to the other 21 players on the field. __Reinforcement learning__ is a great application for football because of the constant flux of the environment. SWARM provides a great approximation of reality for a reinforcement agent to learn how to play this game. Through calibrating the agent on real NFL plays to maximize SWARM, we can answer fundamental questions about the nature of tackling in football:

- Which defenders take the most optimal paths to the defender?
- Which defenses make the most high influence, low probability tackles?
- Defenders often _appear_ uninvolved in a play (ex. a deep safety on an inside run play). How do we measure their impact on the play? Which safeties are more agressive than the optimal RL agent (maximizing SWARM tackle probability by running at the ball carrier)? Which safeties are more conservative than the optimal RL agent (camping the first down marker or endzone)?

## Play Visualizer Usage

```bash
python animate.py
```

# Files

Files hosted on Digital Ocean.

Tracking data:

https://bigdatabowl2023.nyc3.cdn.digitaloceanspaces.com/raw/tracking_data/tracking_week_1.csv

Team colors:

https://bigdatabowl2023.nyc3.cdn.digitaloceanspaces.com/raw/colors.csv

Games:

https://bigdatabowl2023.nyc3.cdn.digitaloceanspaces.com/raw/games.csv

Players:

https://bigdatabowl2023.nyc3.cdn.digitaloceanspaces.com/raw/players.csv

Plays:

https://bigdatabowl2023.nyc3.cdn.digitaloceanspaces.com/raw/plays.csv

Tackles:

https://bigdatabowl2023.nyc3.cdn.digitaloceanspaces.com/raw/tackles.csv
