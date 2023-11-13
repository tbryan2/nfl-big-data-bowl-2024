### Random Notes Here

Notes/Ideas:

Train "12th man" RL agent to become a perfect tackler. Metric is deviance from this path.

GHOST IN THE MACHINE; concept of "ghosts" like in mario kart and other video games

Model from different positions - i.e. agent from defensive lineman initial starting position
    remove impossible scenarios based on distribution of distance covered and speed for example

WHO HAD THE HARDEST CIRCUMSTANCES (IE WHICH STARTING POSITION DOES THE AGENT HAVE THE TOUGHEST TIME WITH)

TODO : 

Define "a tackle" using empirical data - how does the agent know when it's completed its task? Part of the reward function.
gracefully quit the pygame window, also make the pygame window able to go on top of or under other windows like a regular window
#1 priority is defining a reward function after the enviornment is set up 
Modularize the environment - the FootbalPlay class is so huge right now it's hard to work with
PUT PLAYER POSITIONS IN RESET AND OBSERVATION FUNCTIONS, RIGHT NOW THEY DON'T ACTUALLY EXIST AS ANYTHING EXCEPT ANIMATIONS
Fix movement of agent (currently moving whole numbers while every other component is moving at the 0.01 level per frame)

Resources:
https://github.com/DLR-RM/stable-baselines3 # RL algorthims for learning, once the environment is ready
https://huggingface.co/learn/deep-rl-course/unit0/introduction # RL course
https://gymnasium.farama.org/content/basic_usage/ # Gymnasium documentation
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/ # Gymnasium boilerplate custom env