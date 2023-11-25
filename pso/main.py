import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde, multivariate_normal
from matplotlib import pyplot as plt
# data preprocessing

# objective functions
def kde_obj(
        positions, 
        kde_interpolator, 
        gaussian_interpolator, 
        gaussian_weight=0.4
    ) -> float:

    positions = np.atleast_2d(positions)
    print('Positions: ', positions)
    kde_values = kde_interpolator(positions)
    gaussian_values = gaussian_interpolator(positions)
    objective_values = kde_values + -1*gaussian_weight * gaussian_values

    return -objective_values
    
class PSODefense:
    def __init__(self, frame: pd.DataFrame, objective_function):

        self.objective_function = objective_function
        self.frame = frame
        self.xmin = 0
        self.xmax = 120  # Including endzones
        self.ymax = 53.3  # Standard width of a football field
        self.ymin = 0
        self.min_velocity = -0.5
        self.max_velocity = 0.5

        defense = frame.loc[frame['club'] == 'SF']

        self.num_particles = len(defense)
        self.num_dimensions = 2

        self.positions = defense[['x', 'y']].values
        self.velocities = defense[['x_velocity', 'y_velocity']].values

        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(len(defense), np.inf)

        self.global_best_score = np.inf
        self.global_best_position = None

        # initialize the objective function parameters
        self.objective_function_params = self.init_objective_function_params()

        # hyper parameters
        self.w = 1
        self.c1 = 2
        self.c2 = 2
        self.num_iterations = 1

    def init_objective_function_params(self):
        offense = self.frame.loc[(self.frame['club'] == 'CHI') & (self.frame['nflId'] != 53646)]
        x, y = offense['x'].values, offense['y'].values
        data = np.vstack((x, y))
        
        gaussian_kernel = gaussian_kde(data)
        xx, yy = np.mgrid[self.xmin:self.xmax:100j, self.ymin:self.ymax:100j]

        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(gaussian_kernel(positions).T, xx.shape)
        kde_interpolator = RegularGridInterpolator((xx[:, 0], yy[0, :]), f)

        ball_carrier = self.frame.loc[(self.frame['club'] == 'CHI') & (self.frame['nflId'] == 53646)]
        ball_carrier_mean = ball_carrier[['x', 'y']].values[0]
        ball_carrier_std_dev = 1
        ball_carrier_covariance = np.diag([ball_carrier_std_dev**2, ball_carrier_std_dev**2])

        pos = np.dstack((xx, yy))
        rv = multivariate_normal(ball_carrier_mean, ball_carrier_covariance)
        f = rv.pdf(pos)
        gaussian_interpolator = RegularGridInterpolator((xx[:, 0], yy[0, :]), f)

        return {
            'kde_interpolator': kde_interpolator,
            'gaussian_interpolator': gaussian_interpolator,
            'gaussian_weight': 0.5
        }

    def optimize(self):
        for _ in range(self.num_iterations):
            fitness = self.objective_function(self.positions, **self.objective_function_params)

            for i in range(self.num_particles):
                if fitness[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness[i]
                    self.personal_best_positions[i] = self.positions[i]
                
                if fitness[i] < self.global_best_score:
                    self.global_best_score = fitness[i]
                    self.global_best_position = self.positions[i]

            for i in range(self.num_particles):
                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + self.c1 * np.random.rand() * (self.personal_best_positions[i] - self.positions[i])
                    + self.c2 * np.random.rand() * (self.global_best_position - self.positions[i])
                )

                self.velocities[i] = np.clip(self.velocities[i], self.min_velocity, self.max_velocity)

                self.positions[i] += self.velocities[i]
                self.positions = np.clip(self.positions, [self.xmin, self.ymin], [self.xmax, self.ymax])

        return self.global_best_score, self.global_best_position
    
if __name__ == '__main__':
    df = pd.read_csv('example_play.csv')
    frame = df.loc[df['frameId'] == 20]
    pso = PSODefense(
        frame=frame,
        objective_function=kde_obj
    )
    # optimize
    global_best_score, global_best_position = pso.optimize()
    new_velocities = pso.velocities
    new_positions = pso.positions

    fig, ax = plt.subplots(figsize=(10, 6))

    offense = frame.loc[frame['club'] == 'CHI']
    defense = frame.loc[frame['club'] == 'SF']

    ax.scatter(offense['x'], offense['y'], c='blue', label='Offense')
    ax.scatter(defense['x'], defense['y'], c='red', label='Defense')
    ax.scatter(global_best_position[0], global_best_position[1], c='yellow', label='Global Best')
    # plot new positions
    ax.scatter(new_positions[:, 0], new_positions[:, 1], c='green', label='New Positions')
    plt.show()

