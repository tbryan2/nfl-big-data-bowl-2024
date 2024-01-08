# A Biomimetic Approach to the NFL Big Data Bowl: Charting the Future of NFL Secondaries with Swarm Intelligence and PASTA 🍝

## Introduction

<div style="display: flex; flex-direction: row; justify-content: center;">
    <img src="https://i.imgur.com/NWjojAA.gif">
    <img style="height: 250px; width: auto;" src="https://i.imgur.com/bHipbq7.png"/>
</div>

> "That's just our mentality and tackling -- it's a swarm mentality where we want to get as many guys to the ball as possible" - DeMeco Ryans, 2022 [source](https://www.youtube.com/watch?app=desktop&v=I6sNeLW_-EE&ab_channel=SanFrancisco49ers)

**DeMeco Ryans**, recognized as the 2022 Assistant Coach of the Year for his role as the 49ers Defensive Coordinator, has championed the concept of swarming to the ball. This philosophy, which [continues to resonate](https://www.houstontexans.com/video/head-coach-demeco-ryans-we-talked-about-a-swarm-mentality-and-that-s-what-we-did) with Ryans in his current position as head coach of the Houston Texans, emphasizes the importance of players converging on a ball carrier with intensity. In the figure above, you can see how his defensive secondary closely followed the optimal paths suggested by our methodology. 

- **Guiding Question:** How can we quantify defensive secondaries' success in limiting YAC via swarm tackling?
- **Data:** NextGenStats tracking data from weeks 1-9 in the 2022 season merged with play-by-play data and model outputs from [nflfastR](https://www.nflfastr.com/index.html). 

Leveraging advanced analytics and biomimetics, we introduce a new metric called **PASTA (Path Analysis via Swarm-Tackle Accuracy)** that aims to answer our guiding question using swarm intelligence.


# Mathematical Representation of Swarm Tendencies in a Football Context

Our submission proposes a new metric named PASTA, derived from Particle Swarm Optimization, to evaluate and quantify defensive secondaries effectiveness at swarming ball carriers. 
Particle Swarm Optimization (PSO)$^1$ is a computational method for finding optimal paths, leveraging a social component where multiple agents (analogous to a flock of birds or a school of fish, or in this case, a defensive unit) share information to collectively navigate towards the best solutions in complex spaces. This characteristic makes it highly suitable for addressing multi-agent problems where coordination and cooperation are essential. In a football context, each player adjusts their position and strategy relative to others, guided by simple yet effective rules. The collective aim (objective function) in our case is to minimize the distance to some swarm point $T$. 

Guided by DeMeco Ryans' defensive philosophy, our hypothesis is that a successful tackle, achieved by minimizing the total distance of secondary players from the ball carrier immediately after a catch, effectively restricts YAC through efficiently swarming to the ball. 

### 1. Objective Function

Particle Swarm Optimization (PSO) begins with the definition of an objective function, which serves as a guiding metric for the optimization process. This objective function quantitatively evaluates the quality or suitability of a solution within the problem space. Our objective function is simple:

$minimize ||P_i - T||$

Here, $T$ represents the target with position vector components $x_t$ and $y_t$ and $P_i$ is the position of the $i$-th particle, also described as a position vector with components $x^p_i$ and $y^p_i$. The operation $||P_i - T||$ denotes the Euclidean Distance between the particle's position $P_i$ and the target $T$, which we aim to minimize.

While the objective function is simple enough to formulate, the effectiveness of our function relies heavily on the selection of the target $T$, or swarm point, which is described below.

### 2. Target Selection 

To define our swarm point, we first define our search space as each of the ball carriers position following receipt of the football. Then, we narrow our search space to locations where all agents could reach in a straight line given some reasonable minimum velocity, $ \vec{V}_{\text{min}}$. From there, we find the best swarm point weighted on the $\theta$ defenders will have to take to reach $T$ and the distance to $T$, denoted by $d$, for each defender.

In more robust terms:

Let $ \vec{P}_i $ be the position vector of the $ i $-th defensive player, and $ \vec{T}_j $ be the position vector of the $ j $-th potential tackle point along the ball carrier's path. Define the minimum velocity requirement as $ \vec{V}_{\text{min}} = (V_{\text{min}_x}, V_{\text{min}_y}) $. The objective is to find the tackle point $ \vec{T}_{\text{best}} $ that minimizes the total score $ S $, subject to velocity and angle constraints.

The total score $ S $ for a tackle point $ \vec{T}_j $ is given by:

$$
S(\vec{T}_j) = \sum_{i=1}^{N} \left( w_d \cdot d(\vec{P}_i, \vec{T}_j) + w_{\theta} \cdot |\theta(\vec{P}_i, \vec{T}_j)| \right)
$$

where:
- $ d(\vec{P}_i, \vec{T}_j) = \vec{T}_j - \vec{P}_i $ is the Euclidean distance between defensive player $ i $ and tackle point $ j $.
- $ \theta(\vec{P}_i, \vec{T}_j) $ is the angle between the defensive player's position vector and the tackle point, calculated as $ \arctan2(T_{jy} - P_{iy}, T_{jx} - P_{ix}) $, with the constraint that $ -\theta_{\text{max}} \leq \theta \leq \theta_{\text{max}} $.
- $ w_d $ and $ w_{\theta} $ are weighting factors for distance and angle, respectively.

The constraint which narrows our search space for each defensive player-tackle point pair is:

$$
|T_{jx} - P_{ix}| \leq V_{\text{min}_x} \cdot j \quad \text{and} \quad |T_{jy} - P_{iy}| \leq V_{\text{min}_y} \cdot j
$$

If this constraint is violated, the score for that defensive player-tackle point pair is set to infinity, effectively removing it from the search space.

Finally, the swarm point is chosen as:

$$
\vec{T}_{\text{best}} = \underset{\vec{T}_j}{\mathrm{argmin}}\, S(\vec{T}_j)
$$

#### Model Parameters

* *When running our model, we set $ \vec{V}_{\text{min}} = \vec{v}_{\text{avg}} = \frac{1}{n} \sum_{i=1}^{n} \vec{v}_i$, where $\vec{v}_{\text{avg}}$ represents the average velocity of the ball carrier.*

### 3. Particle Swarm Optimization 

Once the objective function and target are defined, we can run PSO. For any given play, we initialize a swarm of particles, each denoted as $ P_i $ for the i-th particle, representing a defensive player's actual position on the field.

As the play progresses following a ball carrier receiving the football, each particle updates its velocity and position based on a combination of its own best-known position $ \vec{p}_{best,i} $, the best-known positions $ \vec{p}_{best,neigh} $ of its neighboring particles, and the best-known position $ \vec{g}_{best} $ of the entire swarm. These updates are mathematically represented as:

$ \vec{v}_i^{(\text{new})} = w \cdot \vec{v}_i + \phi_1 \cdot \text{rand}_1 \cdot (\vec{p}_{\text{best},i} - \vec{x}_i) + \phi_2 \cdot \text{rand}_2 \cdot (\vec{g}_{\text{best}} - \vec{x}_i)$

$ \vec{x}_i^{(new)} = \vec{x}_i + \vec{v}_i^{(new)} $

Where:

* $ w \text{ represents the inertia factor, influencing how much of the previous velocity is retained in the new velocity calculation.} $

* $\vec{v}_i \text{ is the current velocity of the i-th particle.}$

* $\phi_1 \text{ and } \phi_2 \text{ are cognitive and social weighting factors, respectively.}$

* $\text{rand}_1 \text{ and } \text{rand}_2 \text{ are random numbers in the range } [0,1].$

* $\vec{p}_{\text{best},i} \text{ is the best-known position of the i-th particle.}$

* $\vec{g}_{\text{best}} \text{ is the best-known position of the entire swarm.}$

* $\vec{x}_i \text{ is the current position of the i-th particle.}$

Once the updates to the velocities and positions of the particles are made the next crucial step is to evaluate the objective function for each particle. This is done by assessing how close each particle's current position $ \vec{x}_i $ is to the target $ T $, using the objective function $ ||P_i - T|| $.

After evaluating the objective function, each particle's personal best position $ \vec{p}_{best,i} $ is updated if the current position yields a better (lower) value of the objective function than previously recorded. Similarly, the best-known position of the swarm $ \vec{g}_{best} $ is updated if any particle discovers a position that is closer to the target than any previously known positions.

This cycle of updating velocities and positions, followed by evaluating the objective function and updating the personal and global bests, continues iteratively, until a stopping threshold is reached.


![Imgur](https://i.imgur.com/H2320AU.png)

#### Model Parameters

* *When running our model, we clip velocity using the following: $ \vec{v}_i^{(\text{new})} = 
\begin{cases} 
\vec{v}_i^{(\text{new})} & \text{if } \|\vec{v}_i^{(\text{new})}\| \leq V_{\text{max}} \\
V_{\text{max}} \cdot \frac{\vec{v}_i^{(\text{new})}}{\|\vec{v}_i^{(\text{new})}\|} & \text{if } \|\vec{v}_i^{(\text{new})}\| > V_{\text{max}}
\end{cases}$, where $V_{\text{max}}$ represents the max velocity of the ball carrier.*

* *We set $ \phi_1 = \phi_2$*

* *We set $w$, the inertia factor, equal to the golden ratio, $\frac{1 + \sqrt{5}}{2}$ $^4$.*

### 4. Path Evaluation: Frechet Distance

The Fréchet distance is a measure used to quantify the similarity between two curves, which in our context can represent the actual paths of the defenders and the PSO-suggested path. Let $ \mathcal{P} $ and $ \mathcal{Q} $ represent the actual path of a defender and a PSO-suggested path, respectively. The Fréchet distance between $ \mathcal{P} $ and $ \mathcal{Q} $ is defined as the minimum "leash length" required to connect a point moving along $ \mathcal{P} $ and another point moving along $ \mathcal{Q} $, such that both points traverse their respective paths from start to finish.

Formally, the Fréchet distance $ F(\mathcal{P}, \mathcal{Q}) $ can be defined as:

$$
F(\mathcal{P}, \mathcal{Q}) = \inf_{\alpha, \beta}\max_{t \in [0,1]} \left\| \mathcal{P}(\alpha(t)) - \mathcal{Q}(\beta(t)) \right\|
$$

Here, $ \alpha(t) $ and $ \beta(t) $ are continuous non-decreasing functions mapping the interval $ [0,1] $ onto the paths $ \mathcal{P} $ and $ \mathcal{Q} $, respectively. The distance at any point in time $ t $ is given by the Euclidean distance $ \left\| \mathcal{P}(\alpha(t)) - \mathcal{Q}(\beta(t)) \right\| $. The Fréchet distance is the infimum of these distances over all possible mappings $ \alpha $ and $ \beta $.

In our submission, we use fretchet distance as our definition of error for defenders' paths.

Let's take a look at how this frechet distance correlates with a defensive secondary's ability to limit YAC. Note: YAC under Expected (YUX) is calculated by subtracting actual YAC from Expected YAC $ ^2 $. 

![pair plot](https://i.imgur.com/y1QIk5s.png)

After aggregating the merged data to the player level, we were interested in the relationship between the frechet distance in yards and different yac-related metrics. As you can see, Frechet Distance correlates negatively with average yards after catch difference. Additionally, you can see that cornerbacks typically have the largest frechet distance. This is intuitive as they likely have more plays where they traverse the entire width of the field, leaving more room for error in deviating from the optimal path.

![correlation info](https://i.imgur.com/8lXHXel.png)

In this figure, we see that there is a moderately strong correlation between following the PSO-suggested path and YAC under Expected. **This is the basis for our PASTA metric calculation.** 

### 5. Path-Analysis via Swarm Tackle Accuracy (PASTA) Calculation

To calculate PASTA, use frechet distance in yards as the denominator to quantify the difference between the defenders actual path and the PSO-suggested path. The numerator, is based on how many yards after catch (YAC) were given up versus expected.

**The quotient of these two becomes PASTA**. A higher PASTA value could indicate that the a defenders close promixity to the optimal path resulted in YAC under expected.

$$ \text{Yards After Catch Under Expected (YUX)} = \text{Expected Yards After Catch} - \text{Actual Yards After Catch} $$

$$ \text{PASTA} = \frac{\text{YUX}}{\text{frechet distance per player}} $$

# 2022 PASTA Rankings by Team and Player

**For the purposes of our submission, we reduced the problem space to evaluating secondaries' ability to swarm the ball carrier in pass-catching scenarios using our proposed metric PASTA.**

![PASTA Team Rankings](https://i.imgur.com/07jzctA.png)


![PASTA Player Rankings](https://i.imgur.com/h3zHXzp.png)

Remeber, the YUX/play metric is a valuable team-level measure, emphasizing a defense's capability to limit yards after catch. Standout players such as Jaylinn Hopkins contribute significantly to this aspect. While they may not always be directly involved in the tackle, their adherence to the PSO-suggested paths bolsters the team's ability to swiftly converge on the receiver. This strategic positioning underscores their role in reinforcing an effective and cohesive defensive unit.

# Discussion and Applications
PASTA offers vast applications for the NFL as whole, individual teams, the media, and fans.

**Player Evaluation:**
- **Unexpected YAC Assignment:** A large PASTA value could indicate the DB's close proximity to the optimal path generated YAC under expected. While a small or negative PASTA value could indicate suboptimal pursuit of the ball carrier.
- **Synergy in the Secondary:** Locating secondary units or player groupings that work better together to limit YAC.

**Coaching Applications:**
- **Teachable Moments:** Identifying plays with significant deviations from optimal paths and using them in film study. 
- **Reinforcement of Successful Strategies:** Highlighting plays where the secondary closely followed the optimal path ot everyone hustled after getting off the optimal early. 

**Broadcasting and Rankings:**
- **On-field Graphics:** Demonstrating defensive prowess through proximity to "Biomimetic" or "AI" Path on field graphic
- **Player and Defense Rankings:** Showcasing how talented players and teams align with optimal paths.

# Limitations
While PASTA is promising, it has areas for growth:
- **Max Speed Limitation:** A constraint that may not fully mirror real-life scenarios but is still effective. Given more data and time, we could fine tune this to constrain more specifically to individual player profiles. 
- **Target Selection:** Focusing on refining how targets are selected and pursued by the defense. Ideally, we would like to incorporate ideas from this paper $^3$ that utilizes particle swarm optimization on moving targets.

# Ongoing Development
- **Generalizing to Other Groups:** It would be interesting to analyze different position units, entire teams, or relevant combinations of players on different types of plays and different types of expected yards models. 
- **Covering Potential Receivers:** Exploring PASTA's application to broader defensive scenarios that are not specifically tackle-related, such as secondaries converging on a reciever in anticipation of an incoming pass.
- **Incorporating Other Models:** Considering integration with other distance minimization models like [STRAIN](https://www.kaggle.com/code/statsinthewild/strain-sacks-tackles-rushing-aggression-index) for enhanced analysis and perhaps better path optimization. (Potential name for combined implementation : *PASTA STRAINER* )

# Conclusion
PASTA, drawing inspiration from natural efficiency and tailored for football, is poised to revolutionize NFL defensive strategies. Its capacity to assess individual players and units, particularly in their ability to "swarm" to the ball, positions it as a versatile and potent tool in strategic defense planning. This metric provides not just tactical insights for coaches but also engaging analytical content for fans and broadcasters. We are excited to continue develo

# Citations
1. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. In Proceedings of the International Conference on Neural Networks (ICNN'95), Perth, WA, Australia (Vol. 4, pp. 1942-1948). doi: 10.1109/ICNN.1995.488968.

2. Baldwin, B. (2021, Feb. 5). Open Source Football: nflfastR EP, WP, CP xYAC, and xPass models. Retrieved from https://www.opensourcefootball.com/posts/2020-09-28-nflfastr-ep-wp-and-cp-models/

3. Phung, M. D., & Ha, Q. P. (2020). Motion-encoded particle swarm optimization for moving target search using UAVs. Applied Soft Computing, 97(Part B), 106705. ISSN 1568-4946.

4. Xiao, J., & Zhuang, Y. (2012). The Comparison Between Particle Swarm Optimization and K-Means Algorithms in Data Clustering. In Proceedings of the 2012 International Conference on Industrial Control and Electronics Engineering. doi: 10.1109/ICICEE.2012.310.

## Where to find us: 

Nick Gurol
- Twitter: [@run_the_sims](https://twitter.com/run_the_sims)
- LinkedIn: [Nick Gurol](www.linkedin.com/in/nick-gurol-87738a1b7)

Tim Bryan
- Twitter: [@TimBryan000](https://x.com/timbryan000)
- LinkedIn: [Tim Bryan](https://www.linkedin.com/in/timothy-bryan-384360118/)

Ben Dominguez
- Twitter: [@bendominguez011](https://x.com/bendominguez011)

Ben Wolbransky
- Twitter: [@benwolby](https://x.com/benwolby)