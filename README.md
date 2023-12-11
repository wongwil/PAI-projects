# PAI-projects
My implementation of several projects for the course "Probabilistic AI" at ETHZ in 2023.

# Task 1:  Gaussian Process Regression for Air Pollution Prediction.
According to the World Health Organization, air pollution is a major environmental health issue. Both short- and long-term exposure to polluted air increases the risk of heart and respiratory diseases. Hence, reducing the concentration of particulate matter (PM) in the air is an important task.

Our goal is to help a city predict and audit the concentration of fine particulate matter (PM2.5) per cubic meter of air. In an initial phase, the city has collected preliminary measurements using mobile measurement stations. The goal is to develop a pollution model that can predict the air pollution concentration in locations without measurements. This model will then be used to determine suitable residental areas with low air pollution. The city already determined a couple of candidate locations for new residental areas, based on other relevant parameters such as infrastructure, distance to city center, etc.

A pervasive class of models for weather and meteorology data are Gaussian Processes (GPs). In this project, we use Gaussian Process regression in order to model air pollution and to predict the concentration of PM2.5 at previously unmeasured locations.

# Task 2: Approximate Bayesian inference in neural networks via SWA-Gaussian
In this project, we implement approximate inference via SWA-Gaussian (SWAG) (Maddox et al., 2019), an extension of Stochastic Weight Averaging (SWA) (Izmailov et al., 2018). SWAG is a simple method that stores weight statistics during training, and uses those to fit an approximate Gaussian posterior.
To be more specific, the goal is to implement SWA-Gaussian to classify land-use patterns from satellite images, and detect ambiguous/hard images using the model's predicted confidence. For each test sample, our method has to either output a class (0,1,2,3 etc.), or "don't know". Each prediction is assigned on a cost (this criterion is called the Expected Calibration Error (ECE)).

# Task 3: Hyperparameter tuning with Bayesian Optimization
In this task, we use Bayesian optimization to tune the structural features of a drug candidate, which affects its absorption and distribution. These features should be optimized subject to a constraint on how difficulty the candidate is to synthesize. Let x ∈ X be a parameter that quantifies such structural features. We want to find a candidate with x  that is 1) bioavailable enough to reach its intended target, and 2) easy to synthesize. We use logP as our objective - a coarse proxy for bioavailability. To this end, for a specific  x, we simulate the candidate's corresponding logP as well as its synthetic acessiblity score (SA), which is a proxy for how difficult it is to synthesize. Our goal is to find the structural features x∗ that induce the highest possible logP while satisfying a constraint on synthesizability. We are interested in minimizing the normalized regret for not knowing the best hyperparameter value.
In summary, given a black-box function, our goal is to find optimal parameters with having as few unsafe evaluations as possible. Unsafe evaluation is defined as using the black-box function f at a point x s.t. the function outputs a value over a threshold i.e. f(x) > THRESHOLD.
  
  
# Task 4: Implementing an Off-policy RL algorithm.
In this task, the goal was to implement an off-policy RL algorithm (e.g. DDPG or SAC) to train a agent which will swing up an inverted pendulum from an angle of π (downward position) to 0 and try to hold it there. To swing-up the pendulum, agent has a motor that can apply torques u in range of [−1, 1], i.e., u ∈ [−1, 1].
We implemented a newer version of SAC which omits the approximation for the value function and just used approximators for the policy and the critics. A lot of information such as update rules/gradient steps were taken from the openAI documentation: https://spinningup.openai.com/en/latest/algorithms/sac.html. 
