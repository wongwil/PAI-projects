"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, DotProduct, Sum, ConstantKernel, WhiteKernel

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        self.x_arr = []
        self.f_arr = []
        self.v_arr = []

        f_kernel = Matern(length_scale=1.0, nu=2.5)

        v_kernel = ConstantKernel(4) + DotProduct(sigma_0=0.0) * Matern(length_scale=1.0, nu=2.5)

        # prior mean at 4, so adding a constant kernel

        self.objective = GaussianProcessRegressor(kernel=f_kernel, n_restarts_optimizer=10, normalize_y=False, optimizer=None, random_state=42, alpha=0.15**2)
        self.constraint = GaussianProcessRegressor(kernel=v_kernel, n_restarts_optimizer=10, normalize_y=False, optimizer=None, random_state=42, alpha=0.0001**2)


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        if len(self.x_arr) == 0:
            print('please sample initial datapoint first')
            raise InterruptedError
        
        # get our full sampled data
        np_x_arr = np.array(self.x_arr).reshape(len(self.x_arr), -1)
        np_f_arr = np.array(self.f_arr).reshape(len(self.f_arr), -1)
        np_v_arr = np.array(self.v_arr).reshape(len(self.v_arr), -1)


        # fit model
        self.objective.fit(np_x_arr, np_f_arr)
        self.constraint.fit(np_x_arr, np_v_arr)

        #print('model fit!')

        return self.optimize_acquisition_function()

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        
        # get our estimates and std
        mu_f, std_f = self.objective.predict(x, return_std=True)
        mu_v, std_v = self.constraint.predict(x, return_std=True)
        

        # upper confidence bound https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/lecture_notes/12.pdf

        lambda_val = 1
        threshold_par = 3.5
        punishment_lambda = 2

        if mu_v[0] + threshold_par*std_v[0] >= SAFETY_THRESHOLD:
            #print('dangerous eval!')
            return mu_f - (mu_v[0]+ threshold_par*std_v[0] - SAFETY_THRESHOLD)*punishment_lambda
        
            # if we are larger than the safety_threshold we punish our mu_f by substracting the amount
            # of the amount that it is above the threshold with a regulizer lambda
            # TODO: maybe optimize or find better regularizer
        else:
            return mu_f + lambda_val*std_f[0]
            # if we are not above the safety_threshold, we can just return the prediction of f
             

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        self.x_arr.append(x)
        self.f_arr.append(f)
        self.v_arr.append(v)

        #print('successfully added datapoints!')

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        np_x_arr = np.array(self.x_arr).reshape(len(self.x_arr), -1)
        np_f_arr = np.array(self.f_arr).reshape(len(self.f_arr), -1)
        np_v_arr = np.array(self.v_arr).reshape(len(self.v_arr), -1)
        passed_ids =  np_v_arr < SAFETY_THRESHOLD

        passed_f = np_f_arr[passed_ids] # get the ones that are not above safety threshold

        highest_id = np.argmax(passed_f) # get the BEST id

        x_opt = np_x_arr[highest_id]

        return x_opt # return best param

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        # REMOVED THIS SINCE WE RETURN A FLOAT AS RECOMMENDED???

        #assert x.shape == (1, DOMAIN.shape[0]), \
            #f"The function next recommendation must return a numpy array of " \
            #f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
