import torch
import torch.optim as optim
from torch.distributions import Normal, Independent
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union, Tuple
from utils import ReplayBuffer, get_env, run_episode

''' We mostly followed the pseudo code of OpenAIs version of
     SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html
    (We avoid the approximation for the value function)
    Note: We also use cuda if there is one GPU available.
'''



warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str):
        super(NeuralNetwork, self).__init__()

        # TODO: Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.

        activation_dict = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh()
        }

        activation_f = activation_dict.get(activation)
        layers = [nn.Linear(input_dim, hidden_size), activation_f]
        # only input layer first

        for i in range(hidden_layers):
            layers.extend([nn.Linear(hidden_size, hidden_size), activation_f]) # add hidden layers 
        
        # ouptut layer (without activation)
        layers.append(nn.Linear(hidden_size, output_dim))
        self.net = nn.Sequential(*layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        return self.net(s).to(self.device)
    
class Actor:
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        # TODO: Implement this function which sets up the actor network. 
        # Take a look at the NeuralNetwork class in utils.py. 

        # we use 2*self.action_dim as output dimension because we want to have the mean and std as output 
        self.policy_net = NeuralNetwork(self.state_dim, 2*self.action_dim, self.hidden_size, self.hidden_layers, 'relu')
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.actor_lr)

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor, deterministic: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        action , log_prob = torch.zeros(state.shape[0]), torch.ones(state.shape[0])
        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.

        output = self.policy_net.forward(state).to(self.device)
        #print(f'output shape: {output.shape}')
        
        # split output into two tensors: mean&log so we can sample
        means = output[:, 0] # let the first column be the means
        log_stds = output[:, 1] # let the second column be the 
        
        log_stds = self.clamp_log_std(log_stds) 

        stds = torch.exp(log_stds) # get exponential of log_stds to get stds
        
        #print(f'means shape: {means.shape}')
        #print(f'stds shape: {stds.shape}')

        normal = Normal(means, stds)
        
        # See Part C. Enforcing Action bounds in official SAC-paper.
        
        u = normal.sample() if deterministic else normal.rsample()
        ''' we are using random sampling in either case (non-deterministic) but we
        use the boolean variable as deciding whether we should
        use the reparametrization trick or sample directly. in openAIs version the reparametrization trick
        is only used when updating the policy'''
        
        #print(f'samples shape: {u.shape}')
        
        action = torch.tanh(u) 
        
        epsilon = 1e-6 
        ''' add epsilon term to make the log numerically stable. if action is close to 1, we would
        take the log of 0 which could lead to instabilities.'''
        log_prob = normal.log_prob(u) - torch.log(1 - action.pow(2) + epsilon) 
        # Note: we can leave out the summand over D in equation (21) of the paper, as u is one dimensional
        
        #print(action)
        #print(log_prob)
        
        # reshape returns as in the assert statements
        action = action.view(state.shape[0], self.action_dim)
        log_prob = log_prob.view(state.shape[0], self.action_dim)
    
        assert action.shape == (state.shape[0], self.action_dim) and \
            log_prob.shape == (state.shape[0], self.action_dim), 'Incorrect shape for action or log_prob.'
        return action, log_prob


class Critic:
    def __init__(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: int, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr # learning rate
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        # TODO: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork 
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.

        # we use double Q-learning, so we don't overestimate. we use one for one to estimate Q-val and the other one
        # to evaluate the chosen action
        self.network = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size, self.hidden_layers, 'relu')
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.critic_lr)
        pass

    # my own function
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa_concat = torch.cat([state, action], dim=-1)

        return self.network(sa_concat)
    
    def set_grad(self, requires_grad: bool):
        for param in self.network.parameters():
            param.requires_grad = requires_grad

class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        
        self.setup_agent()

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need.   
        # SAC hyperparams
        self.tau = 0.01 # soft update parameter: small tau => quicker adaption to target, large tau => slower adaption to target
        self.gamma = 0.99 # discount factor: small gamma => focusing on short-term rewards, large gamma => focusing on long-term rewards
        self.alpha = 0.08 # entropy regularizer: small a => exploitation, large a => exploration
        
        # network hyperparams
        hidden_layers = 3 # 4 layers achieved -334.9 on cluster. not sure if I can run it on my own laptop. maybe reduce
        hidden_size = 64
        learning_rate = 0.001
        
        self.actor = Actor(hidden_size=hidden_size, hidden_layers=hidden_layers, actor_lr=learning_rate, state_dim=self.state_dim, action_dim=self.action_dim, device=self.device)
    
        self.q = Critic(hidden_size=hidden_size, hidden_layers=hidden_layers, critic_lr=learning_rate, state_dim=self.state_dim, action_dim=self.action_dim, device=self.device)
        self.q_target = Critic(hidden_size=hidden_size, hidden_layers=hidden_layers, critic_lr=learning_rate, state_dim=self.state_dim, action_dim=self.action_dim, device=self.device)

        self.q2 = Critic(hidden_size=hidden_size, hidden_layers=hidden_layers, critic_lr=learning_rate, state_dim=self.state_dim, action_dim=self.action_dim, device=self.device)
        self.q2_target = Critic(hidden_size=hidden_size, hidden_layers=hidden_layers, critic_lr=learning_rate, state_dim=self.state_dim, action_dim=self.action_dim, device=self.device)

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        #action = np.random.uniform(-1, 1, (1,))
        state = torch.tensor(s).unsqueeze(0).float()
        action, _ = self.actor.get_action_and_log_prob(state.to(self.device), deterministic=False)
        action = action.cpu()
        
        action = action[0].detach().cpu().numpy()
        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork, 
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)

        # Batch sampling
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch
        
        # putting the batches to cuda if available
        s_batch, a_batch, r_batch, s_prime_batch = s_batch.to(device), a_batch.to(device), r_batch.to(device), s_prime_batch.to(device)
        
        
        # TODO: Implement Critic(s) update here.
        
        # 12: compute targets for Q-functions
        with torch.no_grad():
            action, log_prob = self.actor.get_action_and_log_prob(s_prime_batch, deterministic=True)
            target_q1 = self.q_target.forward(s_prime_batch, action)
            target_q2 = self.q2_target.forward(s_prime_batch, action)
            target_q = r_batch + self.gamma * torch.min(target_q1, target_q2) - self.alpha * log_prob

        # 13: update q-functions by calculating the difference between q-networks and target
        q_pred = self.q.forward(s_batch, a_batch)
        q_error = torch.mean((q_pred-target_q).pow(2))
        self.run_gradient_update_step(self.q, q_error)

        q2_pred = self.q2.forward(s_batch, a_batch)
        q2_error = torch.mean((q2_pred-target_q).pow(2))
        self.run_gradient_update_step(self.q2, q2_error)


        # TODO: Implement Policy update here
        # freeze q networks for inference as we only need the policy gradients for the update
        self.q.set_grad(False)
        self.q2.set_grad(False)
        
        action, log_prob = self.actor.get_action_and_log_prob(s_batch, deterministic=False)
        q_val = self.q.forward(s_batch, action)
        q2_val = self.q2.forward(s_batch, action)
        self.q.set_grad(True) # enable them again after inference
        self.q2.set_grad(True)
        
        policy_error = -torch.mean(torch.min(q_val, q2_val) - self.alpha * log_prob) # minus for ascending high q-values
        self.run_gradient_update_step(self.actor, policy_error)
        
        # 15: update target networks 
        with torch.no_grad():
            self.critic_target_update(base_net=self.q.network, target_net=self.q_target.network, tau=self.tau, soft_update=True)
            self.critic_target_update(base_net=self.q2.network, target_net=self.q2_target.network, tau=self.tau, soft_update=True)


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = False
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()
