"""
This file contains the Pendulum environment, as well as couple of useful functions,
which you can use for the assignment.

IMPORTANT NOTE: CHANGING THIS FILE OR YOUR LOCAL EVALUATION MIGHT NOT WORK. CHANGING THIS FILE WON'T
AFFECT YOUR SUBMISSION RESULT IN THE CHECKER. 

"""
from typing import Optional

import numpy as np
from gym.envs.classic_control import PendulumEnv
import torch
import random
from collections import deque
from gym.wrappers.rescale_action import RescaleAction
from gym.wrappers.time_limit import TimeLimit

class CustomPendulum(PendulumEnv):
    def __init__(self, g: float = 10.0, eps: float = 0.0, *args, **kwargs):
        super().__init__(g=g, *args, **kwargs)
        self.eps = eps

    def reset(self, *, seed: Optional[int] = 0):
        super().reset(seed=seed)
        eps = self.eps
        high = np.asarray([np.pi + eps, eps])
        low = np.asarray([np.pi - eps, -eps])
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}
    
class ReplayBuffer():
    '''
    This class implements a replay buffer for storing transitions. Upon every transition, 
    it saves data into a buffer for later learning, which is later sampled for training the agent.
    '''
    def __init__(self, min_size, max_size, device):
        self.buffer = deque(maxlen=max_size)
        self.device = device
        self.min_size = min_size

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst = [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime = transition
            s_lst.append(s)
            a_lst.append([a.item()])
            r_lst.append([r])
            s_prime_lst.append(s_prime)

        s_batch = torch.tensor(s_lst, dtype=torch.float, device = self.device)
        a_batch = torch.tensor(a_lst, dtype=torch.float, device = self.device)
        r_batch = torch.tensor(r_lst, dtype=torch.float, device = self.device)
        s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float, device = self.device)

        # Normalize rewards
        r_batch = (r_batch - r_batch.mean()) / (r_batch.std() + 1e-7)

        return s_batch, a_batch, r_batch, s_prime_batch

    def size(self):
        return len(self.buffer)

    def start_training(self):
        # Training starts when the buffer collected enough training data.
        return self.size() >= self.min_size


def get_env(g=10.0, train=True):
    '''
    This function sets the environment for the agent.
    :param g: gravity acceleration
    :param train: whether the training or test environment is needed

    Returns:
    :return: The environment.
    '''
    eps = 0.1 if train else 0.0
    env = TimeLimit(RescaleAction(CustomPendulum(render_mode='rgb_array', g=g, eps=eps),
                                  min_action=-1, max_action=1), max_episode_steps=200)
    return env


def run_episode(env, agent, rec=None, verbose=False, train=True):
    '''
    This function runs one episode of the environment with the agent.
    Until the episode is not finished (200 steps), it samples and performs an action,
    stores the transition in the buffer and if the training is started, it also performs
    a training step for the agent.
    
    :param env: The environment to run the episode on.
    :param agent: The agent to use for the episode.
    :param rec: The video recorder to use for recording the episode, if any.
    :param verbose: Whether to print the episode return and mode.
    :param train: Whether to train the agent.

    Returns:
    :return: The episode return.
    '''
    mode = "TRAIN" if train else "TEST"
    state, _ = env.reset()
    episode_return, truncated = 0.0, False
    while not truncated:
        action = agent.get_action(state, train)

        state_prime, reward, _, truncated, _ = env.step(action)

        if train:
            agent.memory.put((state, action, reward, state_prime))
            if agent.memory.start_training():
                agent.train_agent()
        else:
            if rec is not None:
                rec.capture_frame()

        episode_return += reward
        state = state_prime

    if verbose:
        print("MODE: {}, RETURN: {:.1f}".format(mode, episode_return))

    return episode_return
