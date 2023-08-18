from collections import namedtuple, deque
import random
import matplotlib
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    """Set up Replay Memory object"""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""

        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """Create instance of neural net"""

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """
        Called with either one element to determine next action, or a batch
        during optimization. Returns tensor([[left0exp,right0exp]...]).    
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    def export_parameters(self, filename):
        directory = '/assets/nn_params/'
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        torch.save(self.policy_net.state_dict(), filepath)
        print("Model parameters exported successfully.")
 

def random_motion(action_space_size):
    """Funciton provides random motion to car"""
    no_action_threshold = 0.9
    rand = random.random()
    action_list = [0]*action_space_size
    
    # Biased toward not doing 'nothing'
    if rand > no_action_threshold:
        action_list[0] = 1
        return action_list
    # Output to go in some direction
    else:
        action_list[random.randint(1,len(action_list))] = 1
        return action_list