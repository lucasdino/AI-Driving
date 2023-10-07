import random
import os
import csv
import datetime
from collections import namedtuple, deque

import torch
import torch.nn as nn
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
    
    def save_replay_memory(self, state_size):
        """Method to save replay memory so that it can be loaded into a training run"""
        formatted_datetime = datetime.datetime.now().strftime("%m.%d.%y-%H.%M")
        with open(f'assets/replay_memory/replay_memory-{formatted_datetime}.csv', "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            for memory in self.memory:
                # Unpack the namedtuple into its components
                state, action, next_state, reward = memory
                
                # Convert tensors to CPU and then to lists
                state_list = state.cpu().numpy().tolist()[0]
                action_list = action.cpu().numpy().tolist()[0]
                next_state_list = next_state.cpu().numpy().tolist()[0] if not(next_state == None) else [0]*state_size
                reward_list = [reward.item()]  # Assuming reward is a scalar
                
                # Flatten lists and write to CSV
                csv_writer.writerow(state_list + action_list + next_state_list + reward_list)

        
    def load_replay_memory(self):
        """Class to load replay memory from saved files"""
        pass


class DQN(nn.Module):
    """Create instance of neural net"""

    def __init__(self, n_observations, n_actions):
        # Taking inspiration from 'SWIFT - drone racing paper' (https://www.nature.com/articles/s41586-023-06419-4)
        # Tldr; if they can create world-class drone racing performance (more complex than this and in real-world) on 2 layers of 128 neurons, I make car drive smart w/ small NN :)  

        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """Forward pass of the neural network"""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    def export_parameters(self, filename):
        directory = './assets/nn_params/'
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename) 
        try:
            torch.save(self.state_dict(), filepath)
            print("Model parameters exported successfully.")
        except:
            print("Model parameters failed to export.")