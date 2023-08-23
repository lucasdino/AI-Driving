from collections import namedtuple, deque
import random
import matplotlib
import matplotlib.pyplot as plt
import os
import csv

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
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, 64)
        self.layer5 = nn.Linear(64, n_actions)

    def forward(self, x):
        """
        Called with either one element to determine next action, or a batch
        during optimization. Returns tensor of size of action set    
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)
    
    def export_parameters(self, filename):
        directory = './assets/nn_params/'
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename) 
        try:
            torch.save(self.state_dict(), filepath)
            print("Model parameters exported successfully.")
        except:
            print("Model parameters failed to export.")
    

def instantiate_hardware():
    """Function to load CUDA if GPU is available; otherwise use CPU""" 
    hardware = "tbu"
    num_episodes = 0
    if torch.cuda.is_available():
        hardware = "cuda"
        num_episodes = 5000
    else:
        hardware = "cpu"
        num_episodes = 100

    print(f"Utilizing {hardware} for neural net hardware.")
    return torch.device(hardware), num_episodes
 

def random_action_motion(action_space_size, last_action):
    """Function provides random motion to car"""
    rand = random.random()
    action_list = [0]*action_space_size
    action_index = 0

    # Output to go in some direction; bias going forward (Up is index 2, but adding 1 to everything for quick iteration / prototyping)
    if last_action == None:        
        action_index = 2
    else:
        prev_action_index = last_action.index(1)

        # Bias toward having car do the same action as before. Prevents super wobbly behavior. Otherwise add 1 / subtract 1 (min/max to prevent OOB)
        if rand < 0.9:
            action_index = prev_action_index
        elif rand < 0.95:
            action_index = max(prev_action_index-1, 0)
        else:
            action_index = min(prev_action_index+1, action_space_size-1)
        
    action_list[action_index] = 1
    return action_list
    

def convert_action_tensor(action_tensor, action_space_size):
    """Model that takes in an action tensor and returns a list that can be interpreted by the game"""
    inferred_action = [0] * action_space_size
    inferred_action[action_tensor.item()] = 1
    return inferred_action


def save_loss_to_csv(loss_calculations, filename):
    """Saves the loss to a CSV file."""
    with open(f'assets/loss_results/{filename}.csv', "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        for loss in loss_calculations:
            csv_writer.writerow([loss])