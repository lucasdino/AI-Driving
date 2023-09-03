from collections import namedtuple, deque
import random
from math import floor
import os
import csv
import datetime

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
    
    def save_replay_memory(self, state_size):
        """Method to save replay memory so that it can be loaded into a training run"""
        formatted_datetime = datetime.datetime.now().strftime("%m.%d.%y-%H.%M")
        with open(f'assets/loss_results/replay_memory-{formatted_datetime}.csv', "w", newline="") as csvfile:
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
        action_index = 1
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
    

def early_training_random_action(action_space_size, last_action, steps_done, pretrain_steps):
    """Function provides random motion to car, biased toward going straight so that early training is seeded with information about motion / reward function"""
    rand = random.random()
    action_list = [0]*action_space_size
    action_index = 0

    # Update biased_action so that each potential action is shown to the model multiple times in early training
    biased_action = floor((steps_done/pretrain_steps)*action_space_size)
    # biased_action = 1

    # Similar logic as other random_action function, except this one will more heavily bias going forward
    if last_action == None:
        action_index = biased_action
    else:
        prev_action_index = last_action.index(1)

        if rand < 0.8:
            action_index = biased_action
        elif rand < 0.9:
            action_index = max(prev_action_index-1, 0)
        else:
            action_index = min(prev_action_index+1, action_space_size-1)
            
    action_list[action_index] = 1
    return action_list
    

def convert_action_tensor_to_list(action_tensor, action_space_size):
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