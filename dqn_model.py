# Direction from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import math
import random
from collections import namedtuple, deque
from itertools import count
from dqn_modules import ReplayMemory, DQN, Transition, instantiate_hardware, random_action_motion, convert_action_tensor, save_loss_to_csv
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class DQN_Model:
    """Class encompassing neural network, including training and inference functions"""

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    # N_ACTIONS is the number of possible actions from the game (n=9: Nothing, Up, Left, Right, Down, Up-Left, Up-Right, Down-Left, Down-Right)
    # N_OBSERVATIONS is the number of env vars being passed through. (n=11: 8 'vision lines', racecar_angle, angle_to_reward, dist_to_reward)
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.3
    EPS_DECAY = 10000
    TAU = 0.003
    LR = 1e-3
    N_ACTIONS = 9
    N_OBSERVATIONS = 12
    MEMORY_FRAMES = 10000
    last_action = None

    RENDER = True
    LOSS_CALC_INDEX = 0


    def __init__(self, TRAIN_INFER_TOGGLE):
        """Instantaite class of DQN Model"""
        
        # Create reference to racegame that will be passed through each time a new instance of racegame is created 
        self.racegame_session = None
        self.racegame_episodes = 0
        self.coins_since_last_print_window = 0
        
        # Leverage GPU if exists, otherwise use CPU
        self.device, self.max_episodes = instantiate_hardware()

        # Create neural nets of input size of N_Observations and output size of N_actions. Load to GPU if GPU is available
        self.policy_net = DQN(self.N_OBSERVATIONS, self.N_ACTIONS).to(self.device)
        
        # If we want to be training our model, we need to set up different variables as well as instantiating an optimizer and a Memory object
        if TRAIN_INFER_TOGGLE == "TRAIN":
            
            # If we decide to train the model, we'll need a target net to address overfitting. We'll set up so that it is the same as the policy net
            self.target_net = DQN(self.N_OBSERVATIONS, self.N_ACTIONS).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())

            # Create vars to track training steps / list to show loss in real-time
            # One episode relates to one attempt in the 'Game Instance' (i.e., you finish # of laps or you crash)
            self.steps_done = 0
            self.loss_calculations = []

            # Create instance of optimizer and replay memory object
            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
            self.memory = ReplayMemory(self.MEMORY_FRAMES)
            
        # If not training the model, can simply load existing parameters for run
        elif TRAIN_INFER_TOGGLE == "INFER":
            self.policy_net.load_state_dict(torch.load('/assets/nn_params/Policy_Net_Params-datetime'))

        else:
            print("***ALERT*** Please ensure 'TRAIN_INFER_TOGGLE' is set to either 'TRAIN' or 'INFER'")


    def racegame_session_setter(self, racegame_session):
        """Simple setter function to pass through a racegame each time a new racegame instance is created"""
        self.racegame_session = racegame_session


    def run_model_inference(self, state):
        """Method that leverages the fully trained NN; takes in a state and returns an array for the possible action"""
        # Find index of max Q-value from the trained neural net - then return an array of all zeros except for that max value, which is a 1 (desired action)
        action_tensor = self.policy_net(state).max(1)[1]
        return convert_action_tensor(action_tensor)


    def select_action(self, state):
        """Pass through environment state and return action (i.e., driving motion)"""
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action_tensor = self.policy_net(state).max(1)[1]
                return convert_action_tensor(action_tensor, self.N_ACTIONS)
        else:
            return random_action_motion(self.N_ACTIONS, self.last_action)


    def print_progress(self):
        """Function to print training progress to the console every 'print_freq' episodes"""
        print_freq = 10
        rolling_average = 1000
        
        loss_catalog = torch.tensor(self.loss_calculations, dtype=torch.float)

        if (self.racegame_episodes % print_freq == 0) or (self.racegame_episodes == self.max_episodes-1):
            if len(loss_catalog) >= rolling_average:
                avg_loss = loss_catalog[self.LOSS_CALC_INDEX:].mean().item()
            else:
                avg_loss = loss_catalog.mean().item()

            print(f"Episode {self.racegame_episodes}/{self.max_episodes} - Avg Loss: {avg_loss:.2f} ({self.LOSS_CALC_INDEX}-{len(loss_catalog)}) - Coins Since Last Print: {self.coins_since_last_print_window}")
            self.coins_since_last_print_window = 0
            self.LOSS_CALC_INDEX = len(loss_catalog)


    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)        # Outputs a tensor of dim 12 x 128 (i.e., STATE x BATCH)
        action_batch = torch.cat(batch.action)      # Outputs a tensor of dim 9 x 128 (i.e., ACTION x BATCH)
        reward_batch = torch.cat(batch.reward)      # Outputs a tensor of dim 128

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.argmax(dim=1).unsqueeze(-1)).squeeze()


        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber Loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Append loss calculation for later printout
        self.loss_calculations.append(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def train_model(self):
        
        # Each episode refers to an 'attempt' in the game (i.e., car crashes, finishes # of laps, or is cut off because of time limit)
        for _ in range(self.max_episodes):
            
            state = self.racegame_session.racegame.racecar.return_clean_model_state()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
            for t in count():
                action = self.select_action(state)
                state, reward, crashed, laps_finished, time_limit = self.racegame_session.racegame.ai_train_step(action, self.RENDER)
                reward = torch.tensor([reward], device=self.device)
                done = crashed or laps_finished or time_limit

                if crashed:
                    next_state = None
                else:
                    next_state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                t_action = torch.tensor(action, dtype=torch.long, device=self.device).unsqueeze(0)
                t_state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                self.memory.push(t_state, t_action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.coins_since_last_print_window += self.racegame_session.racegame.coins
                    self.racegame_session.reset_racegame()
                    self.racegame_episodes = self.racegame_session.attempts
                    self.print_progress()
                    break
        
        # When done, export parameters and loss data
        self.export_params_and_loss()


    def export_params_and_loss(self):
        """Function to export neural net weights and loss"""
        formatted_datetime = datetime.datetime.now().strftime("%m.%d.%y-%H.%M")
        save_loss_to_csv(self.loss_calculations, "Loss_Calculations-" + formatted_datetime)
        self.policy_net.export_parameters("Policy_Net_Params-" + formatted_datetime)
        self.target_net.export_parameters("Target_Net_Params-" + formatted_datetime)