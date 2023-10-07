import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from itertools import count
from .dqn_modules import ReplayMemory, DQN, Transition
from .dqn_utilities import *


class DQN_Model:
    """Class encompassing neural network, including training and inference functions"""

    # BATCH_SIZE is the number of transitions sampled from the replay buffer. Given sparsity of events (coins, crashes, etc.) opting for smaller nn w/ a large memory / batch_size
    # GAMMA is the discount factor. Choosing large discount factor since life is relatively long (many steps typically >500)
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon. Keeping large toward end to promote level of random behavior
    # EPS_DECAY lets you add a number of steps of semi-random motion before you start amortizing semi-randomization with EPS_DECAY
    # TAU is the update rate of the target network. To be tuned
    # LR is the learning rate of the ``AdamW`` optimizer. To be tuned
    # N_ACTIONS is the number of possible actions from the game (n=9: Nothing, Up, Left, Right, Down, Up-Left, Up-Right, Down-Left, Down-Right)
    # N_OBSERVATIONS is the number of env vars being passed through. (n=11: 8 'vision lines', racecar_angle, angle_to_reward, dist_to_reward)
    
    BATCH_SIZE = 512
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.2
    EPS_DECAY = 1000000
    TAU = 1e-2
    LR = 3e-4
    N_OUTPUT_SIZE = 5
    M_STATE_SIZE = 15
    MEMORY_FRAMES = 10000
    MAX_CUDA_EPISODES = 10000
    IMPORT_WEIGHTS_FOR_TRAINING = True

    # Setting a dictionary to pass back and forth to store information about user toggling different model actions
    model_toggles = {
        "ClickEligible": True,
        "Render": True,
        "ShowHitboxes": None,
        "SaveWeights": False,
        "ImportWeights": False
    }

    def __init__(self, gamesettings):
        """Instantaite class of DQN Model"""
        
        # Create reference to racegame that will be passed through each time a new instance of racegame is created 
        self.gamesettings = gamesettings
        self.racegame_session = None
        self.last_action = None

        # Leverage GPU if exists, otherwise use CPU
        self.device, self.max_episodes = instantiate_hardware(self.MAX_CUDA_EPISODES)

        # Create neural nets of input size of N_Observations and output size of N_actions. Load to GPU if GPU is available
        self.policy_net = DQN(self.M_STATE_SIZE, self.N_OUTPUT_SIZE).to(self.device)
        
        # If not training the model, can simply load existing parameters for run
        if self.gamesettings['train_infer_toggle'] == 'INFER': self.policy_net.load_state_dict(torch.load('./assets/nn_params/Policy_Net_Params-10.07.23-14.13'))
        
        # If we want to be training our model, we need to create memory object, target_net, and other necessary var
        if self.gamesettings['train_infer_toggle'] == 'TRAIN': 
            self.target_net = DQN(self.M_STATE_SIZE, self.N_OUTPUT_SIZE).to(self.device)
            
            if self.IMPORT_WEIGHTS_FOR_TRAINING:
                self.policy_net.load_state_dict(torch.load('./assets/nn_params/Policy_Net_Params-10.07.23-14.35'))
                self.target_net.load_state_dict(torch.load('./assets/nn_params/Target_Net_Params-10.07.23-14.35'))
                self.EPS_DECAY = self.EPS_DECAY * 0.5            # Reduce amount of random steps at beginning
            else:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Create vars to track and print progress during training
            self.progress_vars = {
                "Episodes": 0,
                "CoinsSinceLastPrint": 0,
                "TimeAtLastPrint": 0,
                "IndexOfLastPrint": 0, 
                "PrintFrequency": 10,
                "EpisodeTotalReward": []
            }
            
            # Vars to track steps done (for EPS decay) and save loss_calculations for each step
            self.steps_done = 0
            self.loss_memory = []

            # Create instance of optimizer and replay memory object
            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
            self.memory = ReplayMemory(self.MEMORY_FRAMES)


    def racegame_session_setter(self, racegame_session):
        """Simple setter function to pass through a racegame each time a new racegame instance is created"""
        self.racegame_session = racegame_session
        self.model_toggle_buttons = racegame_session.racegame.game_background.get_model_toggle_buttons()
        self.model_toggles["ShowHitboxes"] = racegame_session.gamesettings["display_hitboxes"]

    
    def get_model_toggles(self):
        """Simple getter to return the model toggles for use in screen printing"""
        return self.model_toggles
    

    def run_model_inference(self, state):
        """Method that leverages the fully trained NN; takes in a state and returns an array for the possible action"""
        # Find index of max Q-value from the trained neural net - then return an array of all zeros except for that max value, which is a 1 (desired action)\
        if np.random.random() < self.EPS_END: return random_action_motion(self.N_OUTPUT_SIZE, self.last_action)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                action_tensor = self.policy_net(state).max(1)[1]
            return convert_action_tensor_to_list(action_tensor, self.N_OUTPUT_SIZE) 
            

    def select_action_in_training(self, state):
        """Pass through environment state and return action (i.e., driving motion)"""
        # Compute new EPS threshold and update steps done
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            np.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        
        if np.random.random() < eps_threshold: return random_action_motion(self.N_OUTPUT_SIZE, self.last_action)
        else:
            with torch.no_grad():
                action_tensor = self.policy_net(state).max(1)[1]
                return convert_action_tensor_to_list(action_tensor, self.N_OUTPUT_SIZE)


    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE: return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)        # Outputs a tensor of dim 15 x 128 (i.e., STATE x BATCH)
        action_batch = torch.cat(batch.action)      # Outputs a tensor of dim 5 x 128 (i.e., ACTION x BATCH)
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

        # Compute MSE Loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Append loss calculation for later printout
        self.loss_memory.append(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 75)
        self.optimizer.step()


    def train_model(self):
        
        # Each episode refers to an 'attempt' in the game (i.e., car crashes, finishes # of laps, or is cut off because of time limit)
        for _ in range(self.max_episodes):
            
            t_state = self.racegame_session.racegame.racecar.return_clean_model_state()
            t_state = torch.tensor(t_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
            for t in count():
                self.model_toggles = check_toggles(self.model_toggles, self.model_toggle_buttons)
                t_action = self.select_action_in_training(t_state)
                self.last_action = t_action
                t_1_state, t_reward, crashed, laps_finished, time_limit = self.racegame_session.racegame.ai_train_step(t_action, self.model_toggles["Render"])
                done = crashed or laps_finished or time_limit
                
                # Store the transition in memory
                t_action = torch.tensor(t_action, dtype=torch.float32, device=self.device).unsqueeze(0)
                t_1_state = torch.tensor(t_1_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                t_reward = torch.tensor([t_reward], dtype=torch.float32, device=self.device)
                self.memory.push(t_state, t_action, t_1_state, t_reward)

                # Need to then push a 'None' next_state so that we can set up 'final next states'
                if crashed:
                    t_1_state = None
                    self.memory.push(t_state, t_action, t_1_state, t_reward)
                    
                # Move to the next state
                t_state = t_1_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if self.model_toggles["SaveWeights"]:
                    export_params_and_loss(self.loss_memory, self.policy_net, self.target_net)
                    self.memory.save_replay_memory(self.M_STATE_SIZE)
                    self.model_toggles["SaveWeights"] = False
                
                if self.model_toggles["ImportWeights"]:
                    self.memory.load_replay_memory(self.device)
                    self.model_toggles["ImportWeights"] = False
                
                if done:
                    self.progress_vars["CoinsSinceLastPrint"] += self.racegame_session.racegame.coins
                    self.progress_vars["EpisodeTotalReward"].append(self.racegame_session.racegame.cumulative_reward)
                    self.racegame_session.reset_racegame()
                    self.progress_vars["Episodes"] = self.racegame_session.session_metadata['attempts']
                    print_progress(self.progress_vars, self.loss_memory, self.max_episodes, self.racegame_session.racegame.get_time())
                    self.last_action = None
                    break
        
        # When done, export parameters and loss data
        export_params_and_loss(self.loss_memory, self.policy_net, self.target_net)