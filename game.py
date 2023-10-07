import pygame
from models import Racecar, Racetrack, RewardCoin
from game_render import GameBackground
from utility.utils import *
from utility.drawing_module import *


class RaceGame:
    
    """Main game class responsible for handling game logic, input, and rendering."""    
    # Define motion parameters
    _ACCELERATION = 0.6
    _TURNSPEED = 0.09
    _DRAG = 0.9

    # Other metavariables
    _LAPS = 2
    _TIMELIMIT = _LAPS * 20
    _TRAINING_RENDER_TOGGLE = False
    _CLICKELIGIBILITY_MANUALOVERRIDE = True

    def __init__(self, screen, gamesettings, model, session_metadata, session_assets):
        """Initialize the game, including Pygame, screen, sprites, and other game parameters."""

        # Set game env variables
        self.gamesettings, self.session_metadata, self.session_assets = gamesettings, session_metadata, session_assets
        self.screen = screen
        
        # Load assets and create objects
        self.game_background = GameBackground(self.session_assets["GameBackground"], self.screen)
        self.racetrack = Racetrack(self.gamesettings['random_coin_start'], self.session_assets["RacetrackLines"], self.session_assets["RewardLocations"])
        self.coinsprite = session_assets["CoinSprite"]
        self.rewardcoin = RewardCoin(self.session_assets, self.racetrack.rewards[self.racetrack.reward_coin_index])
        self.racecar = Racecar(self.session_assets, self.racetrack.start_position, self.rewardcoin)
        self.racecar_previous_action = 1
        
        # Set game states and timers
        self.running = True
        self.game_termination_reason = "N/a"
        self.coins = 0
        self.cumulative_reward = 0
        self.reward_change = 0
        self.prior_reward_coin_distance = 0
        self.frame_action = []
        self.coins_per_lap = len(self.racetrack.rewards)
        self.clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()

        # Create instance of drawing class if toggled on
        self.drawing_module = Drawing() if self.gamesettings['draw_toggle'] else None
        if self.gamesettings['draw_toggle']: self._TIMELIMIT = 1000

        # Stores instance of DQN model that was passed through if set to 'AI' and calculates first instance of the racecar's state
        self.model = model
        self._update_racecar_state(instantiate=True)


    def human_main_loop(self):
        """Main game loop that handles input, updates game logic, and draws game objects. Also caps the frame rate at 40 fps."""
        self._handle_human_input()
        self._update_game_logic()
        self._render_game()
        self.clock.tick(40)
        
        if self.running == False:
            self.session_metadata['attempts'] += 1


    def ai_infer_loop(self):
        """Main game loop if set to "AI" and 'INFER'. Calls '_handle_ai_input()' to leverage neural network for game inputs"""
        self._handle_ai_input()
        self._update_game_logic()
        self._render_game()
        self.clock.tick()

        if self.running == False:
            self.session_metadata['attempts'] += 1


    def ai_train_step(self, action, render):
        """Method called by AI to step through the game one frame at a time. Returns all necessary variables for training as well"""
        # If user presses the 'Q' key, it toggles between manual override and back to model inference
        self.session_metadata['manual_override'], self._CLICKELIGIBILITY_MANUALOVERRIDE = manual_override_check(pygame.key.get_pressed(), self._CLICKELIGIBILITY_MANUALOVERRIDE, self.session_metadata['manual_override'])
        
        if self.session_metadata['manual_override']: self._handle_ai_input(keypress_to_action(pygame.key.get_pressed(), True))
        else: self._handle_ai_input(action)
        
        self._update_game_logic()
        if render: 
            self._render_game()
            if self.session_metadata['manual_override']: self.clock.tick(40)
            else: self.clock.tick()
        else: self.game_background.render_off(self.screen, self.model.get_model_toggles())
        self.gamesettings["display_hitboxes"] = self.model.get_model_toggles()["ShowHitboxes"]
        
        if self.running == False:
            self.session_metadata['attempts'] += 1

        state = self.racecar.return_clean_model_state()
        reward = self.reward_change
        crashed = True if self.game_termination_reason == "Crash" else False
        laps_finished = True if self.game_termination_reason == "Lap(s) Completed" else False
        time_limit = True if self.game_termination_reason == "Time Limit" else False

        return state, reward, crashed, laps_finished, time_limit


    def _update_racecar_state(self, last_frame_action=[], instantiate=False):
        """Function to update racecar environment variables"""
        self.racecar.calculate_vision_lines(self.gamesettings['grid_dims'] , self.session_assets)
        self.prior_reward_coin_distance = self.racecar.modelinputs['distance_to_reward']
        self.racecar.calculate_reward_line(self.rewardcoin)
        self.racecar.calculate_velocity_vectors()
        self.racecar_previous_action = self.racecar.modelinputs['last_action_index']
        self.racecar.modelinputs['last_action_index'] = 1 if instantiate == True else last_frame_action.index(1)
        

    def _handle_human_input(self):
        """
        Handle user input events such as closing the window, drawing events, and keyboard key presses.
        Also updates the racecar's motion based on the keys pressed.
        """
        game_exit_or_drawing(pygame.event.get(), self.gamesettings['draw_toggle'], self.gamesettings['racetrack_reward_toggle'], self.drawing_module)
        self.frame_action = keypress_to_action(pygame.key.get_pressed(), False)
        # Handle movement then update racecar velocity and position. Once done, update the environment data (vision lines, reward lines)
        action_to_motion(self.racecar, self.frame_action, self._ACCELERATION, self._TURNSPEED)
        self.racecar.apply_drag(self._DRAG)
        self._update_racecar_state(self.frame_action)
        # self.racecar.return_clean_model_state()


    def _handle_ai_input(self, training_action=None):
        """
        Handle driving from AI while still allowing events such as closing the window, drawing events, and keyboard key presses.
        Also updates the racecar's motion based on the keys pressed.
        """
        game_exit_or_drawing(pygame.event.get(), self.gamesettings['draw_toggle'], self.gamesettings['racetrack_reward_toggle'], self.drawing_module)
        
        if training_action == None:
            self.frame_action = self.model.run_model_inference(self.racecar.return_clean_model_state())
        else:
            self.frame_action = training_action

        # Handle movement based on input from either human or AI, then update racecar velocity and position
        # Once done, update the environment data (vision lines, reward lines)
        action_to_motion(self.racecar, self.frame_action, self._ACCELERATION, self._TURNSPEED, True)
        self.racecar.apply_drag(self._DRAG)
        self._update_racecar_state(self.frame_action)


    def _update_game_logic(self):
        """
        Update the game logic by moving the racecar and checking for collisions with the walls.
        """
        # Move the racecar
        self.racecar.move()
        self.racecar.calc_rel_angle_to_reward()
        
        # Reset func metavariables
        _car_survived_frame = True
        self.reward_change = 0

        # Define reward function variables
        Coin_Reward = 2000
        Crash_Penalty = 2000

        VelocityToCoin_Reward = 3
        LackOfMotion_Penalty = 2
        BehaviorKeep_Reward = 1
        BehaviorChange_Penalty = 2

        # Check for collisions with the walls
        neighboring_lines = get_neighbor_lines(self.session_assets, self.gamesettings["grid_dims"], (self.racecar.position[0,0], self.racecar.position[0,1]))
        for line in neighboring_lines:
            if self.racecar.check_line_collision(line): 
                self.running = False
                _car_survived_frame = False
                self.game_termination_reason = "Crash"
                self.reward_change -= Crash_Penalty
                break
        
        # If car is still alive after the move - reward it if it moves closer to the reward coin or further away
        if self.running and _car_survived_frame:
            # self.rewardchange += FacingCoin_Reward - (abs(self.racecar.modelinputs['rel_angle_to_reward']) * 2 * FacingCoin_Reward)
            self.reward_change += min((self.racecar.modelinputs['velocity_to_reward'] * VelocityToCoin_Reward), 10)
            self.reward_change -= min(max(LackOfMotion_Penalty - self.racecar.modelinputs['velocity_to_reward'], 0), LackOfMotion_Penalty)
            self.reward_change += BehaviorKeep_Reward if ((self.racecar_previous_action == self.racecar.modelinputs['last_action_index']) and (self.racecar_previous_action != 4)) else 0
            self.reward_change -= BehaviorChange_Penalty if (self.racecar_previous_action != self.racecar.modelinputs['last_action_index']) else 0

        # Check for collisions with the coins; if so add to score / reward function
        if self.rewardcoin.intersect_with_reward(self.racecar.modelinputs['distance_to_reward']):
            self.coins += 1
            if self.coins == (self._LAPS * self.coins_per_lap)+1:
                self.running = False
                self.game_termination_reason = "Lap(s) Completed"
                self.session_metadata['wins'] += 1
            
            self.racetrack.update_reward_coin_index()
            self.reward_change += Coin_Reward
            self.rewardcoin = RewardCoin(self.session_assets, self.racetrack.rewards[self.racetrack.reward_coin_index])
        
        # Check for violation of time limit
        if ((pygame.time.get_ticks() - self.start_time) // 1000) > self._TIMELIMIT:
            self.running = False
            self.game_termination_reason = "Time Limit"

        # Update score
        self.cumulative_reward += self.reward_change


    def get_time(self):
        """Simple getter function for the DQN module to be able to calculate fps"""
        return pygame.time.get_ticks()


    def _render_game(self):
        """
        Render all game components on the screen including background, racecar, 
        racetrack, rewards, and the game's scoreboard.
        """
        # Calculate frame rate and elapsed time
        frame_rate = int(self.clock.get_fps())
        elapsed_time = (pygame.time.get_ticks() - self.start_time) // 1000

        # Draw the background and the racecar
        self.game_background.draw_scoreboard(self.screen, elapsed_time, self.cumulative_reward, frame_rate, self.session_metadata)
        self.racecar.draw(self.screen, self.gamesettings['display_hitboxes'])
        
        # Draw reward coins and racetrack lines (depending on if toggled / if drawing them), resp.
        if not self.gamesettings['draw_toggle'] or self.gamesettings['racetrack_reward_toggle'] == "REWARD": self.racetrack.draw(self.screen, self.gamesettings['display_hitboxes'])
        if not self.gamesettings['draw_toggle'] or self.gamesettings['racetrack_reward_toggle'] == "RACETRACK": self.rewardcoin.draw(self.screen, self.gamesettings['display_hitboxes'])
        
        # Draw display arrows (if toggled to show them) and draw the neural net training toggles
        if self.gamesettings['display_arrows']: self.game_background.draw_key_status(self.screen, self.frame_action, True if self.gamesettings['human_ai_toggle'] == "AI" else False)
        if ((self.gamesettings['human_ai_toggle'] == "AI") and (self.gamesettings['train_infer_toggle'] == "TRAIN")): self.game_background.draw_model_toggles(self.screen, self.model.get_model_toggles())


        # Show the recently drawn racetrack lines and rewards if the drawing module is turned on
        if self.gamesettings['draw_toggle']: 
            if self.gamesettings['racetrack_reward_toggle'] == "RACETRACK": self.drawing_module.draw_rt_lines(self.screen)
            else: self.drawing_module.draw_rewards(self.rewardcoin, self.screen)

        pygame.display.flip()