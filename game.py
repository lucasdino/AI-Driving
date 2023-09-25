import pygame
from models import *
from utility.utils import *
from utility.drawing_module import *


class RaceGame:
    
    """Main game class responsible for handling game logic, input, and rendering."""    
    # Define motion parameters
    _ACCELERATION = 0.6
    _TURNSPEED = 0.09
    _DRAG = 0.9

    # Other metavariables
    _SCREENSIZE = (800, 600)
    _LAPS = 2
    _TIMELIMIT = _LAPS * 20
    _TRAINING_RENDER_TOGGLE = False
    _CLICKELIGIBILITY_MANUALOVERRIDE = True

    def __init__(self, gamesettings, model, session_metadata):
        """Initialize the game, including Pygame, screen, sprites, and other game parameters."""

        self._initialize_pygame()

        # Set game env variables
        self.gamesettings, self.session_metadata = gamesettings, session_metadata
        
        # Load assets and create objects
        self.game_background = GameBackground(load_sprite("RacetrackSprite", False))
        self.racetrack = Racetrack(self.gamesettings['random_coin_start'])
        self.coinsprite = shrink_sprite(load_sprite("MarioCoin"), 0.02) 
        self.rewardcoin = RewardCoin(self.racetrack.rewards[self.racetrack.reward_coin_index], self.coinsprite)
        self.racecar = Racecar(self.racetrack.start_position, shrink_sprite(load_sprite("RacecarSprite"), 0.15), self.rewardcoin)
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

        # Stores instance of DQN model that was passed through if set to 'AI' and calculates first instance of the racecar's state
        self.model = model
        self._update_racecar_state_vars(instantiate=True)


    def _initialize_pygame(self):
        """Initialize Pygame and set the window title."""
        pygame.init()
        pygame.display.set_caption("AI Driver")
        self.screen = pygame.display.set_mode(self._SCREENSIZE)


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
        self.clock.tick(40)

        if self.running == False:
            self.session_metadata['attempts'] += 1


    def ai_train_step(self, action, render):
        """Method called by AI to step through the game one frame at a time. Returns all necessary variables for training as well"""
        # If user presses the 'Q' key, it toggles between manual override and back to model inference
        self.session_metadata['manual_override'], self._CLICKELIGIBILITY_MANUALOVERRIDE = manual_override_check(pygame.key.get_pressed(), self._CLICKELIGIBILITY_MANUALOVERRIDE, self.session_metadata['manual_override'])
        
        if self.session_metadata['manual_override']:
            self._handle_ai_input(keypress_to_action(pygame.key.get_pressed(), True))
        else:    
            self._handle_ai_input(action)
        
        self._update_game_logic()
        if render:
            self._TRAINING_RENDER_TOGGLE = True
            self._render_game()
            # self.clock.tick(40)
        
        if self.running == False:
            self.session_metadata['attempts'] += 1

        state = self.racecar.return_clean_model_state(self.rewardcoin.get_radius())
        reward = self.reward_change
        crashed = True if self.game_termination_reason == "Crash" else False
        laps_finished = True if self.game_termination_reason == "Lap(s) Completed" else False
        time_limit = True if self.game_termination_reason == "Time Limit" else False

        return state, reward, crashed, laps_finished, time_limit


    def _update_racecar_state_vars(self, last_frame_action=[], instantiate=False):
        """Function to update racecar environment variables"""
        self.racecar.calculate_vision_lines(self.racetrack.lines)
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
        self._update_racecar_state_vars(self.frame_action)
        # self.racecar.return_clean_model_state(self.rewardcoin.get_radius())


    def _handle_ai_input(self, training_action=None):
        """
        Handle driving from AI while still allowing events such as closing the window, drawing events, and keyboard key presses.
        Also updates the racecar's motion based on the keys pressed.
        """
        game_exit_or_drawing(pygame.event.get(), self.gamesettings['draw_toggle'], self.gamesettings['racetrack_reward_toggle'], self.drawing_module)
        
        if training_action == None:
            self.frame_action = self.model.run_model_inference(self.racecar.return_clean_model_state(self.rewardcoin.get_radius()))
        else:
            self.frame_action = training_action

        # Handle movement based on input from either human or AI, then update racecar velocity and position
        # Once done, update the environment data (vision lines, reward lines)
        action_to_motion(self.racecar, self.frame_action, self._ACCELERATION, self._TURNSPEED, True)
        self.racecar.apply_drag(self._DRAG)
        self._update_racecar_state_vars(self.frame_action)


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
        FacingCoin_Reward = 1
        VelocityToCoin_Reward = 1
        Coin_Reward = 2000

        DistanceToCoin_Penalty = 10
        PixelsToWall_PenaltyThreshold = 10
        ProximityToWall_Multiplier = 2
        LackOfMotion_Multiplier_Penalty = 4
        BehaviorChange_Penalty = 3
        Crash_Penalty = 4000

        # Check for collisions with the walls
        for line in self.racetrack.lines:
            if self.racecar.check_line_collision(line): 
                self.running = False
                _car_survived_frame = False
                self.game_termination_reason = "Crash"
                self.reward_change -= Crash_Penalty
                break
        
        # If car is still alive after the move - reward it if it moves closer to the reward coin or further away
        if self.running and _car_survived_frame:
            # self.rewardchange += FacingCoin_Reward - (abs(self.racecar.modelinputs['rel_angle_to_reward']) * 2 * FacingCoin_Reward)
            self.reward_change += min((self.racecar.modelinputs['velocity_to_reward'] * VelocityToCoin_Reward), 3)
            self.reward_change -= max(0, PixelsToWall_PenaltyThreshold-min(self.racecar.modelinputs['vision_distances']))*ProximityToWall_Multiplier
            # self.rewardchange -= BehaviorChange_Penalty if not(self.racecar_previous_action == self.racecar.modelinputs['last_action_index']) else 0
            self.reward_change -= LackOfMotion_Multiplier_Penalty*max(1-(abs(self.racecar.velocity[0,0])+abs(self.racecar.velocity[0,1])), 0)
            # self.rewardchange += BehaviorChange_Penalty if self.racecar.modelinputs['last_action_index'] == 1 else 0
            # self.rewardchange -= BehaviorChange_Penalty/2 if (self.racecar.modelinputs['last_action_index'] == 0 or self.racecar.modelinputs['last_action_index'] == 2) else 0
            # self.rewardchange -= BehaviorChange_Penalty if self.racecar.modelinputs['last_action_index'] == 4 else 0
            self.reward_change -= BehaviorChange_Penalty if not(self.racecar_previous_action == self.racecar.modelinputs['last_action_index']) else 0

        # Check for collisions with the coins; if so add to score / reward function
        if self.rewardcoin.intersect_with_reward(self.racecar.modelinputs['distance_to_reward']):
            self.coins += 1
            if self.coins == (self._LAPS * self.coins_per_lap)+1:
                self.running = False
                self.game_termination_reason = "Lap(s) Completed"
                self.session_metadata['wins'] += 1
            
            self.racetrack.update_reward_coin_index()
            self.reward_change += Coin_Reward
            self.rewardcoin = RewardCoin(self.racetrack.rewards[self.racetrack.reward_coin_index], self.coinsprite)
        
        # Check for violation of time limit
        if ((pygame.time.get_ticks() - self.start_time) // 1000) > self._TIMELIMIT:
            self.running = False
            self.game_termination_reason = "Time Limit"

        # Update score
        self.cumulative_reward += self.reward_change


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
        if self.gamesettings['display_arrows']:
            self.game_background.draw_key_status(self.screen, self.frame_action, True if self.gamesettings['human_ai_toggle'] == "AI" else False)
        
        self.racecar.draw(self.screen, self.gamesettings['display_hitboxes'])

        if not self.gamesettings['draw_toggle'] or self.gamesettings['racetrack_reward_toggle'] == "REWARD":         
            self.racetrack.draw(self.screen, self.gamesettings['display_hitboxes'])
        
        if not self.gamesettings['draw_toggle'] or self.gamesettings['racetrack_reward_toggle'] == "RACETRACK":         
            self.rewardcoin.draw(self.screen, self.gamesettings['display_hitboxes'])

        # Draw racetrack lines and rewards if the drawing module is turned on
        if self.gamesettings['draw_toggle']: 
            if self.gamesettings['racetrack_reward_toggle'] == "RACETRACK":
                self.drawing_module.draw_rt_lines(self.screen)
            else:
                self.drawing_module.draw_rewards(self.rewardcoin, self.screen)

        pygame.display.flip()
