import pygame
import datetime
from utils import load_sprite, shrink_sprite, keypress_to_action, action_to_motion, game_exit_or_drawing, manual_override_check, return_magnitude
from models import GameBackground, Racecar, Racetrack, RewardCoin
from drawing_module import Drawing


DISPLAY_HITBOXES = True                 # Toggle to turn on displaying hitboxes or not
DISPLAY_ARROWS = True                   # Toggle to turn on the arrow key displays
RANDOM_COIN_START = True                # Toggle to have car start at random coin location to begin or to start at starting line
SCREEN_SIZE = (800, 600)                # Define the game screen size

class RaceGame:
    """Main game class responsible for handling game logic, input, and rendering."""    
    # Define motion parameters
    ACCELERATION = 0.6
    TURN_SPEED = 4.0
    DRAG = 0.9

    # Other metavariables
    LAPS = 2
    TIME_LIMIT = LAPS * 20
    TRAINING_RENDER_TOGGLE = False
    CLICK_ELIGIBLE_MANUAL_OVERRIDE = True

    def __init__(self, attempt, wins, draw_toggle, racetrack_reward_toggle, human_ai_toggle, train_infer_toggle, model, manual_override):
        """Initialize the game, including Pygame, screen, sprites, and other game parameters."""

        self._initialize_pygame()
        self.screen = pygame.display.set_mode(SCREEN_SIZE)

        # Set game env variables
        self.human_ai_toggle = human_ai_toggle
        self.train_infer_toggle = train_infer_toggle
        self.draw_toggle = draw_toggle
        self.racetrack_reward_toggle = racetrack_reward_toggle
        
        # Load assets and create objects
        self.game_background = GameBackground(load_sprite("RacetrackSprite", False))
        self.racetrack = Racetrack(RANDOM_COIN_START)
        self.coinsprite = shrink_sprite(load_sprite("MarioCoin"), 0.02) 
        self.rewardcoin = RewardCoin(self.racetrack.rewards[self.racetrack.reward_coin_index], self.coinsprite)
        self.rewardcoinradius = self.rewardcoin.get_radius()
        self.racecar = Racecar(self.racetrack.start_position, shrink_sprite(load_sprite("RacecarSprite"), 0.15), self.rewardcoin)
        self.racecar_previous_action = 1
        
        # Calculate first instance of state to ensure model inputs are set and can be passed through to the neural net if necessary
        self._update_racecar_state_vars(instantiate=True)
        self.manual_override = manual_override

        # Set game states and timers
        self.running = True
        self.game_termination_reason = "N/a"
        self.coins = 0
        self.rewardfunction = 0
        self.rewardchange = 0
        self.prior_reward_coin_distance = 0
        self.frame_action = []
        self.coins_per_lap = len(self.racetrack.rewards)
        self.attempt = attempt
        self.wins = wins
        self.clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()

        # Create instance of drawing class if toggled on
        if self.draw_toggle:
            self.drawing_module = Drawing()
        else: self.drawing_module = None

        # Stores instance of DQN model that was passed through if set to 'AI'. If set to 'HUMAN', model was set as 'none'
        self.model = model

    def _initialize_pygame(self):
        """Initialize Pygame and set the window title."""
        pygame.init()
        pygame.display.set_caption("AI Driver")


    def human_main_loop(self):
        """Main game loop that handles input, updates game logic, and draws game objects. Also caps the frame rate at 40 fps."""
        self._handle_human_input()
        self._update_game_logic()
        self._render_game()
        self.clock.tick(40)
        
        if self.running == False:
            self.attempt += 1


    def ai_infer_loop(self):
        """Main game loop if set to "AI" and 'INFER'. Calls '_handle_ai_input()' to leverage neural network for game inputs"""
        self._handle_ai_input()
        self._update_game_logic()
        self._render_game()
        self.clock.tick(40)

        if self.running == False:
            self.attempt += 1


    def ai_train_step(self, action, render):
        """Method called by AI to step through the game one frame at a time. Returns all necessary variables for training as well"""
        # If user presses the 'Q' key, it toggles between manual override and back to model inference
        self.manual_override, self.CLICK_ELIGIBLE_MANUAL_OVERRIDE = manual_override_check(pygame.key.get_pressed(), self.CLICK_ELIGIBLE_MANUAL_OVERRIDE, self.manual_override)
        
        if self.manual_override:
            self._handle_ai_input(keypress_to_action(pygame.key.get_pressed(), True))
        else:    
            self._handle_ai_input(action)
        
        self._update_game_logic()
        if render:
            self.TRAINING_RENDER_TOGGLE = True
            self._render_game()
            self.clock.tick(40)
        
        if self.running == False:
            self.attempt += 1

        state = self.racecar.return_clean_model_state(self.rewardcoinradius)
        reward = self.rewardchange
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
        game_exit_or_drawing(pygame.event.get(), self.draw_toggle, self.racetrack_reward_toggle, self.drawing_module)
        self.frame_action = keypress_to_action(pygame.key.get_pressed(), False)
        # Handle movement then update racecar velocity and position. Once done, update the environment data (vision lines, reward lines)
        action_to_motion(self.racecar, self.frame_action, self.ACCELERATION, self.TURN_SPEED)
        self.racecar.apply_drag(self.DRAG)
        self._update_racecar_state_vars(self.frame_action)
        # self.racecar.return_clean_model_state(self.rewardcoinradius)


    def _handle_ai_input(self, training_action=None):
        """
        Handle driving from AI while still allowing events such as closing the window, drawing events, and keyboard key presses.
        Also updates the racecar's motion based on the keys pressed.
        """
        game_exit_or_drawing(pygame.event.get(), self.draw_toggle, self.racetrack_reward_toggle, self.drawing_module)
        
        if training_action == None:
            self.frame_action = self.model.run_model_inference(self.racecar.return_clean_model_state(self.rewardcoinradius))
        else:
            self.frame_action = training_action

        # Handle movement based on input from either human or AI, then update racecar velocity and position
        # Once done, update the environment data (vision lines, reward lines)
        action_to_motion(self.racecar, self.frame_action, self.ACCELERATION, self.TURN_SPEED, True)
        self.racecar.apply_drag(self.DRAG)
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
        self.rewardchange = 0

        # Define reward function variables
        FacingCoin_Reward = 1
        VelocityToCoin_Reward = 1
        Coin_Reward = 1500

        DistanceToCoin_Penalty = 10
        ProximityToWall_Penalty = 20
        LackOfMotion_Multiplier_Penalty = 3
        BehaviorChange_Penalty = 3
        Crash_Penalty = 4000

        # Check for collisions with the walls
        for line in self.racetrack.lines:
            if self.racecar.check_line_collision(line): 
                self.running = False
                _car_survived_frame = False
                self.game_termination_reason = "Crash"
                self.rewardchange -= Crash_Penalty
                break
        
        # If car is still alive after the move - reward it if it moves closer to the reward coin or further away
        if self.running and _car_survived_frame:
            # self.rewardchange += FacingCoin_Reward - (abs(self.racecar.modelinputs['rel_angle_to_reward']) * 2 * FacingCoin_Reward)
            self.rewardchange += (self.racecar.modelinputs['velocity_to_reward'] * VelocityToCoin_Reward)
            # self.rewardchange -= BehaviorChange_Penalty if not(self.racecar_previous_action == self.racecar.modelinputs['last_action_index']) else 0
            self.rewardchange -= LackOfMotion_Multiplier_Penalty*max(1-(abs(self.racecar.velocity[0])+abs(self.racecar.velocity[1])), 0)
            # self.rewardchange += BehaviorChange_Penalty if self.racecar.modelinputs['last_action_index'] == 1 else 0
            # self.rewardchange -= BehaviorChange_Penalty/2 if (self.racecar.modelinputs['last_action_index'] == 0 or self.racecar.modelinputs['last_action_index'] == 2) else 0
            # self.rewardchange -= BehaviorChange_Penalty if self.racecar.modelinputs['last_action_index'] == 4 else 0
            self.rewardchange -= BehaviorChange_Penalty if not(self.racecar_previous_action == self.racecar.modelinputs['last_action_index']) else 0

        # Check for collisions with the coins; if so add to score / reward function
        if self.rewardcoin.intersect_with_reward(self.racecar.modelinputs['distance_to_reward']):
            self.coins += 1
            if self.coins == (self.LAPS * self.coins_per_lap)+1:
                self.running = False
                self.game_termination_reason = "Lap(s) Completed"
                self.wins += 1
            
            self.racetrack.update_reward_coin_index()
            self.rewardchange += Coin_Reward
            self.rewardcoin = RewardCoin(self.racetrack.rewards[self.racetrack.reward_coin_index], self.coinsprite)
        
        # Check for violation of time limit
        if ((pygame.time.get_ticks() - self.start_time) // 1000) > self.TIME_LIMIT:
            self.running = False
            self.game_termination_reason = "Time Limit"

        # Update score
        self.rewardfunction += self.rewardchange


    def _render_game(self):
        """
        Render all game components on the screen including background, racecar, 
        racetrack, rewards, and the game's scoreboard.
        """
        # Calculate frame rate and elapsed time
        frame_rate = int(self.clock.get_fps())
        elapsed_time = (pygame.time.get_ticks() - self.start_time) // 1000

        # Draw the background and the racecar
        self.game_background.draw_scoreboard(self.screen, elapsed_time, self.rewardfunction, frame_rate, self.attempt, self.wins)
        if DISPLAY_ARROWS:
            self.game_background.draw_key_status(self.screen, self.frame_action, True if self.human_ai_toggle == "AI" else False)
        
        self.racecar.draw(self.screen, DISPLAY_HITBOXES)

        if not self.draw_toggle or self.racetrack_reward_toggle == "REWARD":         
            self.racetrack.draw(self.screen, DISPLAY_HITBOXES)
        
        if not self.draw_toggle or self.racetrack_reward_toggle == "RACETRACK":         
            self.rewardcoin.draw(self.screen, DISPLAY_HITBOXES)

        # Draw racetrack lines and rewards if the drawing module is turned on
        if self.draw_toggle: 
            if self.racetrack_reward_toggle == "RACETRACK":
                self.drawing_module.draw_rt_lines(self.screen)
            else:
                self.drawing_module.draw_rewards(self.rewardcoin, self.screen)

        pygame.display.flip()
