import pygame
import datetime
from utils import load_sprite, shrink_sprite, keypress_to_action, action_to_motion, game_exit_or_drawing
from models import GameBackground, Racecar, Racetrack, RewardCoin
from drawing_module import Drawing


DISPLAY_HITBOXES = False                 # Toggle to turn on displaying hitboxes or not
DISPLAY_ARROWS = False                   # Toggle to turn on the arrow key displays
SCREEN_SIZE = (800, 600)                # Define the game screen size

class RaceGame:
    """Main game class responsible for handling game logic, input, and rendering."""    
    # Define motion parameters
    ACCELERATION = 0.6
    TURN_SPEED = 4.0
    DRAG = 0.9

    # Define reward function metrics
    APPROACHING_COIN_REWARD = 10
    COINREWARD = 50
    CRASHPENALTY = 50
    LAPS = 1
    SCORE_DECAY_RATE = 1
    TIME_LIMIT = LAPS * 15

    def __init__(self, attempt, wins, draw_toggle, racetrack_reward_toggle, human_ai_toggle, train_infer_toggle, model):
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
        self.racecar = Racecar((72,356), shrink_sprite(load_sprite("RacecarSprite"), 0.15), (0,0))
        self.racetrack = Racetrack(draw_toggle, racetrack_reward_toggle)
        self.coinsprite = shrink_sprite(load_sprite("MarioCoin"), 0.015) 
        self.rewardcoin = RewardCoin(self.racetrack.rewards[0], self.coinsprite)
        
        # Calculate first instance of state to ensure model inputs are set and can be passed through to the neural net if necessary
        self._update_racecar_env_vars()

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
        self._handle_ai_input(action)
        self._update_game_logic()
        if render:
            self._render_game()
            self.clock.tick(40)
        
        if self.running == False:
            self.attempt += 1

        state = self.racecar.return_clean_model_state()
        reward = self.rewardchange
        crashed = True if self.game_termination_reason == "Crash" else False
        laps_finished = True if self.game_termination_reason == "Lap(s) Completed" else False
        time_limit = True if self.game_termination_reason == "Time Limit" else False

        return state, reward, crashed, laps_finished, time_limit


    def _update_racecar_env_vars(self):
        """Function to update racecar environment variables"""
        self.racecar.calculate_vision_lines(self.racetrack.lines)
        self.prior_reward_coin_distance = self.racecar.modelinputs['distance_to_reward']
        self.racecar.calculate_reward_line(self.rewardcoin)
        self.racecar.modelinputs['velocity'] = self.racecar.velocity
        

    def _handle_human_input(self):
        """
        Handle user input events such as closing the window, drawing events, and keyboard key presses.
        Also updates the racecar's motion based on the keys pressed.
        """
        game_exit_or_drawing(pygame.event.get(), self.draw_toggle, self.racetrack_reward_toggle, self.drawing_module)
        self.frame_action = keypress_to_action(pygame.key.get_pressed())

        # Handle movement then update racecar velocity and position. Once done, update the environment data (vision lines, reward lines)
        action_to_motion(self.racecar, self.frame_action, self.ACCELERATION, self.TURN_SPEED)
        self.racecar.apply_drag(self.DRAG)
        self._update_racecar_env_vars()


    def _handle_ai_input(self, training_action=None):
        """
        Handle driving from AI while still allowing events such as closing the window, drawing events, and keyboard key presses.
        Also updates the racecar's motion based on the keys pressed.
        """
        game_exit_or_drawing(pygame.event.get(), self.draw_toggle, self.racetrack_reward_toggle, self.drawing_module)
        
        if self.train_infer_toggle == "TRAIN":
            self.frame_action = training_action
        elif self.train_infer_toggle == "INFER":
            self.frame_action = self.model.run_model_inference(self.racecar.return_clean_model_state)
        
        # Handle movement based on input from either human or AI, then update racecar velocity and position
        # Once done, update the environment data (vision lines, reward lines)
        action_to_motion(self.racecar, self.frame_action, self.ACCELERATION, self.TURN_SPEED)
        self.racecar.apply_drag(self.DRAG)
        self._update_racecar_env_vars()


    def _update_game_logic(self):
        """
        Update the game logic by moving the racecar and checking for collisions with the walls.
        """
        # Move the racecar
        self.racecar.move()
        _car_survived_frame = True
        self.rewardchange = 0

        # Check for collisions with the walls
        for line in self.racetrack.lines:
            if self.racecar.check_line_collision(line): 
                self.running = False
                self.game_termination_reason = "Crash"
                _car_survived_frame = False
                self.rewardchange -= self.CRASHPENALTY
        
        # If car is still alive after the move, add one to the reward function
        if self.running and _car_survived_frame:
            # Making function to add to reward function if car covers > 1 pixels closer to reward in a frame)
            if self.prior_reward_coin_distance > (1 + self.racecar.modelinputs['distance_to_reward']):
                self.rewardchange += self.APPROACHING_COIN_REWARD

        # Check for collisions with the coins; if so add to score / reward function
        for line in self.rewardcoin.lines:
            if self.racecar.check_line_collision(line):
                # Update score (coin count). If you hit the # of laps, then end the game and add a win
                self.coins += 1
                if self.coins == self.LAPS * self.coins_per_lap:
                    self.running = False
                    self.game_termination_reason = "Lap(s) Completed"
                    self.wins += 1
                
                self.rewardchange += self.COINREWARD
                _next_reward_coin = self.coins % len(self.racetrack.rewards)
                self.rewardcoin = RewardCoin(self.racetrack.rewards[_next_reward_coin], self.coinsprite)
                break
        
        # Check for violation of time limit
        if ((pygame.time.get_ticks() - self.start_time) // 1000) > self.TIME_LIMIT:
            self.running = False
            self.game_termination_reason = "Time Limit"

        # Update score
        self.rewardfunction = (self.rewardfunction + self.rewardchange) * self.SCORE_DECAY_RATE


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
            self.game_background.draw_key_status(self.screen, self.frame_action)
        
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