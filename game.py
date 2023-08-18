import pygame
import datetime
from utils import load_sprite, shrink_sprite, keypress_to_action, action_to_motion, game_exit_or_drawing
from models import GameBackground, Racecar, Racetrack, RewardCoin
from drawing_module import Drawing


DISPLAY_HITBOXES = True                # Toggle to turn on displaying hitboxes or not
DISPLAY_ARROWS = True                 # Toggle to turn on the arrow key displays
SCREEN_SIZE = (800, 600)                # Define the game screen size

class RaceGame:
    """Main game class responsible for handling game logic, input, and rendering."""    
    # Define motion parameters
    ACCELERATION = 0.6
    TURN_SPEED = 4.0
    DRAG = 0.9

    # Define reward function metrics
    DECAYRATE = 0.99
    COINREWARD = 500
    CRASHPENALTY = 1000
    LAPS = 1

    def __init__(self, attempt, wins, draw_toggle, racetrack_reward_toggle, human_ai_toggle, train_infer_toggle):
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
        
        # Set game states and timers
        self.running = True
        self.score = 0
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


    def _initialize_pygame(self):
        """Initialize Pygame and set the window title."""
        pygame.init()
        pygame.display.set_caption("AI Driver")


    def human_main_loop(self):
        """Main game loop that handles input, updates game logic, and draws game objects. Also caps the frame rate at 40 fps."""
        self._handle_input()
        self._update_game_logic()
        self._render_game()
        self.clock.tick(40)
        
        if self.running == False:
            self.attempt += 1


    def ai_main_loop(self):
        """Main game loop if set to "AI" - depending on what 'TRAIN_INFER_TOGGLE' is set to will determine how the neural net will be interacted with"""
        if self.train_infer_toggle == "TRAIN":
            pass
        elif self.train_infer_toggle == "INFER":
            pass
        else: 
            print("Please make sure the 'TRAIN_INFER_TOGGLE' in '__main__.py' is set to either 'TRAIN' or 'INFER'.")
        

    def _update_racecar_env_vars(self):
        """Function to update racecar environment variables"""
        self.racecar.vision_line_distance(self.racetrack.lines)
        self.prior_reward_coin_distance = self.racecar.modelinputs['distance_to_reward']
        self.racecar.calculate_reward_line(self.rewardcoin)


    def _handle_input(self):
        """
        Handle user input events such as closing the window, drawing events, and keyboard key presses.
        Also updates the racecar's motion based on the keys pressed.
        """
        game_exit_or_drawing(pygame.event.get(), self.draw_toggle, self.racetrack_reward_toggle, self.drawing_module)
               
        if self.human_ai_toggle == "HUMAN":
            self.frame_action = keypress_to_action(pygame.key.get_pressed())

        elif self.human_ai_toggle == "AI":
            # self.racecar.modelinputs / self.frame_action / self.rewardfunction
            pass     
        
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
                _car_survived_frame = False
                self.rewardchange -= self.CRASHPENALTY
        
        # If car is still alive after the move, add one to the reward function
        if self.running and _car_survived_frame:
            self.rewardchange += 1
            # Making function to add to reward function if car covers > 1 pixel closer to reward in a frame)
            if self.prior_reward_coin_distance > (1 + self.racecar.modelinputs['distance_to_reward']):
                self.rewardchange +=1

        # Check for collisions with the coins; if so add to score / reward function
        for line in self.rewardcoin.lines:
            if self.racecar.check_line_collision(line):
                # Update score (coin count). If you hit the # of laps, then end the game and add a win
                self.score += 1
                if self.score == self.LAPS * self.coins_per_lap:
                    self.running = False
                    self.wins += 1
                
                self.rewardchange += self.COINREWARD
                _next_reward_coin = self.score % len(self.racetrack.rewards)
                self.rewardcoin = RewardCoin(self.racetrack.rewards[_next_reward_coin], self.coinsprite)
                break
        
        # Apply decay factor to reward function
        self.rewardfunction = (self.rewardfunction + self.rewardchange) *self.DECAYRATE


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
        self.racetrack.draw(self.screen, DISPLAY_HITBOXES)
        self.rewardcoin.draw(self.screen, DISPLAY_HITBOXES)

        # Draw racetrack lines and rewards if the drawing module is turned on
        if self.draw_toggle: 
            if self.racetrack_reward_toggle == "RACETRACK":
                self.drawing_module.draw_rt_lines(self.screen)
            else:
                self.drawing_module.draw_rewards(self.rewardcoin, self.screen)

        pygame.display.flip()