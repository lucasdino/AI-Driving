import pygame
import datetime
from utils import load_sprite, shrink_sprite
from models import GameBackground, Racecar, Racetrack, RewardCoin
from drawing_module import Drawing


class Driver:
    """Main game class responsible for handling game logic, input, and rendering."""

    SCREEN_SIZE = (800, 600)                # Define the game screen size
    DRAW_TOGGLE = False                     # Toggle on the drawing module
    RACETRACK_REWARD_TOGGLE = "REWARD"      # If DRAW_TOGGLE, set which you will be drawing with your mouse
    HUMAN_AI_TOGGLE = "HUMAN"               # Set who will be driving the car
    
    # Define motion parameters
    ACCELERATION = 0.3
    TURN_SPEED = 5.0
    DRAG = 0.9

    def __init__(self, attempt):
        """Initialize the game, including Pygame, screen, sprites, and other game parameters."""

        self._initialize_pygame()
        self.screen = pygame.display.set_mode(self.SCREEN_SIZE)
        
        # Load assets and create objects
        self.game_background = GameBackground(load_sprite("RacetrackSprite", False))
        self.racecar = Racecar((72,356), shrink_sprite(load_sprite("RacecarSprite"), 0.15), (0,0))
        self.racetrack = Racetrack()

        self.coinsprite = shrink_sprite(load_sprite("MarioCoin"), 0.015)
        self.rewardcoin = RewardCoin(self.racetrack.rewards[0], self.coinsprite)
        
        # Set game states and timers
        self.running = True
        self.score = 0
        self.attempt = attempt
        self.clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()
        
        # Create instance of drawing class if toggled on
        if self.DRAW_TOGGLE:
            self.drawing_module = Drawing()


    def main_loop(self):
        """
        Main game loop that handles input, updates game logic, and draws game objects.
        Also caps the frame rate at 40 fps.
        """
        self._handle_input()
        self._update_game_logic()
        self._render_game()
        self.clock.tick(40)


    def _initialize_pygame(self):
        """Initialize Pygame and set the window title."""
        pygame.init()
        pygame.display.set_caption("AI Driver")


    def _handle_input(self):
        """
        Handle user input events such as closing the window, drawing events, and keyboard key presses.
        Also updates the racecar's motion based on the keys pressed.
        """
        for event in pygame.event.get():
            # Handle 'quit' events
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                if self.DRAW_TOGGLE:
                    formatted_datetime = datetime.datetime.now().strftime("%m.%d.%y-%H.%M")
                    self.drawing_module.save_drawing_to_csv("drawn_" + self.RACETRACK_REWARD_TOGGLE.lower() + "-" + formatted_datetime)
                quit()

            # Handle drawing events if drawing is toggled on
            if self.DRAW_TOGGLE:
                if self.RACETRACK_REWARD_TOGGLE == "RACETRACK":
                    self.drawing_module.handle_rt_drawing_events(event)
                else:
                    self.drawing_module.handle_reward_drawing_events(event)

        # Handle key presses for racecar motion
        keypress = 0
        if self.HUMAN_AI_TOGGLE == "HUMAN":
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: keypress = 1
            elif keys[pygame.K_DOWN]: keypress = 2
            elif keys[pygame.K_LEFT]: keypress = 3
            elif keys[pygame.K_RIGHT]: keypress = 4
        
        elif self.HUMAN_AI_TOGGLE == "AI":
            # self.racecar.modelinputs
            pass     
        else:
            print("Please make sure 'HUMAN_AI_TOGGLE is set to either 'HUMNAN' or 'AI'")
        
        # Handle movement based on input from either human or AI
        if keypress == 1:
            self.racecar.accelerate(self.ACCELERATION)
        elif keypress == 2:
            self.racecar.brake(self.ACCELERATION)
        if keypress == 3:
            self.racecar.turn_left(self.TURN_SPEED)
        elif keypress == 4:
            self.racecar.turn_right(self.TURN_SPEED)
            
        # Update racecar's velocity and position
        self.racecar.apply_drag(self.DRAG)
        self.racecar.move()
        
        # Update racecar's vision lines
        self.racecar.vision_line_distance(self.racetrack.lines)
        self.racecar.calculate_reward_line(self.rewardcoin)


    def _update_game_logic(self):
        """
        Update the game logic by moving the racecar and checking for collisions with the walls.
        """
        # Move the racecar
        self.racecar.move()

        # Check for collisions with the walls
        for line in self.racetrack.lines:
            if self.racecar.check_line_collision(line): 
                self.running = False
        
        # Check for collisions with the coins
        for line in self.rewardcoin.lines:
            if self.racecar.check_line_collision(line):
                self.score += 1
                _next_reward_coin = self.score % len(self.racetrack.rewards)
                self.rewardcoin = RewardCoin(self.racetrack.rewards[_next_reward_coin], self.coinsprite)
                break


    def _render_game(self):
        """
        Render all game components on the screen including background, racecar, 
        racetrack, rewards, and the game's scoreboard.
        """
        # Calculate frame rate and elapsed time
        frame_rate = int(self.clock.get_fps())
        elapsed_time = (pygame.time.get_ticks() - self.start_time) // 1000

        # Draw the background and the racecar
        self.game_background.draw(self.screen, elapsed_time, self.score, frame_rate, self.attempt)
        self.racecar.draw(self.screen)
        self.racetrack.draw(self.screen)
        self.rewardcoin.draw(self.screen)

        # Draw racetrack lines and rewards if the drawing module is turned on
        if self.DRAW_TOGGLE: 
            if self.RACETRACK_REWARD_TOGGLE == "RACETRACK":
                self.drawing_module.draw_rt_lines(self.screen)
            else:
                self.drawing_module.draw_rewards(self.rewardcoin, self.screen)

        pygame.display.flip()
