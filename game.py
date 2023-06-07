import pygame
import datetime
from utils import load_sprite, shrink_sprite
from models import Racecar, Racetrack
from drawing_module import Drawing


class Driver:
    """Main game class responsible for handling game logic, input, and rendering."""

    SCREEN_SIZE = (800, 600)  # Define the game screen size
    
    def __init__(self, attempt):
        """
        Initialize the game, including Pygame, screen, sprites, and other game parameters.

        :param attempt: The current attempt number.
        """
        self._initialize_pygame()
        self.screen = pygame.display.set_mode(self.SCREEN_SIZE)
        
        # Load assets
        self.background = load_sprite("RacetrackSprite", False)
        self.racecar = Racecar((72,356), shrink_sprite(load_sprite("RacecarSprite"), 0.15), (0,0))
        self.racetrack = Racetrack()
        self.rewardcoin = shrink_sprite(load_sprite("MarioCoin"), 0.015)

        self.clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()
        self.font = pygame.font.Font(None, 20)  # Font for displaying text
        self.score = 0
        self.attempt = attempt

        self.running = True  # Game state
        
        # Drawing toggles
        self.draw_toggle = False
        self.rt_reward_toggle = "Racetrack"
        if self.draw_toggle:
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
                if self.draw_toggle:
                    formatted_datetime = datetime.datetime.now().strftime("%m.%d.%y-%H.%M")
                    self.drawing_module.save_drawing_to_csv(self.rt_reward_toggle + "_drawing-" + formatted_datetime)
                quit()

            # Handle drawing events
            if self.draw_toggle:
                if self.rt_reward_toggle == "Racetrack":
                    self.drawing_module.handle_rt_drawing_events(event)
                else:
                    self.drawing_module.handle_reward_drawing_events(event)

        # Define motion parameters
        acceleration = 0.3
        turn_speed = 5.0
        drag = 0.9

        # Handle key presses for racecar motion
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.racecar.accelerate(acceleration)
        elif keys[pygame.K_DOWN]:
            self.racecar.brake(acceleration)
        if keys[pygame.K_LEFT]:
            self.racecar.turn_left(turn_speed)
        elif keys[pygame.K_RIGHT]:
            self.racecar.turn_right(turn_speed)
            
        # Update racecar's velocity and position
        self.racecar.apply_drag(drag)
        self.racecar.move()

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

    def _render_game(self):
        """
        Render all game components on the screen including background, racecar, 
        racetrack, rewards, and the game's scoreboard.
        """
        # Draw the background and the racecar
        self.screen.blit(self.background, (0, 0))
        self.racecar.draw(self.screen)
        self.racecar.vision_line_distance(self.racetrack.lines)

        # Draw racetrack lines and rewards if the drawing module is turned on
        if self.draw_toggle: 
            if self.rt_reward_toggle == "Racetrack":
                self.drawing_module.draw_rt_lines(self.screen)
            else:
                self.drawing_module.draw_rewards(self.rewardcoin, self.screen)

        # Draw the lines of the racetrack
        for line in self.racetrack.lines:
            pygame.draw.line(self.screen, (255, 0, 0), line[0], line[1], 3)

        # Draw the next coin on the track
        next_coin = self.score % len(self.racetrack.rewards)
        next_coin_x, next_coin_y = self.racetrack.rewards[next_coin]
        coin_width, coin_height = self.rewardcoin.get_size()
        self.screen.blit(self.rewardcoin, (next_coin_x - coin_width // 2, next_coin_y - coin_height // 2))

        # Render and display the scoreboard
        frame_rate = int(self.clock.get_fps())
        elapsed_time = (pygame.time.get_ticks() - self.start_time) // 1000

        # Create a semi-transparent background for the scoreboard
        scoreboard_bg = pygame.Surface((200, 60))
        scoreboard_bg.fill((0, 0, 0))
        scoreboard_bg.set_alpha(150)

        # Render the text for the scoreboard
        score_time_text = self.font.render(f"Time: {elapsed_time}s     Score: {self.score}", True, (255, 255, 255))
        fps_attempt_text = self.font.render(f"FPS: {frame_rate}     Attempt: {self.attempt}", True, (255, 255, 255))

        # Draw the scoreboard text on the background and render it on the screen
        scoreboard_bg.blit(score_time_text, (10, 10))
        scoreboard_bg.blit(fps_attempt_text, (10, 35))
        self.screen.blit(scoreboard_bg, (self.screen.get_width() - 210, 10))

        pygame.display.flip()
