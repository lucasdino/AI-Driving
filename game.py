import pygame
import sys
import datetime
from utils import load_sprite, shrink_sprite
from models import Racecar, Racetrack
from drawing_module import Drawing


class Driver:
    
    # Initialize master class intance, creating pygame instance and drawing game screen
    def __init__(self, attempt):
        """"Initialize the game class and set up necessary game variables"""
        self._init_pygame()
        self.screen = pygame.display.set_mode((800, 600))
        
        # Load in backgound, racecar, and racetrack assets
        self.background = load_sprite("RacetrackSprite", False)
        self.racecar = Racecar((72,356), shrink_sprite(load_sprite("RacecarSprite"), 0.15), (0,0))
        self.racetrack = Racetrack()
        self.rewardcoin = shrink_sprite(load_sprite("MarioCoin"), 0.015)

        # Pull in clock and create a font object for displaying text on screen
        self.clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()
        self.font = pygame.font.Font(None, 20)
        self.score = 0
        self.attempt = attempt

        self.running = True

        # Toggle to turn on or off the various states of the game
        self.draw_toggle = False      # Toggle to turn on the racetrack drawing feature
        self.rt_reward_toggle = "Reward"        # Either type 'Reward' or 'Racetrack' to set up the type of drawing module that will be deployed
        if self.draw_toggle: self.drawing_module = Drawing()


    # Main game loop function which will call on other game loop classes
    def main_loop(self):
        self._handle_input()            # Accepts user inputs (like key changes)
        self._process_game_logic()      # Adjusts / acts on all game assets
        self._draw()                    # Re-renders screen
        self.clock.tick(40)             # Cap frame rate at 40fps


    # Creates pygame instance
    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("AI Driver")


    # Class that accepts user inputs (like pressing keys, etc.)
    def _handle_input(self):

        # Close the window if you hit the 'x' button or 'Esc' key
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):

                # If drawing module is turned on save lines / rewards to CSV
                formatted_datetime = datetime.datetime.now().strftime("%m.%d.%y-%H.%M")
                if self.draw_toggle: self.drawing_module.save_drawing_to_csv(self.rt_reward_toggle + "_drawing-" + formatted_datetime)
                quit()
        
            # If drawing module is toggled on, call on the drawing module
            if self.draw_toggle: 
                if self.rt_reward_toggle == "Racetrack":
                    self.drawing_module.handle_rt_drawing_events(event)
                else:
                    self.drawing_module.handle_reward_drawing_events(event)
        
        # Code to listen for keyboard presses
        acceleration = 0.3     # Set speed of any accel / decel
        turn_speed = 5.0      # Set speed for turning (in degrees)
        drag = 0.9           # Set drag factor to simulate braking
        keys = pygame.key.get_pressed()


        # Update racecar position based on pressed keys
        if keys[pygame.K_UP]:
            self.racecar.accelerate(acceleration)
        if keys[pygame.K_DOWN]:
            self.racecar.brake(acceleration)
        if keys[pygame.K_LEFT]:
            self.racecar.turn_left(turn_speed)
        if keys[pygame.K_RIGHT]:
            self.racecar.turn_right(turn_speed)

        # Update racecar velocity and position
        self.racecar.apply_drag(drag)
        self.racecar.move()
            

    # Class that acts on all the game assets
    def _process_game_logic(self):
        self.racecar.move()

        # Check for collisions with the walls
        for line in self.racetrack.lines:
            if(self.racecar.check_line_collision(line)): 
                self.running = False

    # Class to re-render the screen
    def _draw(self):
        self.screen.blit(self.background, (0, 0))
        self.racecar.draw(self.screen)
        self.racecar.vision_line_distance(self.racetrack.lines)
        
        # If drawing module is turned on, draw lines / rewards on the screen
        if self.draw_toggle: 
            if self.rt_reward_toggle == "Racetrack":
                self.drawing_module.draw_rt_lines(self.screen)
            else:
                self.drawing_module.draw_rewards(self.rewardcoin, self.screen)

        for line in self.racetrack.lines:
            pygame.draw.line(self.screen, (255, 0, 0), line[0], line[1], 3)

        # Print out the scoreboard with time, score, and FPS

        # Calc values
        actual_frame_rate = int(self.clock.get_fps())
        elapsed_time = (pygame.time.get_ticks() - self.start_time) // 1000
        score = self.score 

        # Create the scoreboard background
        scoreboard_bg = pygame.Surface((200, 60))
        scoreboard_bg.fill((0, 0, 0))
        scoreboard_bg.set_alpha(150)  # Make the background slightly transparent

        # Render the timer and score text
        timer_score_text = self.font.render(f"Time: {elapsed_time}s     Score: {score}", True, (255, 255, 255))
        fps_attempt_text = self.font.render(f"FPS: {actual_frame_rate}     Attempt: {self.attempt}", True, (255, 255, 255))
        
        # Draw the text on the scoreboard background
        scoreboard_bg.blit(timer_score_text, (10, 10))
        scoreboard_bg.blit(fps_attempt_text, (10, 35))
        self.screen.blit(scoreboard_bg, (self.screen.get_width() - 210, 10))

        pygame.display.flip()