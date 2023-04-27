import pygame
import sys
from utils import load_sprite, shrink_sprite
from models import Racecar, Racetrack


class Driver:
    
    # Initialize master class intance, creating pygame instance and drawing game screen
    def __init__(self):
        self._init_pygame()
        self.screen = pygame.display.set_mode((800, 600))
        
        # Load in backgound, racecar, and racetrack assets
        self.background = load_sprite("RacetrackSprite", False)
        self.racecar = Racecar((72,356), shrink_sprite(load_sprite("RacecarSprite"), 0.15), (0,0))
        self.racetrack = Racetrack()

        # Pull in clock and create a font object for displaying text on screen
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)


    # Main game loop function which will call on other game loop classes
    def main_loop(self):
        while True:
            self._handle_input()            # Accepts user inputs (like key changes)
            self._process_game_logic()      # Adjusts / acts on all game assets
            self._draw()                    # Re-renders screen
            self.clock.tick(40)                  # Cap frame rate at 40fps


    # Creates pygame instance
    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("AI Driver")


    # Class that accepts user inputs (like pressing keys, etc.)
    def _handle_input(self):

        # Close the window if you hit the 'x' button or 'Esc' key
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                quit()
        
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


    # Class to re-render the screen
    def _draw(self):
        self.screen.blit(self.background, (0, 0))
        self.racecar.draw(self.screen)

        for line in self.racetrack.lines:
            pygame.draw.line(self.screen, (255, 0, 0), line[0], line[1], 3)

        # Calculate frame rate, render text on screen, and draw frame rate text on next screen
        actual_frame_rate = int(self.clock.get_fps())
        frame_rate_text = self.font.render(f"FPS: {actual_frame_rate}", True, (255, 255, 255))
        self.screen.blit(frame_rate_text, (670, 20))

        # print("Collides:", self.racecar.collides_with_rect(self.racecar2))
        pygame.display.flip()