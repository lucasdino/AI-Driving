import pygame
import sys
from utils import load_sprite, shrink_sprite
from models import Racecar


class Driver:
    
    # Initialize master class intance, creating pygame instance and drawing game screen
    def __init__(self):
        self._init_pygame()
        self.screen = pygame.display.set_mode((800, 600))
        self.background = load_sprite("RacetrackSprite", False)
        self.racecar = Racecar((58,336), shrink_sprite(load_sprite("RacecarSprite"), 0.2), (0,0))



    # Main game loop function which will call on other game loop classes
    def main_loop(self):
        while True:
            self._handle_input()            # Accepts user inputs (like key changes)
            self._process_game_logic()      # Adjusts / acts on all game assets
            self._draw()                    # Re-renders screen

    # Creates pygame instance
    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("AI Driver")

    # Class that accepts user inputs (like pressing keys, etc.)
    def _handle_input(self):
        for event in pygame.event.get():
            # Close the window if you hit the 'x' button or 'Esc' key
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                quit()


    # Class that acts on all the game assets
    def _process_game_logic(self):
        self.racecar.move()


    # Class to re-render the screen
    def _draw(self):
        self.screen.blit(self.background, (0, 0))
        self.racecar.draw(self.screen)


        if self.drawing_instance.start_point and self.drawing_instance.end_point:
            pygame.draw.line(self.screen, (255, 0, 0), self.drawing_instance.start_point, self.drawing_instance.end_point, 3)


        # print("Collides:", self.racecar.collides_with_rect(self.racecar2))
        pygame.display.flip()