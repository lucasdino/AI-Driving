import pygame
from game import RaceGame
from utility.sessionassets import get_background_scale, load_session_assets
from utility.assets_dicts import *


class Racegame_Session:
    """Parent class that keeps track of meta variables for the racegame and allows for neural net training"""

    def __init__(self, gamesettings, session_metadata):
        self.session_metadata = session_metadata
        self.assets_dict = OGmap_assets_dict
        self.gamesettings = gamesettings
        self.gamesettings["screensize"], self.gamesettings['grid_dims'], self.assets_dict["Background_Scaler"] = get_background_scale(self.assets_dict)
        self._initialize_pygame()
        self.session_assets = load_session_assets(self.assets_dict, gamesettings)
        self.racegame, self.nn_model = None, None
        
    def _initialize_pygame(self):
        """Initialize Pygame and set the window title."""
        pygame.init()
        pygame.display.set_caption("AI Driver")
        self.screen = pygame.display.set_mode(self.gamesettings['screensize'])

    def reset_racegame(self):
        """Method to reset the Racegame; before resetting, ensure attempts and wins is correctly updated"""
        self.session_metadata = self.racegame.session_metadata
        self.gamesettings = self.racegame.gamesettings
        self.create_racegame()
        
    def model_setter(self, nn_model):
        """Simple setter function to set a nn_model up in the racegame session class"""
        self.nn_model = nn_model

    def create_racegame(self):
        """Launch a new instance of a racegame"""
        self.racegame = RaceGame(self.screen, self.gamesettings, self.nn_model, self.session_metadata, self.session_assets)