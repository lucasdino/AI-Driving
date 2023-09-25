from game import RaceGame


class Racegame_Session:
    """Parent class that keeps track of meta variables for the racegame and allows for neural net training"""

    def __init__(self, gamesettings):
        self.attempts, self.wins = 0, 0
        self.gamesettings = gamesettings
        self.manual_override = False
        self.racegame, self.nn_model = None, None
        
    def reset_racegame(self):
        """Method to reset the Racegame; before resetting, ensure attempts and wins is correctly updated"""
        self.attempts = self.racegame.attempt
        self.wins = self.racegame.wins
        self.manual_override = self.racegame.manual_override
        self.create_racegame()
        
    def model_setter(self, nn_model):
        """Simple setter function to set a nn_model up in the racegame session class"""
        self.nn_model = nn_model

    def create_racegame(self):
        """Launch a new instance of a racegame"""
        self.racegame = RaceGame(self.attempts, self.wins, self.gamesettings, self.nn_model, self.manual_override)