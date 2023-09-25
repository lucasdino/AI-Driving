from game import RaceGame


class Racegame_Session:
    """Parent class that keeps track of meta variables for the racegame and allows for neural net training"""

    def __init__(self, gamesettings, session_metadata):
        self.session_metadata = session_metadata
        self.gamesettings = gamesettings
        self.racegame, self.nn_model = None, None
        
    def reset_racegame(self):
        """Method to reset the Racegame; before resetting, ensure attempts and wins is correctly updated"""
        self.session_metadata = self.racegame.session_metadata
        self.create_racegame()
        
    def model_setter(self, nn_model):
        """Simple setter function to set a nn_model up in the racegame session class"""
        self.nn_model = nn_model

    def create_racegame(self):
        """Launch a new instance of a racegame"""
        self.racegame = RaceGame(self.gamesettings, self.nn_model, self.session_metadata)