from game import RaceGame

# HUMAN_AI_TOGGLE - set who will be driving the car (player input or AI via training or inference)
# TRAIN_INFER_TOGGLE - set 'TRAIN' if model is training; otherwise, set 'INFER' if model weights are in 'assets/nn_params' and you'd like model to drive
# DRAW_TOGGLE - Set 'True' to enable drawing of either the racetrack or rewards
# RACETRACK_REWARD_TOGGLE - If DRAW_TOGGLE = True, then set as 'RACETRACK' or 'REWARD' depending on what you'd like to draw with the mouse in-game
HUMAN_AI_TOGGLE = "AI"
TRAIN_INFER_TOGGLE = "TRAIN"
DRAW_TOGGLE = False
RACETRACK_REWARD_TOGGLE = "REWARD"

if HUMAN_AI_TOGGLE == "AI":
    from dqn_model import DQN_Model

# NN_MODEL - Instance of the DQN model object that can be passed into the racegame for inference
# SESSION - Declaring racegame variable here that will be assigned the Racegame_Session object that will be created
NN_MODEL = None
SESSION = None


class Racegame_Session:
    """Parent class that keeps track of meta variables for the racegame and allows for neural net training"""

    def __init__(self):
        self.attempts = 0
        self.wins = 0
        self.manual_override = False
        self.racegame = RaceGame(self.attempts, self.wins, DRAW_TOGGLE, RACETRACK_REWARD_TOGGLE, HUMAN_AI_TOGGLE, TRAIN_INFER_TOGGLE, NN_MODEL, self.manual_override)

    def reset_racegame(self):
        """Method to reset the Racegame; before resetting, ensure attempts and wins is correctly updated"""
        self.attempts = self.racegame.attempt
        self.wins = self.racegame.wins
        self.manual_override = self.racegame.manual_override
        self.racegame = RaceGame(self.attempts, self.wins, DRAW_TOGGLE, RACETRACK_REWARD_TOGGLE, HUMAN_AI_TOGGLE, TRAIN_INFER_TOGGLE, NN_MODEL, self.manual_override)



if __name__ == "__main__":
    # If AI is used, need to create instance of NN model
    if HUMAN_AI_TOGGLE == "AI":
        NN_MODEL = DQN_Model(TRAIN_INFER_TOGGLE)

    SESSION = Racegame_Session()

    # First case - if human will be driving
    if (HUMAN_AI_TOGGLE == "HUMAN"):
        while True:
            while SESSION.racegame.running:
                SESSION.racegame.human_main_loop()
            SESSION.reset_racegame()

    # Second case - if AI will be training
    elif (HUMAN_AI_TOGGLE == "AI" and TRAIN_INFER_TOGGLE == "TRAIN"):
        NN_MODEL.racegame_session_setter(SESSION)
        NN_MODEL.train_model()

    # Third case - if AI will be infering
    elif (HUMAN_AI_TOGGLE == "AI" and TRAIN_INFER_TOGGLE == "INFER"):
        while True:
            while SESSION.racegame.running:
                SESSION.racegame.ai_infer_loop()
            SESSION.reset_racegame()