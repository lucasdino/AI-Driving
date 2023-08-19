from game import RaceGame

racegame = None
attempts = 0
wins = 0

HUMAN_AI_TOGGLE = "HUMAN"               # Set who will be driving the car
TRAIN_INFER_TOGGLE = "TRAIN"            # If HUMAN_AI_TOGGLE = "AI" -> Set 'TRAIN' if model is training; set 'INFER' if model is driving
DRAW_TOGGLE = False                     # Toggle on the drawing module
RACETRACK_REWARD_TOGGLE = "RACETRACK"      # If DRAW_TOGGLE, set which you will be drawing with your mouse

if HUMAN_AI_TOGGLE == "AI":
    from dqn_model import DQN_Model
    model = DQN_Model(TRAIN_INFER_TOGGLE)
else:
    model = None


if __name__ == "__main__":

    while True:
        # Define RaceGame object and pass through metavariables    
        racegame = RaceGame(attempts, wins, DRAW_TOGGLE, RACETRACK_REWARD_TOGGLE, HUMAN_AI_TOGGLE, TRAIN_INFER_TOGGLE, model)

        if HUMAN_AI_TOGGLE == "HUMAN": 
            while racegame.running:
                racegame.human_main_loop()
        
        elif HUMAN_AI_TOGGLE == "AI":  
            if TRAIN_INFER_TOGGLE == "TRAIN":
                while racegame.running:
                    # Don't call racegame.ai_loop() since we will be stepping through during model training from the dqn_model file
                    model.racegame_setter(racegame)

            elif TRAIN_INFER_TOGGLE == "INFER":
                while racegame.running:
                    racegame.ai_infer_loop()

        # Once main game loop breaks, pull out the updated values for 'attempt' and for 'wins'
        attempts = racegame.attempt
        wins = racegame.wins