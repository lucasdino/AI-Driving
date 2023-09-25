from session import Racegame_Session
from neuralnet.dqn_model import DQN_Model  
from utility.gamesettings import gamesettings, session_metadata


# NN_MODEL - Instance of the DQN model object that can be passed into the racegame for inference
# SESSION - Declaring racegame variable here that will be assigned the Racegame_Session object that will be created
nn_model = None
session = None

if __name__ == '__main__':

    # Start by launching a racegame session
    session = Racegame_Session(gamesettings, session_metadata)

    # First case - if human will be driving
    if (gamesettings['human_ai_toggle'] == 'HUMAN'):
        session.create_racegame()
        while True:
            while session.racegame.running:
                session.racegame.human_main_loop()
            session.reset_racegame()
    
    else:
        nn_model = DQN_Model(gamesettings)
        session.model_setter(nn_model)
        session.create_racegame()
        
        # Second case - if AI will be training
        if (gamesettings['train_infer_toggle'] == 'TRAIN'):
            nn_model.racegame_session_setter(session)
            nn_model.train_model()

        # Third case - if AI will be infering
        elif (gamesettings['train_infer_toggle'] == 'INFER'):
            while True:
                while session.racegame.running:
                    session.racegame.ai_infer_loop()
                session.reset_racegame()

        else: print("Please confirm 'gamesettings' variables are valid inputs.")