# File with the meta-settings for the game to run - defined here and imported into necessary modules for ease / organization

# ------------------------------------------
# Additional information about game settings
# ------------------------------------------
# human_ai_toggle - Set who will be driving the car (player input or AI via training or inference)
# train_infer_toggle - Set 'TRAIN' if model is training; otherwise, set 'INFER' if model weights are in 'assets/nn_params' and you'd like model to drive
# draw_toggle - Set 'True' to enable drawing of either the racetrack or rewards. False disables the drawing module
# racetrack_reward_toggle - If DRAW_TOGGLE = True, then set as 'RACETRACK' or 'REWARD'. This determines what you can draw by using your mouse in the game
# display_hitboxes - Toggle to turn on displaying hitboxes or not
# display_arrows - Toggle to turn on the arrow key displays
# random_coin_start - Toggle to have car start at random coin location to begin or to start at starting line


gamesettings = {
    'human_ai_toggle': 'HUMAN',                 # {'AI', 'HUMAN'}
    'train_infer_toggle': 'TRAIN',              # {'TRAIN', 'INFER'} -> Only used if 'human_ai_toggle' = 'AI'
    'draw_toggle': False,                       # {'True', 'False'}
    'racetrack_reward_toggle': 'REWARD',        # {'RACETRACK', 'REWARD'} -> Only used if draw mode is set to 'True'
    'display_hitboxes': True,                   # {'True', 'False'}
    'display_arrows': True,                     # {'True', 'False'}
    'random_coin_start': True                   # {'True', 'False'}
}


# Data that will be held at the session-level and will be updated and reshared with the racegame each time a new racegame is initialized

session_metadata = {
    'attempts': 0,
    'wins': 0,
    'manual_override': False
}