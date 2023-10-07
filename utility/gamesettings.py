# File with the meta-settings for the game to run - defined here and imported into necessary modules for ease / organization
from utility.assets_dicts import *

# ------------------------------------------
# Additional information about game settings
# ------------------------------------------
# screensize - Tuple relating to the screensize of our PyGame window
# grid_dims - [r x c] of how many grid boxes we will use in our gridding of the racetrack lines
# human_ai_toggle - Set who will be driving the car (player input or AI via training or inference)
# train_infer_toggle - Set 'TRAIN' if model is training; otherwise, set 'INFER' if model weights are in 'assets/nn_params' and you'd like model to drive
# draw_toggle - Set 'True' to enable drawing of either the racetrack or rewards. False disables the drawing module
# racetrack_reward_toggle - If DRAW_TOGGLE = True, then set as 'RACETRACK' or 'REWARD'. This determines what you can draw by using your mouse in the game
# display_hitboxes - Toggle to turn on displaying hitboxes or not
# display_arrows - Toggle to turn on the arrow key displays
# random_coin_start - Toggle to have car start at random coin location to begin or to start at starting line
# gamemap - Choose which gamemap to have the car drive on


gamesettings = {
    'screensize': None,                         # To be filled in when loading game map                   
    'grid_dims':  None,                         # To be filled in when loading map
    'human_ai_toggle': 'AI',                 # {'AI', 'HUMAN'}
    'train_infer_toggle': 'INFER',              # {'TRAIN', 'INFER'} -> Only used if 'human_ai_toggle' = 'AI'
    'draw_toggle': False,                       # {'True', 'False'}
    'racetrack_reward_toggle': 'REWARD',        # {'RACETRACK', 'REWARD'} -> Only used if draw mode is set to 'True'
    'display_hitboxes': True,                   # {'True', 'False'}
    'display_arrows': True,                     # {'True', 'False'}
    'random_coin_start': True,                  # {'True', 'False'}
    'gamemap': underwater_dalle_assets_dict     # {OGmap_assets_dict, cartoon_rt_assets_dict, google_maps_track_assets_dict, underwater_dalle_assets_dict}
}


# Data that will be held at the session-level and will be updated and reshared with the racegame each time a new racegame is initialized

session_metadata = {
    'attempts': 0,
    'wins': 0,
    'manual_override': False
}