# File to call when creating game session that loads in my 'session assets' rather than loading it each time the game starts
import pygame
import os
import csv
import rtree
import shapely.geometry as geom
import numpy as np


def load_session_assets(gamesettings):
    """
    Loads in all assets that will persist across games in the session
    Returns dictionary of assets
    """
    assets = {
        "GameBackground": _load_sprite("RacetrackSprite", False),
        "RacecarSprite": _shrink_sprite(_load_sprite("RacecarSprite"), 0.15),
        "RacecarCorners": None,
        "CoinSprite": _shrink_sprite(_load_sprite("MarioCoin"), 0.02),
        "CoinRadius": None, 
        "RacetrackLines": _load_lines_from_csv(),
        "RewardLocations": _load_rewards_from_csv(),
        "RacetrackLines_GridMap": None,
        "GridMap_RTree": None,
        "GridMap_Boxes": None
    }

    assets["RacecarCorners"] = _calc_racecar_corners(assets["RacecarSprite"])
    assets["CoinRadius"] = _get_coin_radius(assets["CoinSprite"])

    grid_map, rtree, grid_boxes = _init_grid_map(gamesettings['grid_dims'], assets["RacetrackLines"])
    assets["RacetrackLines_GridMap"] = grid_map
    assets["GridMap_RTree"] = rtree
    assets["GridMap_Boxes"] = grid_boxes

    return assets


def _load_sprite(name, with_alpha=True):
    """Loads in the sprite from the assets folder."""
    path = os.path.join("assets/graphics", f"{name}.png")
    loaded_sprite = pygame.image.load(path)
    return loaded_sprite.convert_alpha() if with_alpha else loaded_sprite.convert()


def _shrink_sprite(sprite, scale):
    """Scales down the sprite."""
    return pygame.transform.scale(sprite, tuple(int(dim * scale) for dim in (sprite.get_width(), sprite.get_height())))


def _calc_racecar_corners(sprite):
  """"Calculate the corners wrt the center of the racecar sprite for easier future calculation
  Returns [2x4] np.ndarray"""
  # Since we start car pointing right to align w/ unit circle for rotations, width right here really means 'height'
  width = sprite.get_height()
  height = sprite.get_width()

  # Calculate relative corners from center piece
  front_left = [width/2, height/2]
  front_right = [-width/2, height/2]
  bottom_left = [width/2, -height/2]
  bottom_right = [-width/2, -height/2]

  return np.array([front_left, front_right, bottom_left, bottom_right]).T


def _get_coin_radius(sprite):
    """Simple getter for the coin's radius"""
    return max(sprite.get_rect().width, sprite.get_rect().height)/2

    
def _load_lines_from_csv():
    """Loads race track lines from a CSV file."""
    lines = []
    filename = os.path.join("assets/track", "drawn_racetrack-06.05.23-01.10.csv")
    with open(filename, "r", newline="") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            temp_line = [tuple(map(int, point.strip('()').split(', '))) for point in row]
            lines.append(temp_line)
    return lines


def _load_rewards_from_csv():
    """Loads rewards from a CSV file."""
    rewards = []
    filename = os.path.join("assets/track", "drawn_reward-06.05.23-01.32.csv")
    with open(filename, "r", newline="") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            rewards.append((int(row[0]), int(row[1])))
    return rewards


def _init_grid_map(dims, lines):
    """
    Batch lines in the racetrack into different gridboxes to enable more efficient calculation of distances and intersections in-game. Returns:
        "grid_map": Dictionary of which lines in assets["RacetrackLines"] are in each grid_box
        "rtree": A spatial map to enable quick lookup of which gridbox our racecar is in mid-game
        "grid_boxes": Dictionary of shapely.geometry.Polygon objects that represent each box. Similarly used for quick lookup of which gridbox our racecar is in mid-game 
    """
    grid_r, grid_c = dims
    
    # First calculate min/max x and y values for determining step sizes
    np_lines = np.array(lines)
    min_max_x = (np.amin(np_lines[:, :, 0]), np.amax(np_lines[:, :, 0]))
    min_max_y = (np.amin(np_lines[:, :, 1]), np.amax(np_lines[:, :, 1]))
    step_x = (min_max_x[1] - min_max_x[0]) / grid_c
    step_y = (min_max_y[1] - min_max_y[0]) / grid_r
    
    # Then let's set up a subfunction for future calculations
    def _get_grid_cell(ulx, uly):
        """Sub-function that returns the different corners for the grid
        given an upper-left x and upper-left y coord (wrt PyGame's screen)"""
        ul = (ulx, uly)
        ur = (ulx + step_x, uly)
        ll = (ulx, uly + step_y)
        lr = (ulx + step_x, uly + step_y)
        return geom.Polygon([ul, ur, lr, ll])
    
    # Now for creating our grid map. Iterate through each gridbox, starting at the top left of the screen (idx=0) and moving right until hitting idx=grid_c-1, 
    # then moving down to 1 grid box below the first box we calculated and continuing until the bottom right
    grid_map = {}
    grid_boxes = {}
    for r in range(grid_r):
        for c in range(grid_c):
            rt_lines = []
            ulx = min_max_x[0] + c*step_x
            uly = min_max_y[0] + r*step_y
            grid_cell = _get_grid_cell(ulx, uly)
            
            for i, line in enumerate(lines):
              lstring = geom.LineString(line)
              if lstring.intersects(grid_cell):
                  rt_lines.append(i)
            
            grid_map[c + r*grid_c] = rt_lines
            grid_boxes[c + r*grid_c] = grid_cell
    
    # Create an R-tree spatial index to return. We'll use this to calculate w/ grid the car is in efficiently
    grid_rtree = rtree.index.Index()
    for idx, polygon in grid_boxes.items():
      grid_rtree.insert(idx, polygon.bounds)
    
    return grid_map, grid_rtree, grid_boxes
