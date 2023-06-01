# File to help streamline loading assets into the game
import pygame
import os
import math
import numpy as np
from shapely.geometry import LineString, Point

def load_sprite(name, with_alpha=True):
    """"Helper function that loads in the sprite from the assets folder"""
    path = os.path.join("assets", name+".png")
    loaded_sprite = pygame.image.load(path)

    if with_alpha:
        return loaded_sprite.convert_alpha()
    else:
        return loaded_sprite.convert()

def shrink_sprite(sprite, scale):
    """"Helper function that scales down the sprite"""
    new_width = int(sprite.get_width() * scale)
    new_height = int(sprite.get_height() * scale)
    return pygame.transform.scale(sprite, (new_width, new_height))

def rotate_point(point, center, angle):
    """Helper function for 'sprite_to_lines'"""
    x, y = point[0] - center[0], point[1] - center[1]
    x_new = x * math.cos(math.radians(angle)) - y * math.sin(math.radians(angle))
    y_new = x * math.sin(math.radians(angle)) + y * math.cos(math.radians(angle))
    return center[0] + x_new, center[1] + y_new

def sprite_to_lines(sprite_rect, width, height, angle):
    """Convert sprite into a list of lines that draw the border of the car and a list that draw the vision lines"""    
    center = sprite_rect.center
    
    # --------------------------------
    # Calculate the Sprite Lines
    # --------------------------------
    
    # Calculate front left coordinates
    fl_x = center[0] - (math.cos(math.radians(angle))*(height/2)) + (math.cos(math.radians(90-angle))*(width/2))
    fl_y = center[1] + (math.sin(math.radians(angle))*(height/2)) + (math.sin(math.radians(90-angle))*(width/2))
    front_left = (fl_x, fl_y)
    
    # Calculate front right coordinates
    fr_x = center[0] - (math.cos(math.radians(angle))*(height/2)) - (math.cos(math.radians(90-angle))*(width/2))
    fr_y = center[1] + (math.sin(math.radians(angle))*(height/2)) - (math.sin(math.radians(90-angle))*(width/2))
    front_right = (fr_x, fr_y)
    
    # Calculate bottom right coordinates
    br_x = center[0] + (math.cos(math.radians(angle))*(height/2)) - (math.cos(math.radians(90-angle))*(width/2))
    br_y = center[1] - (math.sin(math.radians(angle))*(height/2)) - (math.sin(math.radians(90-angle))*(width/2))
    bottom_right = (br_x, br_y)
    
    # Calculate bottom left coordinates
    bl_x = center[0] + (math.cos(math.radians(angle))*(height/2)) + (math.cos(math.radians(90-angle))*(width/2))
    bl_y = center[1] - (math.sin(math.radians(angle))*(height/2)) + (math.sin(math.radians(90-angle))*(width/2))
    bottom_left = (bl_x, bl_y)
    
    return [(front_left, front_right), (front_right, bottom_right), (bottom_right, bottom_left), (bottom_left, front_left)]

def ccw(A, B, C):
    """Collision Calculation - helper function to check if the lines are counterclockwise or not"""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def line_intersect(line1, line2):
    """Returns the intersection point of two lines"""
    line1 = LineString(line1)
    line2 = LineString(line2)
    intersect_point = line1.intersection(line2)
    if intersect_point.is_empty:
        return None
    else:
        return intersect_point.x, intersect_point.y

def distance_point_line(center, angle, racetrack_line):
    """Returns the distance to the nearest line based on the desired angle"""
    
    # Calculate the unit direction vector
    direction = np.array([np.sin(angle), np.cos(angle)])

    # Extend this direction from the center point to form a line
    line = [center, (center[0] + 1000*direction[0], center[1] + 1000*direction[1])]

    # Calculate the intersection point between this line and the given line segment
    intersection_point = line_intersect(line, racetrack_line)

    if intersection_point is None:
        return float('inf')
    else:
        # Calculate the distance from the center point to the intersection point
        distance = np.sqrt((center[0] - intersection_point[0])**2 + (center[1] - intersection_point[1])**2)
        return distance