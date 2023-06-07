import os
import pygame
import numpy as np
from math import sin, cos, radians
from shapely.geometry import LineString


def load_sprite(name, with_alpha=True):
    """Loads in the sprite from the assets folder."""
    path = os.path.join("assets", f"{name}.png")
    loaded_sprite = pygame.image.load(path)
    return loaded_sprite.convert_alpha() if with_alpha else loaded_sprite.convert()


def shrink_sprite(sprite, scale):
    """Scales down the sprite."""
    return pygame.transform.scale(sprite, tuple(int(dim * scale) for dim in (sprite.get_width(), sprite.get_height())))


def rotate_point(point, center, angle):
    """Rotates a point around a center by a given angle."""
    angle = radians(angle)
    dx, dy = np.subtract(point, center)
    new_dx = dx * cos(angle) - dy * sin(angle)
    new_dy = dx * sin(angle) + dy * cos(angle)
    return center[0] + new_dx, center[1] + new_dy

def sprite_to_lines(sprite_rect, width, height, angle):
    """Convert sprite into a list of lines that draw the border of the car and a list that draw the vision lines"""    
    center = sprite_rect.center

    # Calculate front left coordinates
    fl_x = center[0] - (cos(radians(angle))*(height/2)) + (cos(radians(90-angle))*(width/2))
    fl_y = center[1] + (sin(radians(angle))*(height/2)) + (sin(radians(90-angle))*(width/2))
    front_left = (fl_x, fl_y)
    
    # Calculate front right coordinates
    fr_x = center[0] - (cos(radians(angle))*(height/2)) - (cos(radians(90-angle))*(width/2))
    fr_y = center[1] + (sin(radians(angle))*(height/2)) - (sin(radians(90-angle))*(width/2))
    front_right = (fr_x, fr_y)
    
    # Calculate bottom right coordinates
    br_x = center[0] + (cos(radians(angle))*(height/2)) - (cos(radians(90-angle))*(width/2))
    br_y = center[1] - (sin(radians(angle))*(height/2)) - (sin(radians(90-angle))*(width/2))
    bottom_right = (br_x, br_y)
    
    # Calculate bottom left coordinates
    bl_x = center[0] + (cos(radians(angle))*(height/2)) + (cos(radians(90-angle))*(width/2))
    bl_y = center[1] - (sin(radians(angle))*(height/2)) + (sin(radians(90-angle))*(width/2))
    bottom_left = (bl_x, bl_y)
    
    return [(front_left, front_right), (front_right, bottom_right), (bottom_right, bottom_left), (bottom_left, front_left)]

def is_counterclockwise(A, B, C):
    """Determines if points A, B, C are ordered in a counterclockwise direction."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def line_intersection(line1, line2):
    """Computes the intersection point of two lines."""
    try:
        intersect_point = LineString(line1).intersection(LineString(line2))
        return (intersect_point.x, intersect_point.y) if not intersect_point.is_empty else None
    except Exception:
        return None

def nearest_line_distance(center, angle, racetrack_line):
    """Computes the distance to the nearest line from a point at a given angle."""
    
    # Calculate the unit direction vector and extend it from the center point
    direction = np.array([np.sin(angle), np.cos(angle)])
    line = [center, (center[0] + 1000*direction[0], center[1] + 1000*direction[1])]

    # Calculate the intersection point between the extended line and the given line segment
    intersection = line_intersection(line, racetrack_line)

    # If the intersection point is not None, calculate the distance from the center point to the intersection point
    if intersection is not None:
        return np.hypot(*np.subtract(center, intersection))

    # If the intersection point is None, return infinity
    else:
        return float('inf')
