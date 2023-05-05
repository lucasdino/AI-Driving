# File to help streamline loading assets into the game
import pygame
import os
import math

def load_sprite(name, with_alpha=True):
    path = os.path.join("assets", name+".png")
    loaded_sprite = pygame.image.load(path)

    if with_alpha:
        return loaded_sprite.convert_alpha()
    else:
        return loaded_sprite.convert()

def shrink_sprite(sprite, scale):
    new_width = int(sprite.get_width() * scale)
    new_height = int(sprite.get_height() * scale)
    return pygame.transform.scale(sprite, (new_width, new_height))

# Helper function for 'sprite_to_lines'
def rotate_point(point, center, angle):
    x, y = point[0] - center[0], point[1] - center[1]
    x_new = x * math.cos(math.radians(angle)) - y * math.sin(math.radians(angle))
    y_new = x * math.sin(math.radians(angle)) + y * math.cos(math.radians(angle))
    return center[0] + x_new, center[1] + y_new

# Convert sprite into set of lines that set the borders
def sprite_to_lines(sprite_rect, width, height, angle):
    center = sprite_rect.center 
    
    # Calculate front left coordinates
    fl_x = center[0] + (math.cos(math.radians(angle))*(height/2)) - (math.cos(math.radians(90-angle))*(width/2))
    fl_y = center[1] + (math.sin(math.radians(angle))*(height/2)) + (math.sin(math.radians(90-angle))*(width/2))
    front_left = (fl_x, fl_y)
    
    # Calculate front right coordinates
    fr_x = center[0] + (math.cos(math.radians(angle))*(height/2)) + (math.cos(math.radians(90-angle))*(width/2))
    fr_y = center[1] + (math.sin(math.radians(angle))*(height/2)) - (math.sin(math.radians(90-angle))*(width/2))
    front_right = (fr_x, fr_y)
    
    # Calculate bottom right coordinates
    br_x = center[0] - (math.cos(math.radians(angle))*(height/2)) + (math.cos(math.radians(90-angle))*(width/2))
    br_y = center[1] - (math.sin(math.radians(angle))*(height/2)) - (math.sin(math.radians(90-angle))*(width/2))
    bottom_right = (br_x, br_y)
    
    # Calculate bottom left coordinates
    bl_x = center[0] - (math.cos(math.radians(angle))*(height/2)) - (math.cos(math.radians(90-angle))*(width/2))
    bl_y = center[1] - (math.sin(math.radians(angle))*(height/2)) + (math.sin(math.radians(90-angle))*(width/2))
    bottom_left = (bl_x, bl_y)

    return [front_left, front_right, bottom_left, bottom_right]
    # return [(front_left, front_right), (front_right, bottom_right), (bottom_right, bottom_left), (bottom_left, front_left)]
