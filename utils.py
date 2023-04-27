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
def sprite_to_lines(sprite_rect, angle):
    center = sprite_rect.center
    # top_left = rotate_point(sprite_rect.topleft, center, angle),
    # top_right = rotate_point(sprite_rect.topright, center, angle),
    # bottom_right = rotate_point(sprite_rect.bottomright, center, angle),
    # bottom_left = rotate_point(sprite_rect.bottomleft, center, angle)
    top_left = sprite_rect.topleft
    top_right = sprite_rect.topright
    bottom_left = sprite_rect.bottomleft
    bottom_right = sprite_rect.bottomright


    return [top_left, top_right, bottom_left, bottom_right]
    # return [(top_left, top_right), (top_right, bottom_right), (bottom_right, bottom_left), (bottom_left, top_left)]
