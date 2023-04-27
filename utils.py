# File to help streamline loading assets into the game
import pygame
import os

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

def sprite_to_lines(sprite_rect):
    top_left = (sprite_rect.left, sprite_rect.top)
    top_right = (sprite_rect.right, sprite_rect.top)
    bottom_right = (sprite_rect.right, sprite_rect.bottom)
    bottom_left = (sprite_rect.left, sprite_rect.bottom)

    return [(top_left, top_right), (top_right, bottom_right), (bottom_right, bottom_left), (bottom_left, top_left)]
