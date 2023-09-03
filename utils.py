import os
import pygame
import datetime
import numpy as np
from math import sin, cos, radians, sqrt, hypot
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

    # If the intersection point is None, return an arbitrarily large number to prevent future busts
    else:
        return 1000


def keypress_to_action(keys, ai_running=False):
    """Takes in a list of binaries relating to keys - convert list to action"""
    # Human playing - add functionality for all 9 keys; default (i.e., nothing is pressed) is 'do nothing'
    
    action_set_size = 5 if ai_running else 9
    action = [0]*action_set_size       
    
    if ai_running:
        if keys[pygame.K_LEFT]: action[0] = 1
        elif keys[pygame.K_UP]: action[1] = 1
        elif keys[pygame.K_RIGHT]: action[2] = 1
        elif keys[pygame.K_DOWN]: action[3] = 1
        else: action[4] = 1
    else:
        if keys[pygame.K_UP] and keys[pygame.K_LEFT]: action[5] = 1
        elif keys[pygame.K_UP] and keys[pygame.K_RIGHT]: action[6] = 1
        elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]: action[7] = 1
        elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]: action[8] = 1
        elif keys[pygame.K_LEFT]: action[0] = 1
        elif keys[pygame.K_UP]: action[1] = 1
        elif keys[pygame.K_RIGHT]: action[2] = 1
        elif keys[pygame.K_DOWN]: action[3] = 1
        else: action[4] = 1

    return action


def action_to_motion(racecar, action, acceleration, turn_speed, ai_running=False):
    """Converts list of actions to racecar motion"""

    # Handle movement based on input from either human or AI
    if action[0]:
        racecar.turn_left(turn_speed)
    if action[1]:
        racecar.accelerate(acceleration)
    if action[2]:
        racecar.turn_right(turn_speed)
    if action[3]:
        racecar.brake(acceleration)
    if action[4]:
        pass    # Do nothing

    # If human is driving, add functionality for these other keys; makes for better gameplay
    if not ai_running:
        if action[5]:
            racecar.accelerate(acceleration)
            racecar.turn_left(turn_speed)
        if action[6]:
            racecar.accelerate(acceleration)
            racecar.turn_right(turn_speed)
        if action[7]:
            racecar.brake(acceleration)
            racecar.turn_left(turn_speed)
        if action[8]:
            racecar.brake(acceleration)
            racecar.turn_right(turn_speed)


def game_exit_or_drawing(events, draw_toggle, racetrack_reward_toggle, drawing_module):
    for event in events:
        # Handle 'quit' events
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            if draw_toggle:
                formatted_datetime = datetime.datetime.now().strftime("%m.%d.%y-%H.%M")
                drawing_module.save_drawing_to_csv("drawn_" + racetrack_reward_toggle.lower() + "-" + formatted_datetime)
            quit()

        # Handle drawing events if drawing is toggled on
        if draw_toggle:
            if racetrack_reward_toggle == "RACETRACK":
                drawing_module.handle_rt_drawing_events(event)
            else:
                drawing_module.handle_reward_drawing_events(event)


# Defining variables externally that don't need to be recreated each time
toggle_render_button_rect = pygame.Rect(10, 10, 80, 20)
save_weights_button_rect = pygame.Rect(10, 40, 80, 20)

RED = (255, 0, 0)
PURPLE = (0, 255, 255)



def render_toggle(screen, render_status, click_eligible = True, toggle_save = False):
    """Class that lets the AI Training turn on and off rendering to speed up the training process but allow a window to peak in on"""

    font = pygame.font.Font(None, 14)
    
    if render_status:
        color = PURPLE
    else:
        color = RED

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            quit()
        if ((event.type == pygame.MOUSEBUTTONDOWN) and click_eligible):
            click_eligible = False
            if toggle_render_button_rect.collidepoint(event.pos):
                render_status = not render_status
            if save_weights_button_rect.collidepoint(event.pos):
                toggle_save = True
        if (event.type == pygame.MOUSEBUTTONUP):
            click_eligible = True
    
    render_text = "Render: ON" if render_status else "Render: OFF"
    render_text_font = font.render(render_text, True, (255, 255, 255))
    save_text = font.render("Save Weights", True, (255, 255, 255))
    
    pygame.draw.rect(screen, color, toggle_render_button_rect)
    pygame.draw.rect(screen, color, save_weights_button_rect)
    screen.blit(render_text_font, (12,13))
    screen.blit(save_text, (12,43))

    pygame.display.flip()
    return render_status, click_eligible, toggle_save
    

def calc_velocity(vel_vector):
    return sqrt(vel_vector[0]**2 + vel_vector[1]**2)


def manual_override_check(key, click_eligible, manual_override):
    """Class to return boolean if we want to manually override"""
    if not (key[pygame.K_q] == 1):
        click_eligible = True
    if (key[pygame.K_q] == 1 and click_eligible):
        click_eligible = False
        manual_override = not manual_override


    return manual_override, click_eligible


def scale_list(list, clipped_dist):
    """Function that takes in a list, a length that we'll remove (adjustment) from the distance to account for the hitbox, and a max 'clipped' distance to remove impact of large outliers"""
    adj_list = [min(max(x, 0.1), clipped_dist) for x in list]
    max_val = max(adj_list)
    scaled_list = [i/max_val for i in adj_list]

    return scaled_list


def return_magnitude(vector):
    return hypot(vector[0], vector[1])