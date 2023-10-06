import pygame
import datetime
import numpy as np
import shapely.geometry as geom


def get_rotated_rc_lines(center, relative_corners, angle):
    """Returns list of tuples relating to the corners of the racecar"""    
    rotation_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle),  np.cos(angle)]])
    rotated_rel_corners = rotation_matrix @ relative_corners            # relative_corners is a [2x4] matrix of the corners wrt the center
    rotated_corners = rotated_rel_corners + center.T                    # center is [1x2] matrix -> need to transpose then it will broadcast

    front_left = (rotated_corners[0,0], rotated_corners[1,0])
    front_right = (rotated_corners[0,1], rotated_corners[1,1])
    bottom_left = (rotated_corners[0,2], rotated_corners[1,2])
    bottom_right = (rotated_corners[0,3], rotated_corners[1,3])
    
    return [(front_left, front_right), (front_right, bottom_right), (bottom_right, bottom_left), (bottom_left, front_left)]


def is_counterclockwise(A, B, C):
    """Determines if points A, B, C are ordered in a counterclockwise direction."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def line_intersection(line1, line2):
    """Computes the intersection point of two lines."""
    intersect_point = geom.LineString(line1).intersection(geom.LineString(line2))
    if intersect_point.is_empty:
        return None
    elif intersect_point.geom_type == 'Point':
        return (intersect_point.x, intersect_point.y)
    else:    
        return None


def nearest_line_distance(center, angle, racetrack_line):
    """Computes the distance to the nearest line from a point at a given angle."""
    
    # Calculate the unit direction vector and extend it from the center point
    direction = np.array([[np.sin(angle), np.cos(angle)]])
    line = [center, (center[0] + 1000*direction[0,0], center[1] + 1000*direction[0,1])]

    # Calculate the intersection point between the extended line and the given line segment
    intersection = line_intersection(line, racetrack_line)

    # If the intersection point is not None, calculate the distance from the center point to the intersection point
    # If the intersection point is None, return 200 as that is the extent of our distance we see given the grid system
    if intersection is not None: return np.hypot(*np.subtract(center, intersection))
    else: return 200


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


def manual_override_check(key, click_eligible, manual_override):
    """Class to return boolean if we want to manually override"""
    if not (key[pygame.K_q] == 1):
        click_eligible = True
    if (key[pygame.K_q] == 1 and click_eligible):
        click_eligible = False
        manual_override = not manual_override

    return manual_override, click_eligible


def standardize_data(data):
    """Func that takes an input and returns the same input but scaled to mean=0 and std_dev = 1"""
    adj_list = (data - np.mean(data)) / np.std(data)
    return adj_list.tolist()
    

def get_neighbor_lines(session_assets, grid_dims, racecar_center):
    """"Function to return the lines from the neighboring gridboxes to enable more efficient mid-game compute"""
    rc_point = geom.Point(racecar_center)

    # Find potential matches quickly using R-tree
    grid_id = 0
    for i in session_assets["GridMap_RTree"].intersection((rc_point.x, rc_point.y, rc_point.x, rc_point.y)):
        if rc_point.within(session_assets["GridMap_Boxes"][i]):
            grid_id = i
            break
    
    def _neighbor_boxes(id):
      grid_r, grid_c = grid_dims
      boxes = [id]
      
      left = (id%grid_c == 0)             # Means that 'id' box is in the far left column
      top = (id < grid_c)                 # Means that 'id' box is on the top row
      right = (id%grid_c == grid_c-1) 
      bottom = (id >= grid_c*(grid_r-1))

      if not left: boxes.append(id-1)
      if not top: boxes.append(id-grid_c)
      if not right: boxes.append(id+1)
      if not bottom: boxes.append(id+grid_c)
      if ((not left) and (not top)): boxes.append(id-grid_c-1)
      if ((not left) and (not bottom)): boxes.append(id+grid_c-1)
      if ((not right) and (not top)): boxes.append(id-grid_c+1)
      if ((not right) and (not bottom)): boxes.append(id+grid_c+1)

      return boxes
    
    lines_to_check = []
    for i in _neighbor_boxes(grid_id):
      lines_to_check.extend(session_assets["RacetrackLines_GridMap"].get(i, []))
    
    return [session_assets["RacetrackLines"][idx] for idx in list(set(lines_to_check))]