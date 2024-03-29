import pygame
import numpy as np
from utility.utils import *


class Racecar:
    """Class representing the racecar, which includes both the sprite and hit box of lines."""
    
    def __init__(self, session_assets, position, next_reward_coin):
        
        self.session_assets = session_assets
        self.position = np.array([[position[0], position[1]]])
        self.velocity = np.array([[0, 0]])

        self.hitbox_lines, self.vision_lines = [], []

        # Set up metavariables and align racecar to point to next reward coin
        self._instantiate_racecar(next_reward_coin)
        
        # Update car rotation so that it is facing the next coin
        self._update_rotated_sprite()

        self.modelinputs = {
            # Inputs to the neural network:
            #   vision_distances: List of the different distances to the nearest wall, in the various vision directions that are hardcoded in the calculate_vision_lines function
            #           This is important as we need to allow the model to learn not to collide with walls. Since these directions are always in terms of the cars' current orientation,
            #           the model can learn to associate turning / forward / backward with how that impacts these values
            #   rel_angle_to_reward: Value from -1 to 1, where if car is directly facing the reward this is 0; as you move further away, the value approaches 1 or -1
            #           Directly at 180 degrees away from the reward, there is a discontinuity where this value flips from -1 to 1. The reason for this is that by turning one direction (e.g., right),
            #           the value will always be getting larger. By turning the other way, the value will always be getting smaller. This was thought as the most convenient way
            #           to pass the relative angle through to the neural network so that it could learn
            #   distance_to_reward: Distance to the reward, measured as the center of the car to the center of the reward in pixels.
            #           Necessary value to pass through to the neural net as it is used in calculating the score given to the car; this way the neural network has the required information for calculating the reward function
            #   velocity_to_reward: Velocity, in pixels, of the magnitude of the velocity in the direction of the reward. 
            #           Calculated as the projected magnitude (dot product of Velocity Vector with Normalized Vector of the line going from the center of the racecar to the center of the coin) 
            #   last_action_index: We also append the last action to the end of our model state before passing into the neural net; this let's us add a smooth behavior reward to our reward function
            "vision_distances": [],
            "rel_angle_to_reward": 0,
            "distance_to_reward": 0,
            "velocity_to_reward": 0,
            "racecar_velocity": 0,
            "last_action_index": 0
        }


    def _update_rotated_sprite(self):
        self.rotated_sprite = pygame.transform.rotate(self.session_assets["RacecarSprite"], np.degrees(self.angle))
        self.hitbox_lines = get_rotated_rc_lines(self.position, self.session_assets["RacecarCorners"], self.angle+np.pi/2)
    

    def _instantiate_racecar(self, next_reward_coin):
        """Self-explanatory class name. Purpose is to set up necessary variables for later call and align racecar to point toward next reward coin"""
        self.angle_to_reward, self.racecar_to_reward_vector = None, None
        self.calculate_reward_line(next_reward_coin, False)
        self.angle = 2*np.random.random()*np.pi
        

    def draw(self, screen, display_hitboxes):
        """Draws the racecar lines and vision lines on the provided screen."""
        
        if display_hitboxes:
            for line in self.hitbox_lines: # Racecar hitbox
                pygame.draw.line(screen, (0, 0, 255), line[0], line[1], 4)
            
            for line in self.vision_lines:
                pygame.draw.line(screen, (200,100,0), line[0], line[1], 2)
            self._draw_reward_line(screen)
                
        screen.blit(self.rotated_sprite, self.rotated_sprite.get_rect(center=tuple(self.position[0])))


    def _draw_reward_line(self, screen):
        """Draw line going to next coin"""
        end_point = (
            self.position[0,0] + self.modelinputs['distance_to_reward'] * np.cos(self.angle_to_reward),
            self.position[0,1] - self.modelinputs['distance_to_reward'] * np.sin(self.angle_to_reward),
        )   
        pygame.draw.line(screen, (0, 255, 255), tuple(self.position[0]), end_point, 2)


    def move(self):
        """Moves the racecar and recalculates its hitbox."""
        self.position = self.position + self.velocity
        self._update_rotated_sprite()


    def accelerate(self, acceleration):
        """Accelerates the racecar based on the given acceleration."""
        dx = acceleration * np.cos(self.angle)
        dy = acceleration * np.sin(self.angle)
        self.velocity = self.velocity + np.array([[dx, -dy]])


    def reverse(self, deceleration):
        """Applies a braking force to the racecar based on the provided deceleration."""
        dx = deceleration * np.cos(self.angle)
        dy = deceleration * np.sin(self.angle)
        self.velocity = self.velocity - np.array([[dx, -dy]])
        

    def turn_left(self, turn_speed):
        """Turns the racecar to the left based on the provided turn_speed."""
        self.angle += turn_speed


    def turn_right(self, turn_speed):
        """Turns the racecar to the right based on the provided turn_speed."""
        self.angle -= turn_speed


    def apply_drag(self, drag):
        """Applies drag to the racecar, slowing it down."""
        self.velocity = self.velocity * drag


    def check_line_collision(self, line_list):
        """
        Checks for a collision between the racecar and a given racetrack line / reward coin.
        Returns True if a collision is detected, False otherwise.
        """
        for line in self.hitbox_lines:
            A, B = line_list[0], line_list[1]
            C, D = line
            if is_counterclockwise(A, C, D) != is_counterclockwise(B, C, D) and is_counterclockwise(A, B, C) != is_counterclockwise(A, B, D): 
                return True
        return False
    

    def calc_rel_angle_to_reward(self):
        """
        Angle to reward needs to be converted so that it is relative to where the angle of the car is pointing.
        If car is facing the reward, rel_angle_to_reward should be 0. As this rotates away, it will go to 1 or -1, where at 180 degrees there is a discontinuity and it jumps between 1 and -1
        This is helpful as the model can learn that always applying 'left' will decrease this value, or vice-versa for right
        """
        t_1 = (self.angle_to_reward+np.pi)
        t_2 = (self.angle+np.pi)%(2*np.pi)
        self.modelinputs['rel_angle_to_reward'] = (np.arctan2(np.sin(t_1 - t_2), np.cos(t_1 - t_2)))/np.pi 


    def calculate_vision_lines(self, grid_dims, session_assets):
        """
        Calculates the distance to the nearest racetrack line for vision lines.
        Updates self.visiondist and self.visionlines.
        """
        # Manually coding in the different vision lines - this creates more frequent lines at the front of the car and less frequent lines at the side / back. In rads
        # vision_line_angles = [0, np.pi/3, np.pi/2, (2*np.pi)/3, np.pi, (4*np.pi)/3, (3*np.pi)/2, (5*np.pi)/3]
        vision_line_angles = [0, np.pi/3, np.pi/2, (2*np.pi)/3, np.pi, (4*np.pi)/3, (5*np.pi)/3]

        center = (self.position[0,0], self.position[0,1])
        self.vision_lines = []
        self.modelinputs['vision_distances'] = []
        neighboring_lines = get_neighbor_lines(session_assets, grid_dims, center)

        for i in vision_line_angles:
            angle = self.angle + i
            # Start by calculating (based on angle) the dist (pixels) to car's hitbox. Then calculate the dist to the nearest racetrack wall (and subtract dist from hitbox
            # to get dist from hitbox to nearest wall)
            dist_to_car_hitbox = min(nearest_line_distance(center, angle, line) for line in self.hitbox_lines)
            self.modelinputs['vision_distances'].append(min(nearest_line_distance(center, angle, line) for line in neighboring_lines) - dist_to_car_hitbox)
            
            # Then calculate the coordinates for the points on the hitbox and on the wall
            hitbox_x = center[0] + dist_to_car_hitbox * np.sin(angle)
            hitbox_y = center[1] + dist_to_car_hitbox * np.cos(angle)
            wall_x = hitbox_x + self.modelinputs['vision_distances'][-1] * np.sin(angle)
            wall_y = hitbox_y + self.modelinputs['vision_distances'][-1] * np.cos(angle)
            
            self.vision_lines.append(((hitbox_x, hitbox_y), (wall_x, wall_y)))


    def calculate_reward_line(self, rewardcoin, update_model_inputs=True):
        """Calculates the distance and angle to the next reward coin and updates 'modelinputs'"""
        rewardcoin_x, rewardcoin_y = rewardcoin.center
        racecar_x, racecar_y = tuple(self.position[0])
        
        dx = racecar_x - rewardcoin_x
        dy = racecar_y - rewardcoin_y

        # Reset the vector_to_reward object. Need to negate the inputs so that the vector is pointing in the direction from the racecar to the reward (due to pygame's coordinate system)
        self.racecar_to_reward_vector = np.array([[-dx, -dy]])
        
        # Calculate angles for the car and the angle to the reward coin in radians; will convert this to sine when cleaning data later on
        self.angle_to_reward = np.arctan2(dy, -dx)
        
        # Calculate the distance between the two points; round to calc precision
        if update_model_inputs:
            self.modelinputs['distance_to_reward'] = np.hypot(dx, dy)


    def calculate_velocity_vectors(self):
        """Function to calculate two velocity vectors for store. One is the velocity to the reward, the other, the relative velocity of the car based on the direction it is facing"""
        # Calculating the absolute velocity vectors using dot products on normalized vectors that relate to the direction of the car
        v_norm_forward = np.array([[np.cos(self.angle), np.sin(self.angle)]])
        v_norm_right = np.array([[np.cos(self.angle - np.pi/2), np.sin(self.angle - np.pi/2)]])
        v_velocity_norm = np.array([[self.velocity[0,0], -self.velocity[0,1]]])             # Need to flip the y around since coordinates in pygame are backward
        self.modelinputs['racecar_velocity'] = [(v_velocity_norm@v_norm_forward.T).item(), (v_velocity_norm@v_norm_right.T).item()]
        
        # Calculating the magnitude of the velocity to the reward vector
        v_norm_to_reward = self.racecar_to_reward_vector / np.linalg.norm(self.racecar_to_reward_vector)
        self.modelinputs['velocity_to_reward'] = (self.velocity@v_norm_to_reward.T).item()


    def return_clean_model_state(self):
        """Function to convert the 'model inputs' dictionary into a 1-D array"""
        flat_clean_list = []
        for label, state in self.modelinputs.items():
            include = True

            # Start by cleaning the data so we pass through data in roughly the same scale    
            if "vision_distances" in label:
                state = standardize_data(state)
            elif "rel_angle_to_reward" in label:
                pass
            elif "distance_to_reward" in label:
                dist_less_radius = max(state-self.session_assets["CoinRadius"], 1)
                state = np.log(min(dist_less_radius/60, 2))
            elif "velocity_to_reward" in label:
                include = False
            elif "racecar_velocity" in label:
                pass
            elif "last_action_index" in label:
                new_list = [0] * 5
                new_list[state] = 1
                state = new_list

            if include:
                # Next, append the value to a list that we pass back to the neural network
                flat_clean_list.extend(state) if isinstance(state, list) else flat_clean_list.append(state)
        
        # rounded_list = [round(x, 2) for x in flat_clean_list]
        # print(f"{rounded_list[6:]}")
        
        return flat_clean_list


class Racetrack:
    """Class representing the racetrack, which consists of a list of border lines and rewards."""
    
    def __init__(self, start_at_random_coin, lines, rewards):
        self.lines, self.rewards = lines, rewards
        self.reward_coin_index = np.random.randint(len(self.rewards)) if start_at_random_coin else 0
        self.start_position = (self.rewards[self.reward_coin_index])
        self.update_reward_coin_index()


    def draw(self, screen, display_hitboxes):
        """Draw racetrack lines on screen"""
        if display_hitboxes:
            # Racetrack lines
            for line in self.lines:
                pygame.draw.line(screen, (255, 0, 0), line[0], line[1], 3)


    def get_next_reward_coin_index(self):
        """Simple getter function to return the next reward coin index"""
        return self.reward_coin_index+1 if self.reward_coin_index+1 < len(self.rewards) else 0
    

    def update_reward_coin_index(self):
        """Simple function to update 'self.reward_coin_index' so that once the car gets a coin, you can point to the next instance of a coin"""
        self.reward_coin_index  = self.get_next_reward_coin_index()


class RewardCoin:
    """Class representing the active reward coin on the track"""

    def __init__(self, session_assets, center):
        self.sprite = session_assets["CoinSprite"]
        self.radius = session_assets["CoinRadius"]
        self.center = center
        self.sprite_rect = self.sprite.get_rect(center=center)
        self.top_left = np.array([[center[0] - self.sprite_rect.height // 2, center[1] - self.sprite_rect.width // 2]])


    def draw(self, screen, display_hitboxes):
        """Draw coin on screen"""
        screen.blit(self.sprite, tuple(self.top_left[0]))
        if display_hitboxes:
            pygame.draw.circle(screen, (0, 255, 255), (self.center), self.radius, 3)


    def intersect_with_reward(self, distance_to_center):
        """Check if a given point intersects with the sprite rectangle"""
        return distance_to_center <= self.radius