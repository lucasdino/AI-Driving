import pygame
import csv
import os
from math import sin, cos, radians, atan2, hypot, pi, exp
from pygame import Vector2, font, Surface
from utils import sprite_to_lines, is_counterclockwise, nearest_line_distance, scale_list
# from dqn_model import select_action


class Racecar:
    """Class representing the racecar, which includes both the sprite and hit box of lines."""
    
    def __init__(self, position, sprite, velocity):
        self.position = Vector2(position)
        self.velocity = Vector2(velocity)
        self.racecar_to_reward_vector = None
        self.angle = 90
        self.sprite = sprite
        self._update_rotated_sprite()
        self._sprite_w = self.rotated_sprite_rect.width
        self._sprite_l = self.rotated_sprite_rect.height
        self.linesegments = []
        self.vision_lines = []
        self.angle_to_reward = 0
        
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

        self.modelinputs = {
            "vision_distances": [],
            "rel_angle_to_reward": 0,
            "distance_to_reward": 0,
            "velocity_to_reward": 0
        }


    def _update_rotated_sprite(self):
        self.rotated_sprite = pygame.transform.rotate(self.sprite, self.angle)
        self.rotated_sprite_rect = self.rotated_sprite.get_rect(center=self.position)


    def draw(self, screen, display_hitboxes):
        """Draws the racecar lines and vision lines on the provided screen."""
        
        if display_hitboxes:
            # Racecar hitbox
            for line in self.linesegments:
                pygame.draw.line(screen, (0, 0, 255), line[0], line[1], 4)
            
            # Racecar vision lines and reward line
            for line in self.vision_lines:
                pygame.draw.line(screen, (200,100,0), line[0], line[1], 2)
            self._draw_reward_line(screen)
                
        screen.blit(self.rotated_sprite, self.rotated_sprite_rect)

    def _draw_reward_line(self, screen):
        """Draw line going to next coin"""
        end_point = (
            self.position[0] + self.modelinputs['distance_to_reward'] * cos(self.angle_to_reward),
            self.position[1] - self.modelinputs['distance_to_reward'] * sin(self.angle_to_reward),
        )   
        pygame.draw.line(screen, (0, 255, 255), self.position, end_point, 2)


    def move(self):
        """Moves the racecar and recalculates its hitbox."""
        self.position += self.velocity
        self._update_rotated_sprite()
        self.linesegments = sprite_to_lines(self.rotated_sprite_rect, self._sprite_w, self._sprite_l, self.angle)


    def accelerate(self, acceleration):
        """Accelerates the racecar based on the given acceleration."""
        dx = acceleration * cos(radians(self.angle))
        dy = acceleration * sin(radians(self.angle))
        self.velocity += Vector2(dx, -dy)


    def brake(self, deceleration):
        """Applies a braking force to the racecar based on the provided deceleration."""
        dx = deceleration * cos(radians(self.angle))
        dy = deceleration * sin(radians(self.angle))
        self.velocity -= Vector2(dx, -dy)


    def turn_left(self, turn_speed):
        """Turns the racecar to the left based on the provided turn_speed."""
        self.angle += turn_speed


    def turn_right(self, turn_speed):
        """Turns the racecar to the right based on the provided turn_speed."""
        self.angle -= turn_speed


    def apply_drag(self, drag):
        """Applies drag to the racecar, slowing it down."""
        self.velocity *= drag


    def check_line_collision(self, line_list):
        """
        Checks for a collision between the racecar and a given racetrack line / reward coin.
        Returns True if a collision is detected, False otherwise.
        """
        for line in self.linesegments:
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
        t_1 = (self.angle_to_reward+pi)
        t_2 = (radians(self.angle)+pi)%(2*pi)
        self.modelinputs['rel_angle_to_reward'] = (atan2(sin(t_1 - t_2), cos(t_1 - t_2)))/pi 


    def calculate_vision_lines(self, racetrack_line):
        """
        Calculates the distance to the nearest racetrack line for vision lines.
        Updates self.visiondist and self.visionlines.
        """
        # Manually coding in the different vision lines - this creates more frequent lines at the front of the car and less frequent lines at the side / back. In rads
        vision_line_angles = [0, pi/3, (2*pi)/5, pi/2, (3*pi)/5, (2*pi)/3, pi, (4*pi)/3, (7*pi)/5, (3*pi)/2, (8*pi)/5, (5*pi)/3]

        center = self.rotated_sprite_rect.center
        self.vision_lines = []
        self.modelinputs['vision_distances'] = []

        for i in vision_line_angles:
            angle = radians(self.angle) + i
            self.modelinputs['vision_distances'].append(min(nearest_line_distance(center, angle, line) for line in racetrack_line))
            temp_x = center[0] + self.modelinputs['vision_distances'][-1] * sin(angle)
            temp_y = center[1] + self.modelinputs['vision_distances'][-1] * cos(angle)
            self.vision_lines.append((center, (temp_x, temp_y)))
    

    def calculate_reward_line(self, rewardcoin):
        """Calculates the distance and angle to the next reward coin and updates 'modelinputs'"""
        rewardcoin_x, rewardcoin_y = rewardcoin.center
        racecar_x, racecar_y = self.position
        
        dx = racecar_x - rewardcoin_x
        dy = racecar_y - rewardcoin_y

        # Reset the vector_to_reward object. Need to negate the inputs so that the vector is pointing in the direction from the racecar to the reward
        self.racecar_to_reward_vector = Vector2(-dx, -dy)
        
        # Calculate angles for the car and the angle to the reward coin in radians; will convert this to sine when cleaning data later on
        self.angle_to_reward = atan2(dy,-dx)
        
        # Calculate the distance between the two points; round to calc precision
        self.modelinputs['distance_to_reward'] = hypot(dx, dy)


    def calculate_velocity_to_reward_vector(self):
        """Function to calculate the magnitude of the velocity toward the reward"""
        v_norm = self.racecar_to_reward_vector.normalize()
        self.modelinputs['velocity_to_reward'] = self.velocity.dot(v_norm)


    def return_clean_model_state(self, reward_coin_radius):
        """Function to convert the 'model inputs' dictionary into a 1-D array"""
        flat_clean_list = []
        for key, value in self.modelinputs.items():
            
            # Start by cleaning the data so we pass through data in roughly the same scale    
            if "vision_distances" in key:
                clipped_distance = 200
                value = scale_list(value, self._sprite_w, clipped_distance)
                # value = [exp(-((x-self._sprite_w)/20)) for x in value]
                # value = [max(x,1)/40 for x in value]
            elif "rel_angle_to_reward" in key:
                pass
            elif "distance_to_reward" in key:
                dist_less_radius = max(value-reward_coin_radius, 1)
                # value = dist_less_radius/40
                value = exp(-(dist_less_radius/20))
            elif "velocity_to_reward" in key:
                value = value / 3

            # Next, append the value to a list that we pass back to the neural network
            flat_clean_list.extend(value) if isinstance(value, list) else flat_clean_list.append(value)
        
        # rounded_list = [round(x, 2) for x in flat_clean_list]
        # print(rounded_list[:8])
        
        return flat_clean_list


class Racetrack:
    """Class representing the racetrack, which consists of a list of border lines and rewards."""
    
    def __init__(self, draw_toggle, racetrack_reward_toggle):
        self.lines = []
        self.rewards = []

        self._load_rewards_from_csv()
        self._load_lines_from_csv()
        

    def _load_lines_from_csv(self):
        """Loads race track lines from a CSV file."""
        filename = os.path.join("assets/track", "drawn_racetrack-06.05.23-01.10.csv")
        with open(filename, "r", newline="") as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                temp_line = [tuple(map(int, point.strip('()').split(', '))) for point in row]
                self.lines.append(temp_line)


    def _load_rewards_from_csv(self):
        """Loads rewards from a CSV file."""
        filename = os.path.join("assets/track", "drawn_reward-06.05.23-01.32.csv")
        with open(filename, "r", newline="") as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                self.rewards.append((int(row[0]), int(row[1])))


    def draw(self, screen, display_hitboxes):
        """Draw racetrack lines on screen"""
        if display_hitboxes:
            # Racetrack lines
            for line in self.lines:
                pygame.draw.line(screen, (255, 0, 0), line[0], line[1], 3)


class RewardCoin:
    """Class representing the active reward coin on the track"""

    def __init__(self, center, sprite):
        self.sprite = sprite
        self.center = center
        self.radius = max(sprite.get_rect().width, sprite.get_rect().height)/2
        self.sprite_rect = self.sprite.get_rect(center=center)
        self.lines = sprite_to_lines(self.sprite_rect, self.sprite_rect.height, self.sprite_rect.width, 0)
        self.top_left = Vector2(center[0] - self.sprite_rect.height // 2, center[1] - self.sprite_rect.width // 2)


    def get_radius(self):
        """Simple getter for getting the reward coin's radius"""
        return self.radius


    def draw(self, screen, display_hitboxes):
        """Draw coin on screen"""
        screen.blit(self.sprite, self.top_left)
        if display_hitboxes:
            pygame.draw.circle(screen, (0, 255, 255), (self.center), self.radius, 3)


    def intersect_with_reward(self, distance_to_center):
        """Check if a given point intersects with the sprite rectangle"""
        return distance_to_center <= self.radius


class GameBackground:
    """Class containing all assets of the game background"""
    def __init__(self, racetrack_background):
        self.racetrack_background = racetrack_background
        self.font = pygame.font.Font(None, 20)              # Font for displaying text
        self._create_scoreboard_bg() 
    

    def _create_scoreboard_bg(self):
        """Create semi-transparent background for scoreboard"""
        self.scoreboard_bg = pygame.Surface((200, 60))
        self.scoreboard_bg.fill((0, 0, 0))
        self.scoreboard_bg.set_alpha(150)


    def draw_scoreboard(self, screen, time, score, frame_rate, attempt, wins):
        """Render background and scoreboard"""
        screen.blit(self.racetrack_background, (0,0))
        
        temp_scoreboard = self.scoreboard_bg.copy()
        score_time_text = self.font.render(f"Time: {time}s        Score: {score:.0f}", True, (255, 255, 255))
        attempt_wins_text = self.font.render(f"Attempt: {attempt}     Wins: {wins}", True, (255, 255, 255))
        FPS_attempt_text = self.font.render(f"FPS: {frame_rate}     Attempt: {attempt}", True, (255, 255, 255))
        
        temp_scoreboard.blit(score_time_text, (10, 10))
        temp_scoreboard.blit(FPS_attempt_text, (10, 35))
        screen.blit(temp_scoreboard, (screen.get_width() - 210, 10))


    def draw_key_status(self, screen, keypress, ai_running=False):
        """Draw the keypad, lighting up when each associated key is pressed"""
        # Define colors
        GRAY = (128, 128, 128)
        RED = (255, 0, 0)
        BLACK = (0, 0, 0)
        width = screen.get_width()
        height = screen.get_height()
        offset = 10                         # Offset from bottom and right side of the walls
        converted_keypress = [0] * 4   # Need four binaries to convey which keys should be lit up red (because of being pressed)

        # Define key positions and sizes
        key_size = (50, 50)
        key_positions = {
            'Up': (width - key_size[0] * 2 - offset * 2, height - key_size[1] * 2 - offset * 2),  # Up is on top
            'Down': (width - key_size[0] * 2 - offset * 2, height - key_size[1] - offset),  # Down is below Up
            'Left': (width - key_size[0] * 3 - offset * 3, height - key_size[1] - offset),  # Left is to the left of Down
            'Right': (width - key_size[0] * 1 - offset, height - key_size[1] - offset),  # Right is to the right of Down
        }

        # Define arrow shapes, centered within the keys
        arrows = {
            'Up': [(25, 15), (15, 35), (35, 35)],  # Arrow pointing up
            'Down': [(25, 35), (15, 15), (35, 15)],  # Arrow pointing down
            'Left': [(15, 25), (35, 15), (35, 35)],  # Arrow pointing left
            'Right': [(35, 25), (15, 15), (15, 35)]  # Arrow pointing right
        }

        # Define background rectangle
        bg_rect = pygame.Rect(width - key_size[0] * 3 - offset*4, height - key_size[1] * 2 - offset*3, key_size[0] * 3 + offset*4, key_size[1] * 2 + offset*3)

        # Draw semi-transparent background
        s = pygame.Surface((bg_rect.width, bg_rect.height))  # The size of your rect
        s.set_alpha(150)  # Alpha level
        s.fill(BLACK)  # This fills the entire surface
        screen.blit(s, (bg_rect.x, bg_rect.y))  # (0,0) are the top-left coordinates

        # Create temporary binary array for keys to pass through which should be highlighted red vs. grey
        if ai_running:
            for i, _ in enumerate(['Left', 'Up', 'Right', 'Down']):
                if keypress[i] == 1:
                    converted_keypress[i] = 1
        
        # Have different loop for if human is playing since more keys
        else:
            for i, _ in enumerate(['Left', 'Up', 'Right', 'Down', 'Do_Nothing', 'Up_Left', 'Up_Right', 'Down_Left', 'Down_Right']):
                if keypress[i] == 1 and i < 4:
                    converted_keypress[i] = 1
                elif keypress[i] == 1 and i == 5:           # Up_Left
                    converted_keypress = [1,1,0,0]
                elif keypress[i] == 1 and i == 6:           # Up_Right
                    converted_keypress = [0,1,1,0]
                elif keypress[i] == 1 and i == 7:           # Down_Left
                    converted_keypress = [1,0,0,1]
                elif keypress[i] == 1 and i == 8:           # Down_Right
                    converted_keypress = [0,0,1,1]


        # Draw keys and arrows
        for i, key in enumerate(['Left', 'Up', 'Right', 'Down']):
            color = RED if converted_keypress[i] else GRAY  # i+1 because keypress[0] is for 'None'
            
            # Draw key with black border
            pygame.draw.rect(screen, BLACK, (*key_positions[key], *key_size), 2)  # Border
            pygame.draw.rect(screen, color, (key_positions[key][0] + 2, key_positions[key][1] + 2, key_size[0] - 4, key_size[1] - 4))  # Key

            # Draw arrows on keys
            arrow = [(pos[0] + key_positions[key][0], pos[1] + key_positions[key][1]) for pos in arrows[key]]
            pygame.draw.polygon(screen, BLACK, arrow)