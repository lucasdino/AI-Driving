import pygame
import csv
import os
from math import sin, cos, radians, atan2, hypot
from pygame import Vector2, font, Surface
from utils import sprite_to_lines, is_counterclockwise, nearest_line_distance


class Racecar:
    """Class representing the racecar, which includes both the sprite and hit box of lines."""
    def __init__(self, position, sprite, velocity):
        self.position = Vector2(position)
        self.velocity = Vector2(velocity)
        self.angle = 90
        self.sprite = sprite
        self._update_rotated_sprite()
        self._sprite_w = self.rotated_sprite_rect.width
        self._sprite_l = self.rotated_sprite_rect.height
        self.linesegments = []
        self.vision_lines = []
        
        self.modelinputs = {
            "vision_distances": [],
            "angle": radians(self.angle),
            "angle_to_reward": 0,
            "distance_to_reward": 0
        }


    def _update_rotated_sprite(self):
        self.rotated_sprite = pygame.transform.rotate(self.sprite, self.angle)
        self.rotated_sprite_rect = self.rotated_sprite.get_rect(center=self.position)


    def draw(self, screen):
        """Draws the racecar lines and vision lines on the provided screen."""
        # Racecar hitbox - uncomment for troubleshooting
        # for line in self.linesegments:
        #     pygame.draw.line(screen, (0, 0, 255), line[0], line[1], 4)
        
        # Racecar vision lines and reward line - uncomment for troubleshooting
        # for line in self.vision_lines:
        #     pygame.draw.line(screen, (200, 100, 0), line[0], line[1], 2)
        # self._draw_reward_line(screen)
        
        screen.blit(self.rotated_sprite, self.rotated_sprite_rect)

    def _draw_reward_line(self, screen):
        """Draw line going to next coin to ensure calculation is correct"""
        end_point = (
            self.position[0] - self.modelinputs['distance_to_reward'] * cos(self.modelinputs['angle_to_reward']),
            self.position[1] - self.modelinputs['distance_to_reward'] * sin(self.modelinputs['angle_to_reward']),
        )   
        pygame.draw.line(screen, (200, 100, 0), self.position, end_point, 2)


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
        self.modelinputs['angle'] = radians(self.angle)


    def turn_right(self, turn_speed):
        """Turns the racecar to the right based on the provided turn_speed."""
        self.angle -= turn_speed
        self.modelinputs['angle'] = radians(self.angle)


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


    def vision_line_distance(self, racetrack_line):
        """
        Calculates the distance to the nearest racetrack line for vision lines.
        Updates self.visiondist and self.visionlines.
        """
        angle_step = 45
        temp_angle = self.angle - angle_step
        num_steps = int(360 / angle_step)
        center = self.rotated_sprite_rect.center
        self.vision_lines = []
        self.modelinputs['vision_distances'] = []

        for _ in range(num_steps):
            temp_angle += angle_step
            angle_rad = radians(temp_angle)
            self.modelinputs['vision_distances'].append(min(nearest_line_distance(center, angle_rad, line) for line in racetrack_line))
            temp_x = center[0] + self.modelinputs['vision_distances'][-1] * sin(angle_rad)
            temp_y = center[1] + self.modelinputs['vision_distances'][-1] * cos(angle_rad)
            self.vision_lines.append((center, (temp_x, temp_y)))
    

    def calculate_reward_line(self, rewardcoin):
        """Calculates the distance and angle to the next reward coin and updates 'modelinputs'"""
        rewardcoin_x, rewardcoin_y = rewardcoin.center
        racecar_x, racecar_y = self.position
        
        dx = racecar_x - rewardcoin_x
        dy = racecar_y - rewardcoin_y
        
        # Calculate angle in radians between the two points
        self.modelinputs['angle_to_reward'] = atan2(dy, dx)
        
        # Calculate the distance between the two points
        self.modelinputs['distance_to_reward'] = hypot(dx, dy)



class Racetrack:
    """Class representing the racetrack, which consists of a list of border lines and rewards."""
    
    def __init__(self):
        self.lines = []
        self.rewards = []
        self._load_lines_from_csv()
        self._load_rewards_from_csv()


    def _load_lines_from_csv(self):
        """Loads race track lines from a CSV file."""
        filename = os.path.join("assets/track", "drawn_racetrack.csv")
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


    def draw(self, screen):
        """Draw racetrack lines on screen"""
        # Racetrack lines - uncomment for troubleshooting
        # for line in self.lines:
        #     pygame.draw.line(screen, (255, 0, 0), line[0], line[1], 3)



class RewardCoin:
    """Class representing the active reward coin on the track"""

    def __init__(self, center, sprite):
        self.sprite = sprite
        self.center = center
        self.sprite_rect = self.sprite.get_rect(center=center)
        self.lines = sprite_to_lines(self.sprite_rect, self.sprite_rect.height, self.sprite_rect.width, 0)
        self.top_left = Vector2(center[0] - self.sprite_rect.height // 2, center[1] - self.sprite_rect.width // 2)


    def draw(self, screen):
        """Draw coin on screen"""
        # Coin hit box - uncomment for troubleshooting
        # for line in self.lines:
        #     pygame.draw.line(screen, (255, 0, 0), line[0], line[1], 3)
        
        screen.blit(self.sprite, self.top_left)



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


    def draw(self, screen, time, score, frame_rate, attempt):
        """Render background and scoreboard"""
        screen.blit(self.racetrack_background, (0,0))
        
        temp_scoreboard = self.scoreboard_bg.copy()
        score_time_text = self.font.render(f"Time: {time}s     Score: {score}", True, (255, 255, 255))
        fps_attempt_text = self.font.render(f"FPS: {frame_rate}     Attempt: {attempt}", True, (255, 255, 255))
        
        temp_scoreboard.blit(score_time_text, (10, 10))
        temp_scoreboard.blit(fps_attempt_text, (10, 35))
        screen.blit(temp_scoreboard, (screen.get_width() - 210, 10))
