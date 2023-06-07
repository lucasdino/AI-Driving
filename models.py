import pygame
import csv
import os
from math import sin, cos, radians
from pygame import Vector2
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
        self.visionlines = []
        self.visiondist = []

    def _update_rotated_sprite(self):
        self.rotated_sprite = pygame.transform.rotate(self.sprite, self.angle)
        self.rotated_sprite_rect = self.rotated_sprite.get_rect(center=self.position)

    def draw(self, surface):
        """Draws the racecar lines and vision lines on the provided surface."""
        for line in self.linesegments:
            pygame.draw.line(surface, (0, 0, 255), line[0], line[1], 4)
        for line in self.visionlines:
            pygame.draw.line(surface, (100, 100, 0), line[0], line[1], 2)
        surface.blit(self.rotated_sprite, self.rotated_sprite_rect)

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

    def check_line_collision(self, racetrack_line):
        """
        Checks for a collision between the racecar and a given racetrack line.
        Returns True if a collision is detected, False otherwise.
        """
        for line in self.linesegments:
            A, B = racetrack_line[0], racetrack_line[1]
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
        self.visionlines = []
        self.visiondist = []

        for _ in range(num_steps):
            temp_angle += angle_step
            angle_rad = radians(temp_angle)
            self.visiondist.append(min(nearest_line_distance(center, angle_rad, line) for line in racetrack_line))
            temp_x = center[0] + self.visiondist[-1] * sin(angle_rad)
            temp_y = center[1] + self.visiondist[-1] * cos(angle_rad)
            self.visionlines.append((center, (temp_x, temp_y)))



class Racetrack:
    """Class representing the racetrack, which consists of a list of border lines and rewards."""
    
    def __init__(self):
        self.lines = []
        self.rewards = []
        self._load_lines_from_csv()
        self._load_rewards_from_csv()

    def _load_lines_from_csv(self):
        """Loads race track lines from a CSV file."""
        filename = os.path.join("assets/track", "racetrack_lines.csv")
        with open(filename, "r", newline="") as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                temp_line = [tuple(map(int, point.strip('()').split(', '))) for point in row]
                self.lines.append(temp_line)

    def _load_rewards_from_csv(self):
        """Loads rewards from a CSV file."""
        filename = os.path.join("assets/track", "Reward_drawing-06.05.23-01.32.csv")
        with open(filename, "r", newline="") as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                self.rewards.append((int(row[0]), int(row[1])))