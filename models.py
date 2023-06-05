import pygame
import csv
import os
import math
from pygame.math import Vector2
from utils import sprite_to_lines, ccw, distance_point_line

#Create class that represents the racecar, which is sprite and box of lines
class Racecar:
    def __init__(self, position, sprite, velocity):
        self.position = Vector2(position)
        self.velocity = Vector2(velocity)
        self.angle = 90
        self.sprite = sprite
        self.rotated_sprite = pygame.transform.rotate(self.sprite, self.angle)
        self.rotated_sprite_rect = self.rotated_sprite.get_rect(center = self.position)

        # Set up vars that we'll use later to calculate the hit box for the car
        self.sprite_w = self.rotated_sprite_rect.width
        self.sprite_l = self.rotated_sprite_rect.height
        self.linesegments = []
        self.visionlines = []
        self.visiondist = []


    def draw(self, surface):
        # Draw the lines that make up the racecar
        for i in range(len(self.linesegments)):
            pygame.draw.line(surface, (0, 0, 255), self.linesegments[i][0], self.linesegments[i][1], 4)

        # Draw the lines that make up the vision
        for i in range(len(self.visionlines)):
            pygame.draw.line(surface, (100, 100, 0), self.visionlines[i][0], self.visionlines[i][1], 2)

        surface.blit(self.rotated_sprite, self.rotated_sprite_rect)

    # ----------------------------------
    # Driving mechanics
    # ----------------------------------

    def move(self):
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]
        
        # Recalculate the lines that make up the racecar hitbox (for collision detection and drawing)
        self.rotated_sprite = pygame.transform.rotate(self.sprite, self.angle)
        self.rotated_sprite_rect = self.rotated_sprite.get_rect(center = self.position)
        self.linesegments = sprite_to_lines(self.rotated_sprite_rect, self.sprite_w, self.sprite_l, self.angle)

    def accelerate(self, acceleration):
        dx = acceleration * math.cos(math.radians(self.angle))
        dy = acceleration * math.sin(math.radians(self.angle))
        self.velocity = (self.velocity[0] + dx, self.velocity[1] - dy)

    def brake(self, deceleration):
        dx = deceleration * math.cos(math.radians(self.angle))
        dy = deceleration * math.sin(math.radians(self.angle))
        self.velocity = (self.velocity[0] - dx, self.velocity[1] + dy)

    def turn_left(self, turn_speed):
        self.angle += turn_speed

    def turn_right(self, turn_speed):
        self.angle -= turn_speed

    def apply_drag(self, drag):
        self.velocity = (self.velocity[0] * drag, self.velocity[1] * drag)

    # ----------------------------------
    # Collision mechanics
    # ----------------------------------

    # Since we'll be representing all objects as lines, have method that can compare if two lines are intersecting
    def check_line_collision(self, racetrack_line):
        # Calculate if intersecting with racetrack line
        for line in self.linesegments:
            A, B = racetrack_line[0], racetrack_line[1]
            C, D = line
            if(ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)): return True

        return False

    # ----------------------------------
    # Determine length to nearest racetrack line for vision lines
    # ----------------------------------

    def vision_line_distance(self, racetrack_line):

        angle_step = 45     # Step change in degrees for each iteration
        temp_angle = self.angle-angle_step
        num_steps = int(360/angle_step)
        center = self.rotated_sprite_rect.center
        self.visionlines = []
        self.visiondist = []

        # Need to iterate through lines and determine how far each is from the nearest racetrack line
        for i in range(num_steps):
            temp_angle += angle_step

            # Calculate the distance to the closest intersection line
            self.visiondist.append(min(distance_point_line(center, math.radians(temp_angle), line) for line in racetrack_line))

            # Draw the line
            temp_x = center[0] + self.visiondist[i]*(math.sin(math.radians(temp_angle)))
            temp_y = center[1] + self.visiondist[i]*(math.cos(math.radians(temp_angle)))
            
            self.visionlines.append((center, (temp_x, temp_y)))


#Create class that represents the racetrack, which is a list of lines that are the borders
class Racetrack:
    def __init__(self):
        self.lines = []         # Pull in tuples from CSV file
        self._load_lines_from_csv()
        self.rewards = []       # Pull in points from CSV file
        self._load_rewards_from_csv()

    def _load_lines_from_csv(self):
        filename = os.path.join("assets/track", "racetrack_lines.csv")
        with open(filename, "r", newline="") as csvfile:
            csv_reader = csv.reader(csvfile)
            
            # Iterate through rows in CSV
            for row in csv_reader:
                temp_line = []
                
                # Iterate through points in row. We want to store the lines (as tuple in loc 0)
                for point in row:
                    temp_line.append(tuple(map(int, point.strip('()').split(', '))))

                self.lines.append(temp_line)

    def _load_rewards_from_csv(self):
        filename = os.path.join("assets/track", "Reward_drawing-06.05.23-01.32.csv")
        with open(filename, "r", newline="") as csvfile:
            csv_reader = csv.reader(csvfile)
            
            # Iterate through rows in CSV
            for row in csv_reader:
                self.rewards.append((int(row[0]), int(row[1])))
            