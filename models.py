import pygame
import csv
import os
import math
from pygame.math import Vector2
from utils import sprite_to_lines, ccw

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


    def draw(self, surface):
        # Draw the lines that make up the racecar
        for i in range(len(self.linesegments)):
            pygame.draw.line(surface, (0, 0, 255), self.linesegments[i][0], self.linesegments[i][1], 4)

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



#Create class that represents the racetrack, which is a list of lines that are the borders
class Racetrack:
    def __init__(self):
        self.lines = []         # Pull in tuples from CSV file
        self._load_lines_from_csv()

    def _load_lines_from_csv(self):
        filename = os.path.join("assets", "racetrack_lines.csv")
        with open(filename, "r", newline="") as csvfile:
            csv_reader = csv.reader(csvfile)
            
            # Iterate through rows in CSV
            for row in csv_reader:
                temp_line = []
                temp_max_x, temp_max_y = 0, 0
                
                # Iterate through points in row. We want to store the lines (as tuple in loc 0) and the max x and y to enable quick parsing for calculation
                for point in row:
                    temp_tuple = tuple(map(int, point.strip('()').split(', ')))

                    # Calculate the max x and y and store for each line. This helps us optimize calculations later on    
                    if temp_max_x < temp_tuple[0]: temp_max_x = temp_tuple[0]
                    if temp_max_y < temp_tuple[1]: temp_max_y = temp_tuple[1]
                    temp_line.append(temp_tuple)

                temp_line.append(tuple([temp_max_x, temp_max_y]))
                self.lines.append(temp_line)
        
        # Returns a list with 3 tuples in each element: Point1 (X,Y), Point2 (X,Y), Max Indices (Max X, Max Y) 
        return self.lines