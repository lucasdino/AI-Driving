import pygame
import csv
import os
import math
from pygame.math import Vector2
from utils import sprite_to_lines

#Create class that represents the racecar, which is sprite and box of lines
class Racecar:
    def __init__(self, position, sprite, velocity):
        self.position = Vector2(position)
        self.velocity = Vector2(velocity)
        self.angle = 90
        self.sprite = sprite
        self.linesegments = sprite_to_lines(sprite.get_rect(), self.angle)


    def draw(self, surface):
        # Draw the racecar sprite
        rotated_sprite = pygame.transform.rotate(self.sprite, self.angle)
        rotated_sprite_rect = rotated_sprite.get_rect(center = self.position)
        self.linesegments = sprite_to_lines(rotated_sprite_rect, self.angle)
        
        # Draw the lines that make up the racecar
        for i in range(len(self.linesegments)):
            pygame.draw.line(surface, (0, 0, 255), self.linesegments[i][0], self.linesegments[i][1], 4)

        surface.blit(rotated_sprite, rotated_sprite_rect)

    # ----------------------------------
    # Driving mechanics
    # ----------------------------------

    def move(self):
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

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
    def check_line_collision(self, line1, line2):
        pass

    def collides_with(self, other_line):
        pass


#Create class that represents the racetrack, which is a list of lines that are the borders
class Racetrack:
    def __init__(self):
        self.lines = []         # Pull in tuples from CSV file
        self._load_lines_from_csv()

    def _load_lines_from_csv(self):
        filename = os.path.join("assets", "racetrack_lines.csv")
        with open(filename, "r", newline="") as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                line = [tuple(map(int, point.strip('()').split(', '))) for point in row]
                self.lines.append(line)
        return self.lines