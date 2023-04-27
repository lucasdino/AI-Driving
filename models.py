import pygame
import csv
import os
from pygame.math import Vector2
from utils import sprite_to_lines

class Racecar:
    def __init__(self, position, sprite, velocity):
        self.position = Vector2(position)
        self.velocity = Vector2(velocity)
        self.sprite = sprite
        self.linesegments = sprite_to_lines(pygame.Rect(position[0], position[1], sprite.get_width(), sprite.get_height()))


    def draw(self, surface):
        # Draw the racecar sprite
        blit_position = self.position
        
        # Draw the lines that make up the racecar
        for i in range(len(self.linesegments)):
            pygame.draw.line(surface, (0, 0, 255), self.linesegments[i][0], self.linesegments[i][1], 4)

        surface.blit(self.sprite, blit_position)


    def move(self):
        self.position = self.position + self.velocity


    # Since we'll be representing all objects as lines, have method that can compare if two lines are intersecting
    def check_line_collision(self, line1, line2):
        p1, p2 = line1
        p3, p4 = line2
        denom = ((p4[1] - p3[1]) * (p2[0] - p1[0])) - ((p4[0] - p3[0]) * (p2[1] - p1[1]))

        if denom == 0:
            return False

        ua = (((p4[0] - p3[0]) * (p1[1] - p3[1])) - ((p4[1] - p3[1]) * (p1[0] - p3[0]))) / denom
        ub = (((p2[0] - p1[0]) * (p1[1] - p3[1])) - ((p2[1] - p1[1]) * (p1[0] - p3[0]))) / denom

        if 0 <= ua <= 1 and 0 <= ub <= 1:
            return True

        return False

    def collides_with(self, other_line):
        pass


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