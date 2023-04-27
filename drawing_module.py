import pygame
import csv
import os

class Drawing:
    def __init__(self):
        self.drawing_flag = False
        self.start_point = None
        self.end_point = None
        self.lines = []

    def draw_straight_lines(self, screen):
        for line in self.lines:
            pygame.draw.line(screen, (255, 0, 0), line[0], line[1], 3)

    def handle_drawing_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.drawing_flag = True
            self.start_point = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            self.drawing_flag = False
            self.end_point = event.pos
            self.lines.append((self.start_point, self.end_point))
        elif event.type == pygame.MOUSEMOTION and self.drawing_flag:
            self.end_point = event.pos

    def save_lines_to_csv(self, filename):
        with open(filename, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            for line in self.lines:
                csv_writer.writerow(line)
