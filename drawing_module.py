import pygame
import csv

class Drawing:
    def __init__(self):
        self.drawing_flag = False
        self.start_point = None
        self.end_point = None
        self.lines_counter = 0
        self.lines = []
        self.rewards = []

    def draw_rt_lines(self, screen):
        for line in self.lines:
            pygame.draw.line(screen, (255, 165, 0), line[0], line[1], 2)

    def draw_rewards(self, coinsprite, screen):
        for reward in self.rewards:
            x,y = reward
            coin_width, coin_height = coinsprite.get_size()
            screen.blit(coinsprite, (x - coin_width // 2, y - coin_height // 2))
    
    def handle_rt_drawing_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.drawing_flag = True
            self.start_point = event.pos
            self.lines.append((self.start_point, self.start_point))
        elif event.type == pygame.MOUSEBUTTONUP:
            self.drawing_flag = False
            self.end_point = event.pos
            self.lines[self.lines_counter] = self.start_point, self.end_point
            self.lines_counter += 1
        elif event.type == pygame.MOUSEMOTION and self.drawing_flag:
            self.end_point = event.pos
            self.lines[self.lines_counter] = self.start_point, self.end_point
            
    
    def handle_reward_drawing_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.rewards.append(event.pos)

    def save_drawing_to_csv(self, filename):
        with open(f'assets/track/{filename}.csv', "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            for line in self.lines:
                csv_writer.writerow(line)
            for reward in self.rewards:
                csv_writer.writerow(reward)
