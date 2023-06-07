import pygame
import csv

class Drawing:
    """The Drawing class provides modules to draw a racetrack map and set rewards with the mouse."""
    
    def __init__(self):
        """Initializes Drawing class and sets up helper variables."""
        self._drawing_flag = False
        self._start_point = None
        self._end_point = None
        self._lines_counter = 0
        self._lines = []
        self._rewards = []

    def draw_rt_lines(self, screen):
        """Draws the racetrack lines on the screen."""
        for line in self._lines:
            pygame.draw.line(screen, (255, 165, 0), line[0], line[1], 2)

    def draw_rewards(self, rewardcoin, screen):
        """Draws the rewards on the screen."""
        for reward in self._rewards:
            x,y = reward
            coin_width = rewardcoin.sprite_rect.width
            coin_height = rewardcoin.sprite_rect.height
            screen.blit(rewardcoin.sprite, (x - coin_width // 2, y - coin_height // 2))
    
    def handle_rt_drawing_events(self, event):
        """Handles racetrack drawing events based on the pygame event given."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            self._drawing_flag = True
            self._start_point = event.pos
            self._lines.append((self._start_point, self._start_point))
        elif event.type == pygame.MOUSEBUTTONUP:
            self._drawing_flag = False
            self._end_point = event.pos
            self._lines[self._lines_counter] = self._start_point, self._end_point
            self._lines_counter += 1
        elif event.type == pygame.MOUSEMOTION and self._drawing_flag:
            self._end_point = event.pos
            self._lines[self._lines_counter] = self._start_point, self._end_point
            
    def handle_reward_drawing_events(self, event):
        """Handles reward drawing events based on the pygame event given."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            self._rewards.append(event.pos)

    def save_drawing_to_csv(self, filename):
        """Saves the lines and rewards to a CSV file."""
        with open(f'assets/track/{filename}.csv', "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            for line in self._lines:
                csv_writer.writerow(line)
            for reward in self._rewards:
                csv_writer.writerow(reward)