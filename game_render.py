import pygame
import numpy as np
from utility.utils import *


class GameBackground:

    # Define colors
    palette = {
        "BLACK": (0, 0, 0),
        "MEDIUM_GRAY": (128, 128, 128),
        "WHITE": (255, 255, 255),
        "RED": (255, 0, 0),
        "LIGHT_GREEN": (144, 238, 144),
        "LIGHT_RED": (255, 182, 193)
    }
    
    # Define reused objects / etc.
    model_toggle_buttons = {
        "toggle_render_button": pygame.Rect(10, 10, 80, 20),
        "export_weights_button": pygame.Rect(10, 35, 80, 20)
    }

    # Define a few font classes
    pygame.font.init()
    fonts = {
        "scoreboard_font": pygame.font.Font(None, 20),
        "model_toggle_font": pygame.font.Font(None, 14)
    }
    fonts["model_toggle_font"].set_bold(True)       # Let's make this font bold

    """Class containing all assets of the game background"""
    def __init__(self, racetrack_background, screen):
        self.width, self.height = screen.get_width(), screen.get_height()
        self.racetrack_background = racetrack_background
        self._init_backgrounds()
    

    def _init_backgrounds(self):
        self._init_scoreboard_bg()
        self._init_key_status_bg()


    def _init_scoreboard_bg(self):
        """Create semi-transparent background for scoreboard"""
        self.scoreboard_bg = pygame.Surface((200, 60))
        self.scoreboard_bg.fill((0, 0, 0))
        self.scoreboard_bg.set_alpha(150)


    def _init_key_status_bg(self):
        offset = 10               # Offset from bottom and right side of the walls
        self.key_size = (50, 50)       # Define key positions and sizes
        
        self.key_positions = {
            'Up': (self.width - self.key_size[0] * 2 - offset * 2, self.height - self.key_size[1] * 2 - offset * 2),  # Up is on top
            'Down': (self.width - self.key_size[0] * 2 - offset * 2, self.height - self.key_size[1] - offset),  # Down is below Up
            'Left': (self.width - self.key_size[0] * 3 - offset * 3, self.height - self.key_size[1] - offset),  # Left is to the left of Down
            'Right': (self.width - self.key_size[0] * 1 - offset, self.height - self.key_size[1] - offset),  # Right is to the right of Down
        }

        # Define arrow shapes, centered within the keys
        self.arrows = {
            'Up': [(25, 15), (15, 35), (35, 35)],  # Arrow pointing up
            'Down': [(25, 35), (15, 15), (35, 15)],  # Arrow pointing down
            'Left': [(15, 25), (35, 15), (35, 35)],  # Arrow pointing left
            'Right': [(35, 25), (15, 15), (15, 35)]  # Arrow pointing right
        }

        # Define background rectangle
        self.keystatus_bg_rect = pygame.Rect(self.width - self.key_size[0] * 3 - offset*4, self.height - self.key_size[1] * 2 - offset*3, self.key_size[0] * 3 + offset*4, self.key_size[1] * 2 + offset*3)


    def get_model_toggle_buttons(self):
        """Getter called by neural net class to return the necessary model toggle buttons"""
        return self.model_toggle_buttons


    def draw_scoreboard(self, screen, time, score, frame_rate, session_metadata):
        """Render background and scoreboard"""
        screen.blit(self.racetrack_background, (0,0))
        
        temp_scoreboard = self.scoreboard_bg.copy()
        score_time_text = self.fonts["scoreboard_font"].render(f"Time: {time}s        Score: {score:.0f}", True, self.palette["WHITE"])
        attempt_wins_text = self.fonts["scoreboard_font"].render(f"Attempt: {session_metadata['attempts']}     Wins: {session_metadata['wins']}", True, self.palette["WHITE"])
        FPS_attempt_text = self.fonts["scoreboard_font"].render(f"FPS: {frame_rate}     Attempt: {session_metadata['attempts']}", True, self.palette["WHITE"])
        temp_scoreboard.blit(score_time_text, (10, 10))
        temp_scoreboard.blit(FPS_attempt_text, (10, 35))
        
        screen.blit(temp_scoreboard, (self.width - 210, 10))


    def draw_key_status(self, screen, keypress, ai_running=False):
        """Draw the keypad, lighting up when each associated key is pressed"""
        converted_keypress = [0] * 4   # Need four binaries to convey which keys should be lit up red (because of being pressed)

        # Draw semi-transparent background
        s = pygame.Surface((self.keystatus_bg_rect.width, self.keystatus_bg_rect.height))  # The size of your rect
        s.set_alpha(150)  # Alpha level
        s.fill(self.palette["BLACK"])  # This fills the entire surface
        screen.blit(s, (self.keystatus_bg_rect.x, self.keystatus_bg_rect.y))  # (0,0) are the top-left coordinates

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
            color = self.palette["RED"] if converted_keypress[i] else self.palette["MEDIUM_GRAY"]  # i+1 because keypress[0] is for 'None'
            
            # Draw key with black border
            pygame.draw.rect(screen, self.palette["BLACK"], (*self.key_positions[key], *self.key_size), 2)  # Border
            pygame.draw.rect(screen, color, (self.key_positions[key][0] + 2, self.key_positions[key][1] + 2, self.key_size[0] - 4, self.key_size[1] - 4))  # Key

            # Draw arrows on keys
            arrow = [(pos[0] + self.key_positions[key][0], pos[1] + self.key_positions[key][1]) for pos in self.arrows[key]]
            pygame.draw.polygon(screen, self.palette["BLACK"], arrow)


    def draw_model_toggles(self, screen, model_toggles):
        """Draws the model toggles that you can clicck in AI Traiing to turn on / off rendering and speed up FPS"""
        
        if model_toggles["Render"]: toggle_color = self.palette["LIGHT_GREEN"]
        else: toggle_color = self.palette['LIGHT_RED']

        
        render_text = "Render: ON" if model_toggles["Render"] else "Render: OFF"
        render_text_font = self.fonts["model_toggle_font"].render(render_text, True, self.palette["BLACK"])
        export_weights_text = self.fonts["model_toggle_font"].render("Export Weights", True, self.palette["BLACK"])

        # Drawing the shadows
        pygame.draw.rect(screen, self.palette["MEDIUM_GRAY"], (self.model_toggle_buttons["toggle_render_button"].x+2, self.model_toggle_buttons["toggle_render_button"].y+2, self.model_toggle_buttons["toggle_render_button"].width, self.model_toggle_buttons["toggle_render_button"].height))
        pygame.draw.rect(screen, self.palette["MEDIUM_GRAY"], (self.model_toggle_buttons["export_weights_button"].x+2, self.model_toggle_buttons["export_weights_button"].y+2, self.model_toggle_buttons["export_weights_button"].width, self.model_toggle_buttons["export_weights_button"].height))
        
        # Drawing the buttons
        pygame.draw.rect(screen, toggle_color, self.model_toggle_buttons["toggle_render_button"])
        pygame.draw.rect(screen, self.palette["MEDIUM_GRAY"], self.model_toggle_buttons["export_weights_button"])

        # Drawing the borders
        pygame.draw.rect(screen, self.palette["BLACK"], self.model_toggle_buttons["toggle_render_button"], 2)
        pygame.draw.rect(screen, self.palette["BLACK"], self.model_toggle_buttons["export_weights_button"], 2)
        
        screen.blit(render_text_font, (13, 15))
        screen.blit(export_weights_text, (13, 40))

    
    def render_off(self, screen, model_toggles):
        """Class to call if game is rendered off during NN training"""
        screen.fill(self.palette["BLACK"])
        self.draw_model_toggles(screen, model_toggles)
        pygame.display.flip()