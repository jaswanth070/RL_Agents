import pygame
import numpy as np
import random

# --- Constants ---
SCREEN_WIDTH, SCREEN_HEIGHT = 600, 400
DRONE_SIZE = 20
OBSTACLE_WIDTH = 60
GAP_HEIGHT = 100   # narrower gap for more frequent action
SCROLL_SPEED = 4   # slightly faster to keep it engaging
EPISODE_MAX_STEPS = 600  # ~10 sec at 60 FPS

BG_COLOR = (25, 25, 35)
DRONE_COLOR = (0, 200, 200)
OBSTACLE_COLOR = (200, 60, 60)
TEXT_COLOR = (240, 240, 240)

class Drone:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.x = 100
        self.y = SCREEN_HEIGHT // 2
    
    def update(self, action):
        if action == 0:   # Up
            self.y -= 5
        elif action == 2: # Down
            self.y += 5
        # Clamp inside screen
        self.y = np.clip(self.y, 0, SCREEN_HEIGHT-DRONE_SIZE)
    
    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, DRONE_SIZE, DRONE_SIZE)

class Environment:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Drone Imitation Learning")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24)
        self.reset()
    
    def reset(self):
        self.drone = Drone()
        self.obstacles = []
        self.steps = 0
        self._spawn_obstacle()
        return self._get_state()
    
    def _spawn_obstacle(self):
        gap_y = random.randint(50, SCREEN_HEIGHT - GAP_HEIGHT - 50)
        self.obstacles.append({
            'x': SCREEN_WIDTH,
            'gap_y': gap_y
        })
    
    def _get_state(self):
        obs = self.obstacles[0]
        dist_x = (obs['x'] - self.drone.x) / SCREEN_WIDTH
        gap_center = (obs['gap_y'] + GAP_HEIGHT//2) / SCREEN_HEIGHT
        drone_y_norm = self.drone.y / SCREEN_HEIGHT
        return np.array([dist_x, gap_center, drone_y_norm])
    
    def step(self, action):
        self.drone.update(action)
        for obs in self.obstacles:
            obs['x'] -= SCROLL_SPEED
        if self.obstacles and self.obstacles[0]['x'] < -OBSTACLE_WIDTH:
            self.obstacles.pop(0)
        if not self.obstacles or self.obstacles[-1]['x'] < SCREEN_WIDTH - 200:
            self._spawn_obstacle()
        
        done = False
        self.steps += 1
        # Collision check
        drone_rect = self.drone.rect
        for obs in self.obstacles:
            top_rect = pygame.Rect(obs['x'], 0, OBSTACLE_WIDTH, obs['gap_y'])
            bottom_rect = pygame.Rect(obs['x'], obs['gap_y']+GAP_HEIGHT,
                                      OBSTACLE_WIDTH, SCREEN_HEIGHT)
            if drone_rect.colliderect(top_rect) or drone_rect.colliderect(bottom_rect):
                done = True
        
        # Auto-end after max steps
        if self.steps >= EPISODE_MAX_STEPS:
            done = True
        
        return self._get_state(), done
    
    def get_human_action(self):
        action = 1 # Stay default
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action = 0
        elif keys[pygame.K_DOWN]: action = 2
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return -1
        return action
    
    def render(self, text_lines):
        self.screen.fill(BG_COLOR)
        pygame.draw.rect(self.screen, DRONE_COLOR, self.drone.rect)
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, OBSTACLE_COLOR, (obs['x'], 0, OBSTACLE_WIDTH, obs['gap_y']))
            pygame.draw.rect(self.screen, OBSTACLE_COLOR, (obs['x'], obs['gap_y']+GAP_HEIGHT,
                                                          OBSTACLE_WIDTH, SCREEN_HEIGHT))
        for i, line in enumerate(text_lines):
            txt = self.font.render(line, True, TEXT_COLOR)
            self.screen.blit(txt, (10, 10 + i*25))
        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        pygame.quit()
