import pygame
import numpy as np

# --- Constants ---
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
ROAD_WIDTH = 200
ROAD_LEFT = (SCREEN_WIDTH - ROAD_WIDTH) / 2
ROAD_RIGHT = ROAD_LEFT + ROAD_WIDTH

class Car:
    def __init__(self):
        self.width = 30
        self.height = 50
        self.reset()

    def reset(self):
        self.x = SCREEN_WIDTH / 2
        self.y = SCREEN_HEIGHT - 100
        self.velocity_x = 0

    def update(self, action):
        if action == 0: self.velocity_x -= 0.5
        elif action == 2: self.velocity_x += 0.5
        self.velocity_x *= 0.95
        self.x += self.velocity_x
    
    # Pygame rect for collision and drawing
    @property
    def rect(self):
        return pygame.Rect(self.x - self.width / 2, self.y - self.height / 2, self.width, self.height)

class Environment:
    def __init__(self, headless=False):
        self.headless = headless
        if not self.headless:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Policy Gradient Car")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)
            self.car_image = pygame.Surface([30, 50])
            self.car_image.fill((255, 0, 0))

        self.car = Car()
        self.scroll_speed = 5
        self.center_lines = []
        if not self.headless:
            for y in range(-40, SCREEN_HEIGHT, 40):
                self.center_lines.append(pygame.Rect(SCREEN_WIDTH/2 - 2, y, 4, 20))

    def reset(self):
        self.car.reset()
        pos_norm = (self.car.x - SCREEN_WIDTH / 2) / (ROAD_WIDTH / 2)
        vel_norm = self.car.velocity_x / 5.0
        return np.array([pos_norm, vel_norm])

    def step(self, action):
        self.car.update(action)
        
        done = False
        if self.car.x - self.car.width / 2 < ROAD_LEFT or self.car.x + self.car.width / 2 > ROAD_RIGHT:
            reward = -10
            done = True
        else:
            distance_from_center = abs(self.car.x - SCREEN_WIDTH / 2)
            reward = 1.0 - (distance_from_center / (ROAD_WIDTH / 2))

        pos_norm = (self.car.x - SCREEN_WIDTH / 2) / (ROAD_WIDTH / 2)
        vel_norm = self.car.velocity_x / 5.0
        next_state = np.array([pos_norm, vel_norm])

        return next_state, reward, done

    def render(self, episode=0, total_reward=0):
        if self.headless: return

        self.screen.fill((100, 100, 100))
        pygame.draw.rect(self.screen, (50, 50, 50), (ROAD_LEFT, 0, ROAD_WIDTH, SCREEN_HEIGHT))
        
        for line in self.center_lines:
            line.y += self.scroll_speed
            if line.y > SCREEN_HEIGHT: line.y -= (SCREEN_HEIGHT + 40)
            pygame.draw.rect(self.screen, (255, 255, 255), line)

        self.screen.blit(self.car_image, self.car.rect)
        
        episode_text = self.font.render(f"Episode: {episode}", True, (255, 255, 255))
        reward_text = self.font.render(f"Score: {total_reward:.2f}", True, (255, 255, 255))
        self.screen.blit(episode_text, (10, 10))
        self.screen.blit(reward_text, (10, 40))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if not self.headless:
            pygame.quit()
