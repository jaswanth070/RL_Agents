import pygame
import numpy as np
import math

# --- Constants ---
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
ROAD_WIDTH = 200

# --- Colors ---
GRASS_COLOR = (20, 110, 40)
ROAD_COLOR = (50, 50, 50)
LINE_COLOR = (255, 255, 255)

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
    
    @property
    def rect(self):
        return pygame.Rect(self.x - self.width / 2, self.y - self.height / 2, self.width, self.height)

class Environment:
    def __init__(self, headless=False):
        self.headless = headless
        if not self.headless:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Curved Track Driving")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)
            self._create_car_sprite()

        self.car = Car()
        self.scroll_y = 0
        self.road_points = []
        self.reset()

    def _create_car_sprite(self):
        self.car_sprite = pygame.Surface((30, 50), pygame.SRCALPHA)
        # Main body
        pygame.draw.rect(self.car_sprite, (200, 0, 0), (0, 5, 30, 40), border_radius=5)
        # Windshield
        pygame.draw.polygon(self.car_sprite, (173, 216, 230), [(5, 10), (25, 10), (20, 20), (10, 20)])
        # Roof
        pygame.draw.rect(self.car_sprite, (180, 0, 0), (5, 20, 20, 20), border_radius=3)


    def _generate_road(self):
        self.road_points = []
        frequency = np.random.uniform(0.01, 0.03)
        amplitude = np.random.uniform(50, 100)
        phase = np.random.uniform(0, np.pi * 2)
        
        # Generate points for a much longer road to scroll through
        for y in range(-SCREEN_HEIGHT, SCREEN_HEIGHT * 2):
            center_x = SCREEN_WIDTH / 2 + amplitude * math.sin(frequency * y + phase)
            self.road_points.append({'y': y, 'x': center_x})
        self.scroll_y = 0


    def reset(self):
        self._generate_road()
        self.car.reset()
        return self._get_state()

    def _get_centerline_x_at(self, y_pos):
        # Find the closest point in our list to the given y_pos
        # This is a simple way; more complex interpolation could be used for more precision
        closest_point = min(self.road_points, key=lambda p: abs(p['y'] - y_pos))
        return closest_point['x']

    def _get_state(self):
        car_scroll_y = self.scroll_y + self.car.y

        current_center_x = self._get_centerline_x_at(car_scroll_y)
        dist_from_center = (self.car.x - current_center_x) / (ROAD_WIDTH / 2)
        velocity_norm = self.car.velocity_x / 5.0

        lookahead1_center_x = self._get_centerline_x_at(car_scroll_y - 40)
        lookahead2_center_x = self._get_centerline_x_at(car_scroll_y - 80)
        
        offset1 = (lookahead1_center_x - current_center_x) / (ROAD_WIDTH / 2)
        offset2 = (lookahead2_center_x - current_center_x) / (ROAD_WIDTH / 2)

        return np.array([dist_from_center, velocity_norm, offset1, offset2])

    def step(self, action):
        self.car.update(action)
        self.scroll_y += 5

        done = False
        center_x_at_car = self._get_centerline_x_at(self.scroll_y + self.car.y)
        road_left_at_car = center_x_at_car - ROAD_WIDTH / 2
        road_right_at_car = center_x_at_car + ROAD_WIDTH / 2
        
        if self.car.x < road_left_at_car or self.car.x > road_right_at_car:
            reward = -10
            done = True
        else:
            distance = abs(self.car.x - center_x_at_car)
            reward = 1.0 - (distance / (ROAD_WIDTH / 2))

        # Check if we've run out of road (end of episode)
        if self.scroll_y + SCREEN_HEIGHT > self.road_points[-1]['y']:
             done = True

        return self._get_state(), reward, done

    def render(self, episode=0, total_reward=0):
        if self.headless: return

        self.screen.fill(GRASS_COLOR)

        # Draw the visible part of the road
        visible_points = [p for p in self.road_points if self.scroll_y <= p['y'] < self.scroll_y + SCREEN_HEIGHT + 20]
        
        for i in range(len(visible_points) - 1):
            p1 = visible_points[i]
            p2 = visible_points[i+1]
            
            # Road surface
            pygame.draw.polygon(self.screen, ROAD_COLOR, [
                (p1['x'] - ROAD_WIDTH / 2, p1['y'] - self.scroll_y),
                (p2['x'] - ROAD_WIDTH / 2, p2['y'] - self.scroll_y),
                (p2['x'] + ROAD_WIDTH / 2, p2['y'] - self.scroll_y),
                (p1['x'] + ROAD_WIDTH / 2, p1['y'] - self.scroll_y)
            ])
            # Road Edges (Anti-aliased)
            pygame.draw.aaline(self.screen, LINE_COLOR, (p1['x'] - ROAD_WIDTH/2, p1['y']-self.scroll_y), (p2['x'] - ROAD_WIDTH/2, p2['y']-self.scroll_y))
            pygame.draw.aaline(self.screen, LINE_COLOR, (p1['x'] + ROAD_WIDTH/2, p1['y']-self.scroll_y), (p2['x'] + ROAD_WIDTH/2, p2['y']-self.scroll_y))
            
            # Centerline dashes
            if i % 10 == 0:
                 pygame.draw.line(self.screen, LINE_COLOR, (p1['x'], p1['y']-self.scroll_y), (p2['x'], p2['y']-self.scroll_y), 5)


        self.screen.blit(self.car_sprite, self.car.rect)
        
        episode_text = self.font.render(f"Episode: {episode}", True, LINE_COLOR)
        reward_text = self.font.render(f"Score: {total_reward:.2f}", True, LINE_COLOR)
        self.screen.blit(episode_text, (10, 10))
        self.screen.blit(reward_text, (10, 40))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if not self.headless:
            pygame.quit()

