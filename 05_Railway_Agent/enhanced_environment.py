# enhanced_environment.py

import pygame
import numpy as np
import random
import time
import math
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 600
FONT_COLOR, BG_COLOR = (220, 220, 220), (20, 110, 40)
TRACK_COLOR, SLEEPER_COLOR = (80, 80, 80), (100, 80, 60)
CONFLICT_X_START, CONFLICT_X_END = 500, 700

# Enhanced train types with realistic properties
TRAIN_TYPES = {
    0: {'name': 'Freight Heavy', 'color': (50, 80, 200), 'priority': 0.3, 'max_speed': 60, 'passenger_capacity': 0},
    1: {'name': 'Express Passenger', 'color': (200, 50, 50), 'priority': 0.8, 'max_speed': 120, 'passenger_capacity': 400},
    2: {'name': 'Local Passenger', 'color': (50, 200, 50), 'priority': 0.6, 'max_speed': 80, 'passenger_capacity': 200},
    3: {'name': 'Emergency Medical', 'color': (255, 165, 0), 'priority': 1.0, 'max_speed': 100, 'passenger_capacity': 50}
}

class WeatherCondition(Enum):
    CLEAR = "clear"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    FOG = "fog"
    SNOW = "snow"

@dataclass
class TrainState:
    train_id: str
    train_type: int
    direction: str  # 'UP' or 'DOWN'
    position: float  # 0.0 to 1.0 across the screen
    velocity: float  # current speed in km/h
    target_velocity: float
    passenger_count: int
    delay_minutes: float
    priority_boost: float = 0.0
    emergency_flag: bool = False
    
class EnhancedTrain:
    def __init__(self, direction: str, train_type: Optional[int] = None, scenario_config: Optional[Dict] = None):
        self.direction = direction
        self.train_type = train_type if train_type is not None else random.choice(list(TRAIN_TYPES.keys()))
        self.type_info = TRAIN_TYPES[self.train_type]
        
        # Physical properties
        self.width, self.height = 100, 25
        self.position = 0.0 if direction == 'UP' else 1.0  # Normalized position
        self.velocity = random.uniform(40, self.type_info['max_speed'] * 0.8)  # km/h
        self.target_velocity = self.velocity
        
        # Operational properties
        self.passenger_count = self._generate_passenger_count()
        self.delay_minutes = random.uniform(0, 15) if random.random() < 0.3 else 0
        self.priority_boost = 0.0
        self.emergency_flag = (self.train_type == 3) or (random.random() < 0.05)
        
        # Visual properties
        self.y_pos = 200 if direction == 'UP' else 300
        self.state = 'approaching'  # approaching, waiting, passing, passed
        
        # Scenario-specific adjustments
        if scenario_config:
            self._apply_scenario_config(scenario_config)
    
    def _generate_passenger_count(self) -> int:
        """Generate realistic passenger count based on train type."""
        if self.type_info['passenger_capacity'] == 0:
            return 0
        
        # Vary occupancy by time and train type
        base_occupancy = random.uniform(0.3, 0.9)
        if self.train_type == 1:  # Express
            base_occupancy = random.uniform(0.6, 1.0)
        
        return int(self.type_info['passenger_capacity'] * base_occupancy)
    
    def _apply_scenario_config(self, config: Dict):
        """Apply scenario-specific configuration."""
        if 'weather_factor' in config:
            self.target_velocity *= config['weather_factor']
            self.velocity = min(self.velocity, self.target_velocity)
        
        if 'emergency_override' in config and config['emergency_override']:
            self.emergency_flag = True
            self.priority_boost = 0.5
        
        if 'delay_scenario' in config:
            self.delay_minutes = config['delay_scenario']
    
    def update(self, dt: float = 1/60):
        """Update train position and state."""
        if self.state in ['approaching', 'passing']:
            # Convert velocity from km/h to normalized screen units per second
            velocity_normalized = (self.velocity / 3.6) * dt / 100  # Simplified conversion
            
            if self.direction == 'UP':
                self.position += velocity_normalized
                if self.position >= 1.2:  # Off screen
                    self.state = 'passed'
            else:
                self.position -= velocity_normalized
                if self.position <= -0.2:  # Off screen
                    self.state = 'passed'
    
    def get_screen_position(self) -> Tuple[int, int]:
        """Convert normalized position to screen coordinates."""
        x = int(self.position * SCREEN_WIDTH - self.width/2)
        return x, int(self.y_pos - self.height/2)
    
    def is_in_conflict_zone(self) -> bool:
        """Check if train is in the conflict zone."""
        screen_x, _ = self.get_screen_position()
        return CONFLICT_X_START <= screen_x + self.width/2 <= CONFLICT_X_END
    
    def get_priority_score(self, weather: WeatherCondition, maintenance_active: bool) -> float:
        """Calculate dynamic priority score."""
        base_priority = self.type_info['priority']
        
        # Emergency boost
        if self.emergency_flag:
            base_priority = min(1.0, base_priority + 0.5)
        
        # Delay penalty/boost
        if self.delay_minutes > 10:
            base_priority += 0.2  # Boost delayed trains
        
        # Weather impact
        weather_factors = {
            WeatherCondition.CLEAR: 1.0,
            WeatherCondition.RAIN: 0.9,
            WeatherCondition.HEAVY_RAIN: 0.7,
            WeatherCondition.FOG: 0.8,
            WeatherCondition.SNOW: 0.6
        }
        base_priority *= weather_factors.get(weather, 1.0)
        
        # Maintenance impact
        if maintenance_active and self.train_type == 0:  # Freight during maintenance
            base_priority *= 0.5
        
        return min(1.0, base_priority + self.priority_boost)
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        """Enhanced drawing with detailed information."""
        screen_x, screen_y = self.get_screen_position()
        
        # Draw train body
        train_rect = pygame.Rect(screen_x, screen_y, self.width, self.height)
        color = self.type_info['color']
        
        # Highlight emergency trains
        if self.emergency_flag:
            pygame.draw.rect(screen, (255, 255, 0), train_rect.inflate(6, 6))
        
        pygame.draw.rect(screen, color, train_rect, border_radius=5)
        pygame.draw.rect(screen, (255, 255, 255), train_rect, width=2, border_radius=5)
        
        # Train label
        label = f"{self.type_info['name'][:8]}"
        text_surface = font.render(label, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(screen_x + self.width/2, screen_y + self.height/2))
        screen.blit(text_surface, text_rect)
        
        # Speed indicator
        speed_text = f"{int(self.velocity)} km/h"
        speed_surface = font.render(speed_text, True, FONT_COLOR)
        screen.blit(speed_surface, (screen_x, screen_y - 25))
        
        # Emergency indicator
        if self.emergency_flag:
            emergency_surface = font.render("EMERGENCY", True, (255, 0, 0))
            screen.blit(emergency_surface, (screen_x, screen_y + self.height + 5))

class ScenarioGenerator:
    """Generates training scenarios with varying difficulty levels."""
    
    def __init__(self):
        self.scenario_types = {
            'basic_priority': {'weight': 1.0, 'complexity': 1},
            'speed_differential': {'weight': 0.8, 'complexity': 2},
            'emergency': {'weight': 0.3, 'complexity': 4},
            'weather_impact': {'weight': 0.6, 'complexity': 3},
            'maintenance_window': {'weight': 0.4, 'complexity': 3},
            'passenger_freight': {'weight': 0.7, 'complexity': 2},
            'peak_hour': {'weight': 0.5, 'complexity': 4},
            'night_ops': {'weight': 0.4, 'complexity': 3}
        }
    
    def generate_scenario(self, scenario_type: str = None, difficulty_level: int = 1) -> Dict:
        """Generate a scenario configuration."""
        if scenario_type is None:
            # Choose based on difficulty level
            available_scenarios = [
                name for name, props in self.scenario_types.items()
                if props['complexity'] <= difficulty_level
            ]
            scenario_type = random.choices(
                available_scenarios,
                weights=[self.scenario_types[s]['weight'] for s in available_scenarios]
            )[0]
        
        config = {
            'scenario_type': scenario_type,
            'weather': random.choice(list(WeatherCondition)),
            'maintenance_active': random.random() < 0.1,
            'time_of_day': random.choice(['day', 'night', 'rush_hour']),
            'difficulty_level': difficulty_level
        }
        
        # Scenario-specific configurations
        if scenario_type == 'emergency':
            config['emergency_override'] = True
        elif scenario_type == 'weather_impact':
            config['weather'] = random.choice([WeatherCondition.RAIN, WeatherCondition.FOG, WeatherCondition.SNOW])
            config['weather_factor'] = random.uniform(0.6, 0.9)
        elif scenario_type == 'peak_hour':
            config['time_of_day'] = 'rush_hour'
            config['passenger_boost'] = True
        elif scenario_type == 'maintenance_window':
            config['maintenance_active'] = True
        
        return config

class EnhancedEnvironment:
    """Enhanced railway junction environment with rich state representation."""
    
    def __init__(self, gui_enabled: bool = True):
        self.gui_enabled = gui_enabled
        if gui_enabled:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Enhanced Railway Junction Controller")
            self.clock = pygame.time.Clock()
            self.font_small = pygame.font.Font(None, 24)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_large = pygame.font.Font(None, 48)
        
        self.scenario_generator = ScenarioGenerator()
        self.reset()
    
    def reset(self, scenario_config: Optional[Dict] = None) -> np.ndarray:
        """Reset environment and return initial state."""
        if scenario_config is None:
            scenario_config = self.scenario_generator.generate_scenario()
        
        self.scenario_config = scenario_config
        self.weather = scenario_config.get('weather', WeatherCondition.CLEAR)
        self.maintenance_active = scenario_config.get('maintenance_active', False)
        self.time_of_day = scenario_config.get('time_of_day', 'day')
        
        # Create trains
        self.up_train = EnhancedTrain('UP', scenario_config=scenario_config)
        self.down_train = EnhancedTrain('DOWN', scenario_config=scenario_config)
        
        # Environment state
        self.conflict_detected = False
        self.decision_made = False
        self.episode_complete = False
        self.conflict_resolution_time = 0
        self.total_delay = 0

        ## NEW ##: Acknowledgment message state
        self.acknowledgment_message = None
        self.acknowledgment_timer = 0
        
        return self.get_state_vector()
    
    def get_state_vector(self) -> np.ndarray:
        """Get comprehensive state representation (14 features)."""
        # One-hot encoding for train types (simplified to main categories)
        up_type_onehot = [0, 0]
        down_type_onehot = [0, 0]
        
        # Simplify to freight(0,1) vs passenger(1,2,3) for one-hot
        up_category = 0 if self.up_train.train_type == 0 else 1
        down_category = 0 if self.down_train.train_type == 0 else 1
        
        up_type_onehot[up_category] = 1
        down_type_onehot[down_category] = 1
        
        # Calculate derived features
        time_to_collision = self._calculate_collision_time()
        
        state = np.array([
            # Train type one-hot (4 features)
            up_type_onehot[0], up_type_onehot[1],
            down_type_onehot[0], down_type_onehot[1],
            
            # Position and velocity (4 features)
            self.up_train.position,
            self.down_train.position,
            self.up_train.velocity / 120.0,  # Normalize by max speed
            self.down_train.velocity / 120.0,
            
            # Temporal and priority factors (4 features)
            min(time_to_collision / 60.0, 1.0),  # Normalize to minutes
            self.up_train.get_priority_score(self.weather, self.maintenance_active),
            self.down_train.get_priority_score(self.weather, self.maintenance_active),
            
            # Passenger and operational factors (2 features)
            self.up_train.passenger_count / 500.0,  # Normalize passenger count
            self.down_train.passenger_count / 500.0,
            
            # Environmental factors (2 features)
            self._get_weather_factor(),
            1.0 if self.maintenance_active else 0.0,
            
            # Emergency flag (1 feature)
            1.0 if (self.up_train.emergency_flag or self.down_train.emergency_flag) else 0.0
        ], dtype=np.float32)
        
        return state
    
    def _calculate_collision_time(self) -> float:
        """Calculate time to potential collision."""
        if self.up_train.position >= self.down_train.position:
            return float('inf')  # Already passed
        
        relative_speed = (self.up_train.velocity + self.down_train.velocity) / 3.6  # m/s
        distance = abs(self.down_train.position - self.up_train.position) * 1000  # meters
        
        if relative_speed <= 0:
            return float('inf')
        
        return max(distance / relative_speed, 1.0)
    
    def _get_weather_factor(self) -> float:
        """Get weather impact factor."""
        weather_factors = {
            WeatherCondition.CLEAR: 1.0,
            WeatherCondition.RAIN: 0.9,
            WeatherCondition.HEAVY_RAIN: 0.7,
            WeatherCondition.FOG: 0.8,
            WeatherCondition.SNOW: 0.6
        }
        return weather_factors.get(self.weather, 1.0)
    
    def step(self, action: Optional[int] = None) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute environment step."""
        # ## NEW ##: Update the acknowledgment timer
        if self.acknowledgment_timer > 0:
            self.acknowledgment_timer -= 1
        else:
            self.acknowledgment_message = None

        # Update trains
        self.up_train.update()
        self.down_train.update()
        
        # Check for conflict
        if not self.conflict_detected and not self.decision_made:
            if self._check_conflict():
                self.conflict_detected = True
                if action is not None:
                    self._resolve_conflict(action)
                    self.decision_made = True
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if episode is complete
        done = self.up_train.state == 'passed' and self.down_train.state == 'passed'
        
        # Info dict
        info = {
            'conflict_detected': self.conflict_detected,
            'decision_made': self.decision_made,
            'total_delay': self.total_delay,
            'scenario_type': self.scenario_config.get('scenario_type', 'unknown'),
            'weather': self.weather.value,
            'emergency_active': self.up_train.emergency_flag or self.down_train.emergency_flag
        }
        
        return self.get_state_vector(), reward, done, info
    
    def _check_conflict(self) -> bool:
        """Check if trains are approaching conflict zone."""
        up_approaching = (0.4 <= self.up_train.position <= 0.7)
        down_approaching = (0.3 <= self.down_train.position <= 0.6)
        return up_approaching and down_approaching
    
    def _resolve_conflict(self, action: int):
        """Resolve conflict based on action (0=UP priority, 1=DOWN priority)."""
        self.conflict_resolution_time = time.time()
        
        if action == 0:  # Prioritize UP train
            self.up_train.state = 'passing'
            self.down_train.state = 'waiting'
            self.down_train.y_pos = 380  # Move to siding
            self.total_delay += random.uniform(2, 5)
            ## NEW ##: Set acknowledgment message
            self.acknowledgment_message = "UP TRAIN PRIORITIZED"
        else:  # Prioritize DOWN train
            self.down_train.state = 'passing'
            self.up_train.state = 'waiting'
            self.up_train.y_pos = 120  # Move to siding
            self.total_delay += random.uniform(2, 5)
            ## NEW ##: Set acknowledgment message
            self.acknowledgment_message = "DOWN TRAIN PRIORITIZED"

        ## NEW ##: Set timer for the message to be displayed (120 frames = 2 seconds)
        self.acknowledgment_timer = 120
    
    def _calculate_reward(self, action: Optional[int]) -> float:
        """Calculate reward based on decision quality."""
        if not self.decision_made or action is None:
            return 0.0
        
        up_priority = self.up_train.get_priority_score(self.weather, self.maintenance_active)
        down_priority = self.down_train.get_priority_score(self.weather, self.maintenance_active)
        
        # Base reward for correct priority decision
        if action == 0:  # Chose UP
            priority_reward = up_priority - down_priority
        else:  # Chose DOWN
            priority_reward = down_priority - up_priority
        
        # Bonus for emergency situations
        emergency_bonus = 0.0
        if self.up_train.emergency_flag and action == 0:
            emergency_bonus = 0.5
        elif self.down_train.emergency_flag and action == 1:
            emergency_bonus = 0.5
        
        # Penalty for delay
        delay_penalty = -self.total_delay * 0.1
        
        total_reward = priority_reward + emergency_bonus + delay_penalty
        return np.clip(total_reward, -1.0, 1.0)
    
    ## MODIFIED ##: Added `decision_info` parameter for AI testing display
    def render(self, info_lines: List[str] = None, decision_info: Optional[Dict] = None):
        """Render the environment."""
        if not self.gui_enabled:
            return
        
        self.screen.fill(BG_COLOR)
        
        # Draw tracks
        self._draw_tracks()
        
        # Draw trains
        self.up_train.draw(self.screen, self.font_small)
        self.down_train.draw(self.screen, self.font_small)
        
        # Draw conflict zone
        conflict_rect = pygame.Rect(CONFLICT_X_START, 150, CONFLICT_X_END - CONFLICT_X_START, 200)
        pygame.draw.rect(self.screen, (255, 255, 0), conflict_rect, width=3)
        
        # Draw UI panels
        self._draw_ui_panels()
        
        # Draw info lines
        if info_lines:
            for i, line in enumerate(info_lines):
                text_surface = self.font_medium.render(line, True, FONT_COLOR)
                self.screen.blit(text_surface, (10, 10 + i * 35))
        
        # ## NEW ##: Draw various overlays (decision prompt, ack message, AI info)
        self._draw_overlays(decision_info)
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def _draw_tracks(self):
        """Draw enhanced track system."""
        # Main tracks
        pygame.draw.line(self.screen, TRACK_COLOR, (0, 200), (CONFLICT_X_START, 200), 8)
        pygame.draw.line(self.screen, TRACK_COLOR, (0, 300), (SCREEN_WIDTH, 300), 8)
        pygame.draw.line(self.screen, TRACK_COLOR, (CONFLICT_X_END, 200), (SCREEN_WIDTH, 200), 8)
        
        # Junction
        pygame.draw.line(self.screen, TRACK_COLOR, (CONFLICT_X_START, 200), (CONFLICT_X_END, 300), 8)
        
        # Sidings
        pygame.draw.line(self.screen, TRACK_COLOR, (CONFLICT_X_START - 100, 120), (CONFLICT_X_END + 100, 120), 4)
        pygame.draw.line(self.screen, TRACK_COLOR, (CONFLICT_X_START - 100, 380), (CONFLICT_X_END + 100, 380), 4)
        
        # Connecting lines
        pygame.draw.line(self.screen, TRACK_COLOR, (CONFLICT_X_START, 200), (CONFLICT_X_START - 50, 120), 4)
        pygame.draw.line(self.screen, TRACK_COLOR, (CONFLICT_X_END, 200), (CONFLICT_X_END + 50, 120), 4)
        pygame.draw.line(self.screen, TRACK_COLOR, (CONFLICT_X_START, 300), (CONFLICT_X_START - 50, 380), 4)
        pygame.draw.line(self.screen, TRACK_COLOR, (CONFLICT_X_END, 300), (CONFLICT_X_END + 50, 380), 4)
    
    def _draw_ui_panels(self):
        """Draw information panels."""
        # Weather panel
        weather_text = f"Weather: {self.weather.value.title()}"
        weather_surface = self.font_medium.render(weather_text, True, FONT_COLOR)
        self.screen.blit(weather_surface, (SCREEN_WIDTH - 250, 10))
        
        # Maintenance panel
        if self.maintenance_active:
            maint_surface = self.font_medium.render("MAINTENANCE ACTIVE", True, (255, 165, 0))
            self.screen.blit(maint_surface, (SCREEN_WIDTH - 250, 40))
        
        # Train info panels
        up_info = f"UP: {self.up_train.type_info['name']} | {self.up_train.passenger_count} pax"
        down_info = f"DOWN: {self.down_train.type_info['name']} | {self.down_train.passenger_count} pax"
        
        up_surface = self.font_small.render(up_info, True, FONT_COLOR)
        down_surface = self.font_small.render(down_info, True, FONT_COLOR)
        
        self.screen.blit(up_surface, (10, SCREEN_HEIGHT - 80))
        self.screen.blit(down_surface, (10, SCREEN_HEIGHT - 50))

    ## NEW ##: Central function to handle all screen overlays
    def _draw_overlays(self, decision_info: Optional[Dict] = None):
        """Draw overlays for decisions, acknowledgments, and AI info."""
        if self.conflict_detected and not self.decision_made:
            self._draw_decision_overlay()
        elif self.acknowledgment_message and self.acknowledgment_timer > 0:
            self._draw_acknowledgment_overlay()
        
        if decision_info:
            self._draw_ai_decision_overlay(decision_info)

    def _draw_decision_overlay(self):
        """Draw decision prompt overlay."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        # Title
        title = self.font_large.render("CONFLICT DETECTED - DECISION REQUIRED", True, (255, 200, 0))
        title_rect = title.get_rect(center=(SCREEN_WIDTH/2, 100))
        self.screen.blit(title, title_rect)
        
        # Train information
        up_priority = self.up_train.get_priority_score(self.weather, self.maintenance_active)
        down_priority = self.down_train.get_priority_score(self.weather, self.maintenance_active)
        
        up_text = f"UP: {self.up_train.type_info['name']} | Priority: {up_priority:.2f} | {self.up_train.passenger_count} passengers"
        down_text = f"DOWN: {self.down_train.type_info['name']} | Priority: {down_priority:.2f} | {self.down_train.passenger_count} passengers"
        
        up_surface = self.font_medium.render(up_text, True, FONT_COLOR)
        down_surface = self.font_medium.render(down_text, True, FONT_COLOR)
        
        self.screen.blit(up_surface, (50, 200))
        self.screen.blit(down_surface, (50, 240))
        
        # Instructions
        instruction = "Press [U] to prioritize UP train or [D] to prioritize DOWN train"
        inst_surface = self.font_medium.render(instruction, True, (255, 255, 255))
        inst_rect = inst_surface.get_rect(center=(SCREEN_WIDTH/2, 350))
        self.screen.blit(inst_surface, inst_rect)
        
        # Additional context
        collision_time = self._calculate_collision_time()
        if collision_time != float('inf'):
            time_text = f"Time to collision: {collision_time:.1f} seconds"
            time_surface = self.font_small.render(time_text, True, (255, 100, 100))
            time_rect = time_surface.get_rect(center=(SCREEN_WIDTH/2, 400))
            self.screen.blit(time_surface, time_rect)

    ## NEW ##: Draw the action acknowledgment message
    def _draw_acknowledgment_overlay(self):
        """Draw a message confirming which action was taken."""
        ack_surface = self.font_large.render(self.acknowledgment_message, True, (100, 255, 100))
        ack_rect = ack_surface.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
        self.screen.blit(ack_surface, ack_rect)

    ## NEW ##: Draw AI decision details during testing
    def _draw_ai_decision_overlay(self, decision_info: Dict):
        """Draw an overlay showing the AI's decision during testing."""
        ai_action = "UP" if decision_info['ai_action'] == 0 else "DOWN"
        expert_action = "UP" if decision_info['expert_action'] == 0 else "DOWN"
        confidence = decision_info['confidence']
        is_match = (ai_action == expert_action)
        
        # Box for info
        info_panel = pygame.Rect(SCREEN_WIDTH - 320, SCREEN_HEIGHT - 130, 310, 120)
        pygame.draw.rect(self.screen, (30, 30, 30), info_panel, border_radius=10)
        pygame.draw.rect(self.screen, (200, 200, 200), info_panel, width=2, border_radius=10)

        # Title
        title_surf = self.font_medium.render("AI Decision", True, (255, 200, 0))
        self.screen.blit(title_surf, (info_panel.x + 15, info_panel.y + 10))
        
        # AI choice
        ai_text = f"Choice: {ai_action} (Conf: {confidence:.2f})"
        ai_surf = self.font_small.render(ai_text, True, FONT_COLOR)
        self.screen.blit(ai_surf, (info_panel.x + 15, info_panel.y + 45))

        # Expert choice
        expert_text = f"Expert Choice: {expert_action}"
        expert_surf = self.font_small.render(expert_text, True, FONT_COLOR)
        self.screen.blit(expert_surf, (info_panel.x + 15, info_panel.y + 70))

        # Match status
        match_color = (100, 255, 100) if is_match else (255, 100, 100)
        match_text = "MATCH ✓" if is_match else "MISMATCH ✗"
        match_surf = self.font_medium.render(match_text, True, match_color)
        self.screen.blit(match_surf, (info_panel.x + 15, info_panel.y + 90))

    def get_human_action(self) -> Optional[int]:
        """Get action from human input."""
        if not self.gui_enabled:
            return None
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return -1
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_u:
                    return 0  # UP priority
                elif event.key == pygame.K_d:
                    return 1  # DOWN priority
                elif event.key == pygame.K_ESCAPE:
                    return -1  # Quit
        return None
    
    def close(self):
        """Clean up resources."""
        if self.gui_enabled:
            pygame.quit()

# Demo function
def run_demo():
    """Run a demonstration of the enhanced environment."""
    env = EnhancedEnvironment(gui_enabled=True)
    
    print("Enhanced Railway Junction Environment Demo")
    print("Press U or D to make decisions when conflicts arise")
    print("Press ESC to quit")
    
    episode = 0
    while episode < 5:  # Run 5 demo episodes
        episode += 1
        print(f"\n=== Episode {episode} ===")
        
        state = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < 1000:
            # Render environment
            info_lines = [
                f"Episode: {episode}/5",
                f"Step: {step_count}",
                f"Scenario: {env.scenario_config.get('scenario_type', 'unknown')}",
                f"Weather: {env.weather.value}",
                f"Emergency: {'YES' if env.up_train.emergency_flag or env.down_train.emergency_flag else 'NO'}"
            ]
            
            env.render(info_lines)
            
            # Get human action if conflict detected
            action = None
            if env.conflict_detected and not env.decision_made:
                action = env.get_human_action()
                if action == -1:  # Quit
                    env.close()
                    return
            
            # Step environment
            state, reward, done, info = env.step(action)
            
            if action is not None and action >= 0:
                print(f"Action taken: {'UP' if action == 0 else 'DOWN'} priority")
                print(f"Reward: {reward:.3f}")
                print(f"Total delay: {info['total_delay']:.1f} minutes")
            
            step_count += 1
            time.sleep(0.05)  # Small delay for visualization
        
        print(f"Episode {episode} completed in {step_count} steps")
        time.sleep(2)
    
    env.close()
    print("Demo completed!")

if __name__ == "__main__":
    run_demo()