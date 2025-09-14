# Enhanced Railway Junction RL Agent with Imitation Learning
# This implementation addresses the bugs and limitations in the original code

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pickle
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedJunctionModel(nn.Module):
    """Enhanced neural network model for railway junction control."""
    
    def __init__(self, input_size=14, hidden_sizes=[128, 64, 32, 16]):
        super(EnhancedJunctionModel, self).__init__()
        
        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        
        # Hidden layers with dropout
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        
        # Dropout layers
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        
        # Output layers
        self.priority_head = nn.Linear(hidden_sizes[-1], 2)  # Priority scores
        self.confidence_head = nn.Linear(hidden_sizes[-1], 1)  # Decision confidence
        self.delay_head = nn.Linear(hidden_sizes[-1], 2)  # Expected delays
        
    def forward(self, x):
        # Input layer
        x = F.relu(self.input_layer(x))
        x = self.dropout1(x)
        
        # First hidden layer
        x = F.relu(self.hidden_layers[0](x))
        x = self.dropout2(x)
        
        # Remaining hidden layers
        for layer in self.hidden_layers[1:]:
            x = F.relu(layer(x))
        
        # Output heads
        priorities = torch.softmax(self.priority_head(x), dim=-1)
        confidence = torch.sigmoid(self.confidence_head(x))
        delays = F.relu(self.delay_head(x))
        
        return {
            'priorities': priorities,
            'confidence': confidence,
            'expected_delays': delays
        }

class EnhancedTrainingSystem:
    """Enhanced training system with curriculum learning and comprehensive evaluation."""
    
    def __init__(self, model_dir="models", data_dir="data"):
        self.model_dir = model_dir
        self.data_dir = data_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        self.model = EnhancedJunctionModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training history
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'phase': [],
            'episode': []
        }
        
        # Curriculum phases
        self.curriculum_phases = [
            {'name': 'Foundation', 'episodes': 100, 'scenarios': ['basic_priority', 'speed_differential']},
            {'name': 'Complexity', 'episodes': 100, 'scenarios': ['emergency', 'passenger_freight', 'maintenance']},
            {'name': 'Multi-factor', 'episodes': 80, 'scenarios': ['weather', 'schedule_recovery', 'night_ops']},
            {'name': 'Expert', 'episodes': 35, 'scenarios': ['peak_hour', 'multiple_trains', 'breakdown']}
        ]
        
    def collect_enhanced_data(self, environment, num_episodes_per_phase):
        """Collect training data with enhanced features and curriculum learning."""
        
        all_data = []
        current_episode = 0
        
        for phase_idx, phase in enumerate(self.curriculum_phases):
            logger.info(f"Starting {phase['name']} phase - {phase['episodes']} episodes")
            
            phase_data = []
            for episode in range(phase['episodes']):
                # Generate scenario for this phase
                scenario_data = self._generate_scenario(phase['scenarios'])
                
                # Get human decision with rich context
                decision_data = self._get_expert_decision(scenario_data, environment)
                
                if decision_data:
                    # Enhance data with computed features
                    enhanced_data = self._enhance_data_point(scenario_data, decision_data)
                    phase_data.append(enhanced_data)
                    all_data.append(enhanced_data)
                    
                current_episode += 1
                
                # Progress reporting
                if episode % 10 == 0:
                    logger.info(f"Phase {phase['name']}: {episode}/{phase['episodes']} episodes completed")
            
            # Save phase data
            phase_filename = f"{self.data_dir}/phase_{phase_idx+1}_{phase['name'].lower()}_data.pkl"
            with open(phase_filename, 'wb') as f:
                pickle.dump(phase_data, f)
            
            logger.info(f"Phase {phase['name']} completed. {len(phase_data)} decisions collected.")
        
        # Save complete dataset
        dataset_filename = f"{self.data_dir}/complete_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(dataset_filename, 'wb') as f:
            pickle.dump(all_data, f)
        
        logger.info(f"Total dataset: {len(all_data)} decisions saved to {dataset_filename}")
        return all_data
    
    def _generate_scenario(self, scenario_types):
        """Generate a scenario based on the current training phase."""
        scenario_type = np.random.choice(scenario_types)
        
        # Base scenario generation (simplified for demonstration)
        scenario = {
            'type': scenario_type,
            'up_train_type': np.random.randint(0, 4),  # Expanded train types
            'down_train_type': np.random.randint(0, 4),
            'up_position': np.random.uniform(0, 1),
            'down_position': np.random.uniform(0, 1),
            'up_velocity': np.random.uniform(20, 80),  # km/h
            'down_velocity': np.random.uniform(20, 80),
            'weather_factor': np.random.uniform(0.7, 1.0),
            'maintenance_flag': np.random.choice([0, 1], p=[0.9, 0.1]),
            'emergency_flag': np.random.choice([0, 1], p=[0.95, 0.05]),
            'timestamp': time.time()
        }
        
        # Compute derived features
        scenario['time_to_collision'] = self._compute_collision_time(scenario)
        scenario['up_priority'] = self._compute_train_priority(scenario, 'up')
        scenario['down_priority'] = self._compute_train_priority(scenario, 'down')
        
        return scenario
    
    def _compute_collision_time(self, scenario):
        """Compute time to potential collision."""
        # Simplified calculation
        relative_speed = (scenario['up_velocity'] + scenario['down_velocity']) / 3.6  # m/s
        distance = abs(scenario['up_position'] - scenario['down_position']) * 1000  # meters
        return max(distance / relative_speed, 1.0)  # minimum 1 second
    
    def _compute_train_priority(self, scenario, direction):
        """Compute base priority for a train based on type and conditions."""
        train_type = scenario[f'{direction}_train_type']
        
        # Base priorities by type
        base_priorities = {0: 0.3, 1: 0.8, 2: 0.6, 3: 0.9}  # freight, express, passenger, emergency
        priority = base_priorities.get(train_type, 0.5)
        
        # Adjust for emergency
        if scenario['emergency_flag'] and train_type == 3:
            priority = 1.0
        
        return priority
    
    def _get_expert_decision(self, scenario, environment):
        """Get expert decision with justification (simplified for demo)."""
        # In real implementation, this would present the scenario to human expert
        # For demo, we'll simulate expert decision based on priorities
        
        up_score = scenario['up_priority']
        down_score = scenario['down_priority']
        
        # Add some realistic decision logic
        if scenario['emergency_flag']:
            if scenario['up_train_type'] == 3:  # Emergency train going up
                decision = 0
                confidence = 0.95
            elif scenario['down_train_type'] == 3:  # Emergency train going down
                decision = 1
                confidence = 0.95
            else:
                decision = 0 if up_score > down_score else 1
                confidence = 0.7
        else:
            decision = 0 if up_score > down_score else 1
            confidence = abs(up_score - down_score)
        
        return {
            'decision': decision,
            'confidence': confidence,
            'justification': f"Chose {'UP' if decision == 0 else 'DOWN'} train based on priority scores",
            'timestamp': time.time()
        }
    
    def _enhance_data_point(self, scenario, decision):
        """Create enhanced feature vector for training."""
        # Convert categorical train types to one-hot
        up_train_onehot = [0] * 4
        down_train_onehot = [0] * 4
        up_train_onehot[scenario['up_train_type']] = 1
        down_train_onehot[scenario['down_train_type']] = 1
        
        # Create feature vector (14 features total)
        features = (
            up_train_onehot[:2] +  # Simplified to 2 main types for demo
            down_train_onehot[:2] +
            [
                scenario['up_position'],
                scenario['down_position'],
                scenario['up_velocity'] / 100.0,  # Normalize
                scenario['down_velocity'] / 100.0,
                min(scenario['time_to_collision'] / 60.0, 1.0),  # Normalize to minutes
                scenario['up_priority'],
                scenario['down_priority'],
                np.random.uniform(0.5, 1.0),  # Passenger count (simulated)
                np.random.uniform(0.5, 1.0),  # Passenger count (simulated)
                scenario['weather_factor'],
                scenario['maintenance_flag'],
                scenario['emergency_flag']
            ]
        )
        
        return {
            'features': np.array(features, dtype=np.float32),
            'decision': decision['decision'],
            'confidence': decision['confidence'],
            'metadata': {
                'scenario_type': scenario['type'],
                'justification': decision['justification'],
                'timestamp': scenario['timestamp']
            }
        }
    
    def train_model(self, training_data, validation_split=0.2, epochs=100):
        """Train the enhanced model with comprehensive loss function."""
        
        # Prepare data
        features = np.array([d['features'] for d in training_data])
        decisions = np.array([d['decision'] for d in training_data])
        confidences = np.array([d['confidence'] for d in training_data])
        
        # Split data
        n_train = int(len(features) * (1 - validation_split))
        
        train_features = torch.FloatTensor(features[:n_train])
        train_decisions = torch.LongTensor(decisions[:n_train])
        train_confidences = torch.FloatTensor(confidences[:n_train])
        
        val_features = torch.FloatTensor(features[n_train:])
        val_decisions = torch.LongTensor(decisions[n_train:])
        val_confidences = torch.FloatTensor(confidences[n_train:])
        
        # Create data loaders
        train_dataset = TensorDataset(train_features, train_decisions, train_confidences)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Training loop
        self.model.train()
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            n_batches = 0
            
            for batch_features, batch_decisions, batch_confidences in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_features)
                
                # Multi-objective loss
                priority_loss = F.cross_entropy(outputs['priorities'], batch_decisions)
                confidence_loss = F.mse_loss(outputs['confidence'].squeeze(), batch_confidences)
                
                # Combined loss (weighted)
                total_loss = 0.7 * priority_loss + 0.3 * confidence_loss
                
                # Backward pass
                total_loss.backward()
                self.optimizer.step()
                
                # Metrics
                epoch_loss += total_loss.item()
                predictions = torch.argmax(outputs['priorities'], dim=1)
                epoch_accuracy += (predictions == batch_decisions).float().mean().item()
                n_batches += 1
            
            # Validation
            val_loss, val_accuracy = self._evaluate_model(val_features, val_decisions, val_confidences)
            
            # Record history
            avg_loss = epoch_loss / n_batches
            avg_accuracy = epoch_accuracy / n_batches
            
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(avg_accuracy)
            self.training_history['episode'].append(epoch)
            
            # Logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}: "
                          f"Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}, "
                          f"Val_Loss={val_loss:.4f}, Val_Acc={val_accuracy:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f"{self.model_dir}/best_model.pth")
        
        logger.info("Training completed!")
        return self.training_history
    
    def _evaluate_model(self, features, decisions, confidences):
        """Evaluate model on validation data."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features)
            
            priority_loss = F.cross_entropy(outputs['priorities'], decisions)
            confidence_loss = F.mse_loss(outputs['confidence'].squeeze(), confidences)
            total_loss = 0.7 * priority_loss + 0.3 * confidence_loss
            
            predictions = torch.argmax(outputs['priorities'], dim=1)
            accuracy = (predictions == decisions).float().mean().item()
            
        self.model.train()
        return total_loss.item(), accuracy
    
    def save_model(self, filepath):
        """Save model with metadata."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat(),
            'model_architecture': str(self.model)
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model with metadata."""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        logger.info(f"Model loaded from {filepath}")
        return checkpoint
    
    def plot_training_progress(self):
        """Plot training progress."""
        if not self.training_history['loss']:
            logger.warning("No training history to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss plot
        ax1.plot(self.training_history['episode'], self.training_history['loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.training_history['episode'], self.training_history['accuracy'])
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.model_dir}/training_progress.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Training progress plots saved and displayed")

# Demo usage
if __name__ == "__main__":
    # Initialize training system
    training_system = EnhancedTrainingSystem()
    
    # Note: This is a simplified demo. In real implementation:
    # 1. Connect to enhanced_environment.py for rich simulation
    # 2. Implement GUI for expert data collection
    # 3. Add comprehensive evaluation metrics
    # 4. Implement model comparison and benchmarking
    
    logger.info("Enhanced Railway Junction RL Agent initialized")
    logger.info("Ready for training with curriculum learning and comprehensive evaluation")