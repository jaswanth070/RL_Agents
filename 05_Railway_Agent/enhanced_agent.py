# enhanced_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import os
import time
from datetime import datetime
import logging
from enhanced_environment import EnhancedEnvironment, ScenarioGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedJunctionModel(nn.Module):
    """Enhanced neural network model for railway junction control."""
    
    def __init__(self, input_size=14, hidden_sizes=[128, 64, 32, 16]):
        super(EnhancedJunctionModel, self).__init__()
        
        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        
        # Dropout layers
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        
        # Output layers
        self.priority_head = nn.Linear(hidden_sizes[-1], 2)  # Priority scores for [UP, DOWN]
        self.confidence_head = nn.Linear(hidden_sizes[-1], 1)  # Decision confidence
        
    def forward(self, x):
        # Input layer with dropout
        x = F.relu(self.input_layer(x))
        x = self.dropout1(x)
        
        # First hidden layer with dropout
        x = F.relu(self.hidden_layers[0](x))
        x = self.dropout2(x)
        
        # Remaining hidden layers
        for layer in self.hidden_layers[1:]:
            x = F.relu(layer(x))
        
        # Output heads
        priorities = torch.softmax(self.priority_head(x), dim=-1)
        confidence = torch.sigmoid(self.confidence_head(x))
        
        return priorities, confidence

class EnhancedAgent:
    """Enhanced RL agent with curriculum learning and comprehensive training."""
    
    def __init__(self, model_dir="models", data_dir="data"):
        self.model_dir = model_dir
        self.data_dir = data_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        self.model = EnhancedJunctionModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.8)
        
        # Training history
        self.training_history = {
            'loss': [], 'accuracy': [], 'episode': [], 'phase': []
        }
        
        # ## MODIFIED ##: Default curriculum phases. These can now be overridden by user input.
        self.default_curriculum_phases = [
            {
                'name': 'Foundation',
                'episodes': 30, # Reduced for quicker demos
                'scenarios': ['basic_priority', 'speed_differential'],
                'target_accuracy': 0.85
            },
            {
                'name': 'Complexity', 
                'episodes': 25,
                'scenarios': ['emergency', 'passenger_freight', 'maintenance_window'],
                'target_accuracy': 0.75
            },
            {
                'name': 'Multi-factor',
                'episodes': 20,
                'scenarios': ['weather_impact', 'night_ops'],
                'target_accuracy': 0.70
            },
            {
                'name': 'Expert',
                'episodes': 15,
                'scenarios': ['peak_hour'],
                'target_accuracy': 0.60
            }
        ]
    
    ## MODIFIED ##: Accepts a `curriculum_phases` config to allow for customization.
    def collect_training_data(self, curriculum_phases: list, gui_enabled=True):
        """Collect training data through human demonstrations based on a curriculum."""
        logger.info("Starting enhanced data collection with curriculum learning")
        
        env = EnhancedEnvironment(gui_enabled=gui_enabled)
        scenario_gen = ScenarioGenerator()
        all_data = []
        total_episodes_to_run = sum(p['episodes'] for p in curriculum_phases)
        
        total_episodes_completed = 0
        for phase_idx, phase in enumerate(curriculum_phases):
            logger.info(f"Phase {phase_idx + 1}: {phase['name']} - {phase['episodes']} episodes")
            phase_data = []
            
            for episode in range(phase['episodes']):
                total_episodes_completed += 1
                
                # Generate scenario for current phase
                scenario_type = np.random.choice(phase['scenarios'])
                scenario_config = scenario_gen.generate_scenario(scenario_type, phase_idx + 1)
                
                # Reset environment with scenario
                state = env.reset(scenario_config)
                done = False
                step_count = 0
                decisions_made = 0
                
                print(f"\n=== Phase '{phase['name']}' - Episode {episode + 1}/{phase['episodes']} ===")
                print(f"Scenario: {scenario_type}")
                print(f"Total Progress: {total_episodes_completed}/{total_episodes_to_run}")
                
                while not done and step_count < 500:  # Max steps per episode
                    # Render environment
                    info_lines = [
                        f"Phase: {phase['name']} ({episode + 1}/{phase['episodes']})",
                        f"Scenario: {scenario_type}",
                        f"Total Progress: {total_episodes_completed}/{total_episodes_to_run}",
                        f"Total Decisions Collected: {len(all_data)}"
                    ]
                    
                    if gui_enabled:
                        env.render(info_lines)
                    
                    # Get human decision if conflict detected
                    action = None
                    if env.conflict_detected and not env.decision_made:
                        if gui_enabled:
                            # The decision overlay is now handled by env.render()
                            action = env.get_human_action()
                            if action == -1:  # User wants to quit
                                env.close()
                                return all_data
                        else:
                            # Auto-generate expert decisions for non-GUI mode
                            action = self._generate_expert_decision(env)
                        
                        if action is not None and action >= 0:
                            # Record the decision
                            decision_data = {
                                'state': state.copy(),
                                'action': action,
                                'scenario_type': scenario_type,
                                'phase': phase['name'],
                                'confidence': self._estimate_decision_quality(env, action),
                                'timestamp': time.time()
                            }
                            
                            phase_data.append(decision_data)
                            all_data.append(decision_data)
                            decisions_made += 1
                            
                            if not gui_enabled:
                                print(f"Auto-decision recorded: {'UP' if action == 0 else 'DOWN'} priority")
                    
                    # Step environment
                    state, reward, done, info = env.step(action)
                    step_count += 1
                
                if decisions_made == 0:
                    print("Episode completed with no conflicts. This episode will be repeated.")
                    episode -= 1  # Repeat this episode to ensure data is collected
            
            # Save phase data
            phase_filename = f"{self.data_dir}/phase_{phase_idx + 1}_{phase['name'].lower()}_data.pkl"
            with open(phase_filename, 'wb') as f:
                pickle.dump(phase_data, f)
            
            logger.info(f"Phase {phase['name']} completed. {len(phase_data)} decisions collected.")
        
        env.close()
        
        # Save complete dataset
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_filename = f"{self.data_dir}/complete_training_data_{timestamp}.pkl"
        with open(dataset_filename, 'wb') as f:
            pickle.dump(all_data, f)
        
        logger.info(f"Data collection complete! Total decisions: {len(all_data)}")
        logger.info(f"Dataset saved to: {dataset_filename}")
        
        return all_data
    
    def _generate_expert_decision(self, env):
        """Generate expert decision based on priorities and rules."""
        up_priority = env.up_train.get_priority_score(env.weather, env.maintenance_active)
        down_priority = env.down_train.get_priority_score(env.weather, env.maintenance_active)
        
        # Emergency override
        if env.up_train.emergency_flag and not env.down_train.emergency_flag:
            return 0
        elif env.down_train.emergency_flag and not env.up_train.emergency_flag:
            return 1
        
        # Priority-based decision with some randomness for diversity
        if abs(up_priority - down_priority) < 0.1:  # Very close priorities
            return np.random.choice([0, 1])  # Random choice
        else:
            return 0 if up_priority > down_priority else 1
    
    def _estimate_decision_quality(self, env, action):
        """Estimate the quality/confidence of a decision."""
        up_priority = env.up_train.get_priority_score(env.weather, env.maintenance_active)
        down_priority = env.down_train.get_priority_score(env.weather, env.maintenance_active)
        
        priority_diff = abs(up_priority - down_priority)
        
        # High confidence for clear priority differences
        if priority_diff > 0.3:
            return 0.9
        elif priority_diff > 0.1:
            return 0.7
        else:
            return 0.5  # Low confidence for close priorities
    
    def train_model(self, training_data=None, epochs=100, validation_split=0.2):
        """Train the model using collected data."""
        
        # Load training data if not provided
        if training_data is None:
            data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pkl')]
            if not data_files:
                logger.error("No training data found! Please collect data first.")
                return
            
            # Load the most recent dataset
            latest_file = max(data_files, key=lambda f: os.path.getctime(os.path.join(self.data_dir, f)))
            with open(os.path.join(self.data_dir, latest_file), 'rb') as f:
                training_data = pickle.load(f)
            
            logger.info(f"Loaded training data from {latest_file}: {len(training_data)} samples")
        
        if len(training_data) < 10:
            logger.error("Insufficient training data! Need at least 10 samples.")
            return
        
        # Prepare data
        states = np.array([d['state'] for d in training_data])
        actions = np.array([d['action'] for d in training_data])
        confidences = np.array([d.get('confidence', 0.5) for d in training_data])
        
        # Split data
        n_samples = len(states)
        n_train = int(n_samples * (1 - validation_split))
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        
        # Create tensors
        train_states = torch.FloatTensor(states[train_idx])
        train_actions = torch.LongTensor(actions[train_idx])
        train_confidences = torch.FloatTensor(confidences[train_idx])
        
        val_states = torch.FloatTensor(states[val_idx])
        val_actions = torch.LongTensor(actions[val_idx])
        val_confidences = torch.FloatTensor(confidences[val_idx])
        
        # Create data loader
        train_dataset = TensorDataset(train_states, train_actions, train_confidences)
        train_loader = DataLoader(train_dataset, batch_size=min(32, len(train_dataset)), shuffle=True)
        
        logger.info(f"Training on {n_train} samples, validating on {len(val_idx)} samples")
        
        # Training loop
        self.model.train()
        best_val_accuracy = 0.0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            n_batches = 0
            
            for batch_states, batch_actions, batch_confidences in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                priorities, pred_confidences = self.model(batch_states)
                
                # Losses
                priority_loss = F.cross_entropy(priorities, batch_actions)
                confidence_loss = F.mse_loss(pred_confidences.squeeze(), batch_confidences)
                
                # Combined loss
                total_loss = 0.8 * priority_loss + 0.2 * confidence_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Metrics
                epoch_loss += total_loss.item()
                predictions = torch.argmax(priorities, dim=1)
                epoch_accuracy += (predictions == batch_actions).float().mean().item()
                n_batches += 1
            
            # Validation
            val_loss, val_accuracy = self._evaluate_model(val_states, val_actions, val_confidences)
            
            # Update learning rate
            self.scheduler.step()
            
            # Record history
            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
            avg_accuracy = epoch_accuracy / n_batches if n_batches > 0 else 0
            
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(avg_accuracy)
            self.training_history['episode'].append(epoch)
            
            # Logging
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch + 1}/{epochs}: "
                           f"Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}, "
                           f"Val_Loss={val_loss:.4f}, Val_Acc={val_accuracy:.4f}")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model(f"{self.model_dir}/best_model.pth")
        
        # Save final model
        self.save_model(f"{self.model_dir}/final_model.pth")
        logger.info(f"Training completed! Best validation accuracy: {best_val_accuracy:.4f}")
        
        return self.training_history
    
    def _evaluate_model(self, states, actions, confidences):
        """Evaluate model on validation data."""
        self.model.eval()
        with torch.no_grad():
            priorities, pred_confidences = self.model(states)
            
            priority_loss = F.cross_entropy(priorities, actions)
            confidence_loss = F.mse_loss(pred_confidences.squeeze(), confidences)
            total_loss = 0.8 * priority_loss + 0.2 * confidence_loss
            
            predictions = torch.argmax(priorities, dim=1)
            accuracy = (predictions == actions).float().mean().item()
            
        self.model.train()
        return total_loss.item(), accuracy
    
    def test_agent(self, num_episodes=5, gui_enabled=True):
        """Test the trained agent."""
        if not os.path.exists(f"{self.model_dir}/best_model.pth"):
            logger.error("No trained model found! Please train the model first.")
            return
        
        # Load best model
        self.load_model(f"{self.model_dir}/best_model.pth")
        logger.info("Testing trained agent...")
        
        env = EnhancedEnvironment(gui_enabled=gui_enabled)
        scenario_gen = ScenarioGenerator()
        
        total_reward = 0
        total_decisions = 0
        correct_decisions = 0
        
        for episode in range(num_episodes):
            # Generate random scenario
            scenario_config = scenario_gen.generate_scenario(difficulty_level=np.random.randint(1, 5))
            state = env.reset(scenario_config)
            
            done = False
            step_count = 0
            episode_reward = 0
            
            print(f"\n=== Testing Episode {episode + 1}/{num_episodes} ===")
            print(f"Scenario: {scenario_config['scenario_type']}")
            
            while not done and step_count < 500:
                info_lines = [
                    f"AI Testing - Episode {episode + 1}/{num_episodes}",
                    f"Scenario: {scenario_config['scenario_type']}",
                    f"Accuracy: {correct_decisions/max(total_decisions, 1)*100:.1f}% ({correct_decisions}/{total_decisions})"
                ]
                
                # AI decision making
                action = None
                decision_info_for_render = None ## NEW
                if env.conflict_detected and not env.decision_made:
                    # Get AI prediction
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    self.model.eval()
                    with torch.no_grad():
                        priorities, confidence = self.model(state_tensor)
                        action = torch.argmax(priorities, dim=1).item()
                        pred_confidence = confidence.item()
                    
                    # Get expert decision for comparison
                    expert_action = self._generate_expert_decision(env)
                    
                    total_decisions += 1
                    if action == expert_action:
                        correct_decisions += 1
                    
                    print(f"AI Decision: {'UP' if action == 0 else 'DOWN'} (confidence: {pred_confidence:.3f})")
                    print(f"Expert would choose: {'UP' if expert_action == 0 else 'DOWN'}")
                    print(f"Match: {'✓' if action == expert_action else '✗'}")

                    ## NEW ##: Prepare decision info to be rendered on screen
                    decision_info_for_render = {
                        'ai_action': action,
                        'confidence': pred_confidence,
                        'expert_action': expert_action
                    }
                
                ## MODIFIED ##: Pass decision info to the render method
                if gui_enabled:
                    env.render(info_lines, decision_info=decision_info_for_render)
                
                # Step environment
                state, reward, done, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                if gui_enabled and action is not None:
                    time.sleep(2)  # Pause to see decision
            
            total_reward += episode_reward
            print(f"Episode {episode + 1} completed. Reward: {episode_reward:.3f}")
            if gui_enabled:
                time.sleep(1) # Pause between episodes
        
        env.close()
        
        # Results summary
        avg_reward = total_reward / num_episodes
        accuracy = correct_decisions / max(total_decisions, 1) * 100
        
        logger.info(f"Testing completed!")
        logger.info(f"Average reward: {avg_reward:.3f}")
        logger.info(f"Decision accuracy: {accuracy:.1f}% ({correct_decisions}/{total_decisions})")
        
        return {
            'average_reward': avg_reward,
            'accuracy': accuracy,
            'total_decisions': total_decisions,
            'correct_decisions': correct_decisions
        }
    
    def save_model(self, filepath):
        """Save model with metadata."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'input_size': 14,
                'hidden_sizes': [128, 64, 32, 16],
                'total_parameters': sum(p.numel() for p in self.model.parameters())
            }
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model with metadata."""
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
            logger.info(f"Model loaded from {filepath}")
            return checkpoint
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {e}")
            return None

## NEW ##: Helper function to get user configuration for training episodes
def get_episode_config(default_phases):
    """Prompts the user to configure the number of episodes for each training phase."""
    print("\n--- Configure Training Episodes ---")
    print("Press ENTER to use the default value for each phase.")
    
    custom_phases = []
    for phase in default_phases:
        while True:
            try:
                user_input = input(f"Enter episodes for '{phase['name']}' phase (default: {phase['episodes']}): ")
                if user_input == "":
                    num_episodes = phase['episodes']
                else:
                    num_episodes = int(user_input)
                
                if num_episodes < 0:
                    raise ValueError("Number of episodes cannot be negative.")
                
                # Create a copy of the phase and update episodes
                new_phase = phase.copy()
                new_phase['episodes'] = num_episodes
                custom_phases.append(new_phase)
                break
            except ValueError as e:
                print(f"Invalid input. Please enter a positive whole number. Error: {e}")
    
    print("\nCustom training schedule set:")
    for phase in custom_phases:
        print(f"  - {phase['name']}: {phase['episodes']} episodes")
    print("-" * 30)
    
    return custom_phases

## MODIFIED ##: Refactored the main loop for better user experience
def main():
    """Main training and testing pipeline with an interactive menu."""
    print("=" * 50)
    print("   Enhanced Railway Junction RL Agent")
    print("=" * 50)
    
    agent = EnhancedAgent()
    
    while True:
        print("\nMain Menu:")
        print("1. Collect training data (with GUI)")
        print("2. Collect training data (auto-generate for testing)")
        print("3. Train model from latest data")
        print("4. Test trained agent (with GUI)")
        print("5. Run environment demo only")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice in ['1', '2']:
            is_gui = (choice == '1')
            mode = "GUI" if is_gui else "Auto-generation"
            print(f"\nStarting data collection via {mode}...")
            
            # Get custom episode configuration from user
            episode_config = get_episode_config(agent.default_curriculum_phases)
            
            # Check if there are any episodes to run
            if sum(p['episodes'] for p in episode_config) == 0:
                print("No episodes configured to run. Returning to menu.")
                continue

            agent.collect_training_data(curriculum_phases=episode_config, gui_enabled=is_gui)
        
        elif choice == '3':
            print("\nStarting model training...")
            try:
                epochs = int(input("Enter number of training epochs (e.g., 50): "))
                if epochs <= 0: raise ValueError
            except ValueError:
                print("Invalid number. Using default of 50 epochs.")
                epochs = 50
            
            history = agent.train_model(epochs=epochs)
            if history:
                print("Training completed successfully!")
        
        elif choice == '4':
            print("\nTesting trained agent...")
            try:
                episodes = int(input("Enter number of testing episodes (e.g., 5): "))
                if episodes <= 0: raise ValueError
            except ValueError:
                print("Invalid number. Using default of 5 episodes.")
                episodes = 5

            results = agent.test_agent(num_episodes=episodes, gui_enabled=True)
            if results:
                print(f"Testing completed! Accuracy: {results['accuracy']:.1f}%")
        
        elif choice == '5':
            print("\nRunning environment demo...")
            from enhanced_environment import run_demo
            run_demo()
        
        elif choice == '6':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()