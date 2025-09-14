import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from car_env import Environment
import pickle
import os

# --- Configuration ---
MODEL_FILENAME = "imitation_learning_model.pth"
DATA_FILENAME = "human_driving_data.pkl"

# --- Model Definition ---
# A simple neural network to map state -> action
class DrivingModel(nn.Module):
    def __init__(self):
        super(DrivingModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128), # State size is 4
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)    # Action size is 3 (Left, Straight, Right)
        )
    def forward(self, x):
        return self.network(x)

def collect_human_data(num_episodes):
    print("--- Phase 1: Human Data Collection ---")
    print(f"Please drive for {num_episodes} episodes. Use LEFT/RIGHT arrow keys.")
    print("Close the window to stop early.")
    
    env = Environment()
    dataset = []
    
    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        while not done:
            action = env.get_human_action()
            if action == -1: # Quit signal
                print("Data collection stopped early.")
                env.close()
                return dataset
            
            next_state, done = env.step(action)
            dataset.append((state, action))
            state = next_state
            
            env.render([f"Recording Episode: {i_episode}/{num_episodes}",
                        f"Data points: {len(dataset)}"])
    
    env.close()
    print(f"\nFinished data collection. Total samples: {len(dataset)}")
    
    with open(DATA_FILENAME, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {DATA_FILENAME}")
    return dataset

def train_from_data():
    print("\n--- Phase 2: Model Training ---")
    if not os.path.exists(DATA_FILENAME):
        print("Data file not found! Please collect data first.")
        return

    with open(DATA_FILENAME, 'rb') as f:
        dataset = pickle.load(f)

    states = torch.FloatTensor(np.array([d[0] for d in dataset]))
    actions = torch.LongTensor(np.array([d[1] for d in dataset]))

    train_dataset = TensorDataset(states, actions)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = DrivingModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_states, batch_actions in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_actions)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), MODEL_FILENAME)
    print(f"Training complete. Model saved to {MODEL_FILENAME}")

def test_agent_with_gui(num_episodes):
    print("\n--- Phase 3: Agent Testing ---")
    if not os.path.exists(MODEL_FILENAME):
        print("Model file not found! Please train the model first.")
        return
        
    model = DrivingModel()
    model.load_state_dict(torch.load(MODEL_FILENAME))
    model.eval()
    
    env = Environment()
    total_steps = 0
    
    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_logits = model(state_tensor)
                action = torch.argmax(action_logits, dim=1).item()

            state, done = env.step(action)
            steps += 1
            
            if env.get_human_action() == -1: # Allow quitting
                env.close()
                return

            env.render([f"Testing Agent - Episode {i_episode}",
                        f"Steps Survived: {steps}"])
        
        total_steps += steps
        print(f"Test Episode {i_episode}: Survived for {steps} steps.")
        
    env.close()
    print(f"\nAverage survival time: {total_steps / num_episodes:.2f} steps.")

if __name__ == "__main__":
    # Phase 1: Record 10 episodes of human driving
    collect_human_data(num_episodes=10)
    
    # Phase 2: Train the model on the recorded data
    train_from_data()
    
    # Phase 3: Test the trained agent on new tracks
    test_agent_with_gui(num_episodes=10)
