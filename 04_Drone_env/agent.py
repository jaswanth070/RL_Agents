# imitation_drone_agent.py
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle, os, random
from drone_env import Environment

MODEL_FILENAME = "drone_model.pth"
DATA_FILENAME = "human_drone_data.pkl"

class DroneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),  # state: dist_x, gap_center, drone_y
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # Up, Stay, Down
        )
    def forward(self, x):
        return self.net(x)

def collect_human_data(num_episodes=3):   # fewer episodes needed now
    env = Environment()
    dataset = []
    for ep in range(1, num_episodes+1):
        state = env.reset()
        done = False
        while not done:
            action = env.get_human_action()
            if action == -1:
                env.close()
                return dataset
            next_state, done = env.step(action)
            dataset.append((state, action))
            state = next_state
            env.render([f"Recording {ep}/{num_episodes}", f"Samples: {len(dataset)}"])
    env.close()
    with open(DATA_FILENAME, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved {len(dataset)} samples to {DATA_FILENAME}")
    return dataset

def train_from_data():
    if not os.path.exists(DATA_FILENAME):
        print("No dataset found.")
        return
    with open(DATA_FILENAME, "rb") as f:
        dataset = pickle.load(f)

    # --- Balance dataset ---
    actions = [d[1] for d in dataset]
    min_count = min([actions.count(a) for a in set(actions)])
    balanced = []
    for a in set(actions):
        samples = [d for d in dataset if d[1] == a]
        balanced.extend(random.sample(samples, min_count))
    dataset = balanced

    states = torch.FloatTensor([d[0] for d in dataset])
    actions = torch.LongTensor([d[1] for d in dataset])
    loader = DataLoader(TensorDataset(states, actions), batch_size=32, shuffle=True)

    model = DroneModel()
    opt = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(15):
        loss_sum = 0
        for s, a in loader:
            opt.zero_grad()
            loss = criterion(model(s), a)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        print(f"Epoch {epoch+1}: Loss {loss_sum/len(loader):.4f}")
    torch.save(model.state_dict(), MODEL_FILENAME)
    print("Model saved.")
    return model

def test_agent(num_episodes=5):
    if not os.path.exists(MODEL_FILENAME):
        print("No model trained.")
        return
    model = DroneModel()
    model.load_state_dict(torch.load(MODEL_FILENAME))
    model.eval()
    env = Environment()
    for ep in range(1, num_episodes+1):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            st = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(model(st), dim=1).item()
            state, done = env.step(action)
            steps += 1
            # only allow quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            env.render([f"AI Testing Episode {ep}/{num_episodes}", f"Steps: {steps}"])
        print(f"Episode {ep}: survived {steps} steps.")
    env.close()


if __name__ == "__main__":
    collect_human_data(10)
    train_from_data()
    test_agent(5)