import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from car_env import Environment
import os

# --- Configuration ---
MODEL_FILENAME = "policy_gradient_curved_track_model.pth"

# --- Policy Network Definition ---
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        # State size is now 4, Action size is 3
        self.network = nn.Sequential(
            nn.Linear(4, 128), # MODIFIED: Input size is now 4
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)

# --- Agent Definition ---
class Agent:
    def __init__(self, learning_rate=0.001, gamma=0.99):
        self.policy_network = PolicyNetwork()
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_network(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def update_policy(self):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(self.rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
            
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_loss = []
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []
        return policy_loss.item()
    
    def save_model(self, filename):
        torch.save(self.policy_network.state_dict(), filename)
        print(f"\nModel saved to {filename}")

    def load_model(self, filename):
        self.policy_network.load_state_dict(torch.load(filename))
        self.policy_network.eval() # Set the network to evaluation mode
        print(f"Model loaded from {filename}")

def train_headless(agent, num_episodes):
    print("--- Starting Headless Training on Curved Track ---")
    env = Environment(headless=True)
    all_rewards = []
    all_losses = []

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        for t in range(1000): # Max steps per episode
            action = agent.select_action(state)
            state, reward, done = env.step(action)
            agent.rewards.append(reward)
            episode_reward += reward
            if done: break
        
        loss = agent.update_policy()
        all_rewards.append(episode_reward)
        all_losses.append(loss)
        
        if i_episode % 100 == 0:
            avg_reward = np.mean(all_rewards[-100:])
            avg_loss = np.mean(all_losses[-100:])
            print(f"Episode {i_episode}\tAvg Reward: {avg_reward:.2f}\tAvg Loss: {avg_loss:.2f}")
    
    agent.save_model(MODEL_FILENAME)
    print("--- Headless Training Finished ---")

def test_with_gui(agent, num_episodes):
    print("\n--- Starting GUI Testing on Curved Track ---")
    if not os.path.exists(MODEL_FILENAME):
        print("Model file not found! Please train the agent first.")
        return

    agent.load_model(MODEL_FILENAME)
    env = Environment(headless=False)
    all_rewards = []

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        for t in range(1000):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad(): # No need to calculate gradients during testing
                action_probs = agent.policy_network(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample().item()
            
            state, reward, done = env.step(action)
            episode_reward += reward
            env.render(episode=i_episode, total_reward=episode_reward)
            
            # Allow quitting during testing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            
            if done: break
        
        all_rewards.append(episode_reward)
        print(f"Test Episode {i_episode}\tScore: {episode_reward:.2f}")

    avg_score = np.mean(all_rewards)
    print(f"\n--- Testing Finished ---")
    print(f"Average score over {num_episodes} test episodes: {avg_score:.2f}")
    env.close()

if __name__ == "__main__":
    agent = Agent()
    
    # Phase 1: Train the model for 5000 episodes without graphics
    train_headless(agent, num_episodes=900)
    
    # Phase 2: Test the trained model for 10 episodes with graphics
    test_with_gui(agent, num_episodes=10)

