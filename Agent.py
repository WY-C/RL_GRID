import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class DQNAgent:
    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95

        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        
        self.learning_rate = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(batch_size=64, buffer_size=10000)
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values, dim=1).item()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def update(self):
        if len(self.replay_buffer) < 64:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.get_batch()
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        qs = self.model(states).gather(1, actions.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            next_qs = self.target_model(next_states).max(dim=1)[0]
            target = rewards + (1 - dones) * self.gamma * next_qs
        
        loss = self.criterion(qs, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
class ReplayBuffer:
    def __init__(self, batch_size, buffer_size = 10000):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)


    def get_batch(self):
        experience = random.sample(self.buffer, self.batch_size)
        
        state = np.array([x[0] for x in experience])
        action = np.array([x[1] for x in experience])
        reward = np.array([x[2] for x in experience])
        next_state = np.array([x[3] for x in experience])
        done = np.array([x[4] for x in experience])

        state = torch.FloatTensor(state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)