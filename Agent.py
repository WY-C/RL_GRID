import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

state_size = 8
action_size = 4
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

class PER:
    def __init__(self, batch_size, buffer_size):
        self.batch_size = batch_size
        self.buffer = []
        self.priority = []
        self.buffer_size = buffer_size
        self.epsilon = 1e-5
        self.alpha = 0.6
        self.beta = 0.4

    def add(self, state, action, reward, next_state, done, td_error):
        priority = abs(td_error) + self.epsilon
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            self.priority.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
        self.priority.append(priority)

    def get_batch(self):
        #print(self.priority)
        priorities = np.array(self.priority)
        probs = priorities / np.sum(priorities) #alpha = 1

        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        experience = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # normalize

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
        weights = torch.FloatTensor(weights)
        return state, action, reward, next_state, done, weights, indices
        
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):   
            self.priority[idx] = (abs(td_error.item()) + self.epsilon) ** self.alpha


    def __len__(self):
        return len(self.buffer)

    
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
    def __init__(self, batch_size, buffer_size):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done, td_error):
        experience = (state, action, reward, next_state, done, td_error)
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


class DQNAgent:

    model = DQN(state_size, action_size).to(device)
    target_model = DQN(state_size, action_size).to(device)
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    replay_buffer = PER(batch_size=64, buffer_size=10000)

    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.95

    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size


      
    def save_model(self, path):
        torch.save(DQNAgent.model.state_dict(), path)
        
    def choose_action(self, state):
        if np.random.rand() <= DQNAgent.epsilon:
            return np.random.choice(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = DQNAgent.model(state)

        return torch.argmax(q_values, dim=1).item()
    
    def update_target_model(self):
        DQNAgent.target_model.load_state_dict(DQNAgent.model.state_dict())

    def update(self):
        if len(DQNAgent.replay_buffer) < 64:
            return
        
        states, actions, rewards, next_states, dones, weights, indices = DQNAgent.replay_buffer.get_batch()
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        weights = weights.to(device)
        qs = DQNAgent.model(states).gather(1, actions.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            next_qs = DQNAgent.target_model(next_states).max(dim=1)[0]
            target = rewards + (1 - dones) * DQNAgent.gamma * next_qs
        
        td_errors = target - qs
        loss = (td_errors.pow(2) * weights).mean()
        DQNAgent.optimizer.zero_grad()
               
        loss.backward()
        DQNAgent.optimizer.step()

        DQNAgent.replay_buffer.update_priorities(indices, td_errors)
        
