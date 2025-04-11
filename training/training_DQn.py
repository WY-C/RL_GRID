import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

x = []
y = []

device = torch.device("cuda")
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
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99

        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        
        self.learning_rate = 0.01
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()




        self.replay_buffer = ReplayBuffer(batch_size=64, buffer_size=10000)
        
        
    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
    
        # state를 torch 텐서로 변환하고 GPU로 이동
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # (1, state_size)

        with torch.no_grad():
            q_values = self.model(state)
    
        return torch.argmax(q_values, dim=1).item()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    #폐기
    def visualize(self, grid_size):
        actions = ['↑', '↓', '←', '→']
        fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12))

        for i in range(grid_size):
            for j in range(grid_size):
                # 상태 정규화
                state = np.array([i / (grid_size - 1), j / (grid_size - 1)], dtype=np.float32)
                state = torch.FloatTensor(state).unsqueeze(0).to(device)

                with torch.no_grad():
                    q_values = self.model(state).cpu().numpy().flatten()

                max_action = np.argmax(q_values)
                ax[i, j].imshow(np.zeros((1, 1)), cmap='gray', vmin=0, vmax=1)
                ax[i, j].set_title(f"{actions[max_action]}\n{q_values[max_action]:.2f}", fontsize=10)
                ax[i, j].axis('off')

        plt.tight_layout()
        plt.show()


    def update(self):
        if len(self.replay_buffer) < 64:
            return
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        done = done.to(device)
        qs = self.model(state).gather(1, action.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            next_qs = self.target_model(next_state).max(dim=1)[0]
            target = reward + (1 - done) * self.gamma * next_qs
        
        loss = self.criterion(qs, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
class ReplayBuffer:
    def __init__(self, batch_size, buffer_size = 10000):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)


    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.array([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.array([x[3] for x in data])
        done = np.array([x[4] for x in data])
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

def solo_play(env, agent, episodes, test = False):
    if test == True:
        printing = 10
    else:
        #agent.eval()
        printing = 100
    
    total_reward = 0
    total_ticks = 0
    for episode in range(episodes):
        env.reset()
        done = False
        episode_ticks = 0  # 에피소드 틱 수
        while not done:
            episode_ticks += 1  # 한 틱 증가
            action = agent.choose_action(env.agent_pos)
            reward, done, old_pos = env.step(env.agent_pos, action, episode_ticks)

            agent.replay_buffer.add(old_pos, action, reward, env.agent_pos, done)
            agent.update()
            

        total_ticks += episode_ticks
        total_reward += reward
        
        if episode % printing == 0:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon - 0.05)
        
        if episode % (printing * 5) == 0:
            agent.update_target_model()
        
        if episode % printing == 0:
            print(f"Episode {episode}/{episodes}, Ticks: {episode_ticks}, Total Ticks: {total_ticks/printing}, Total Reward: {total_reward/printing:.2f}, Epsilon: {agent.epsilon:.2f}")
            x.append(episode)
            y.append(total_reward / printing)            
            episode_ticks = 0
            total_reward = 0
            total_ticks = 0

    return x, y, agent.model.state_dict()


 