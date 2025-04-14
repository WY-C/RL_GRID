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
        x = torch.relu(self.fc1( x))
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
        self.epsilon_decay = 0.978

        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        
        self.learning_rate = 0.01
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(batch_size=64, buffer_size=10000)
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def choose_action(self, state, reward_pos, wall_pos):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
    
        #랜덤벽에서의 수정사항

        concat = state + reward_pos + wall_pos
        concat = torch.FloatTensor(concat).unsqueeze(0).to(device)  # (1, state_size + reward_pos_size)
        with torch.no_grad():
            q_values = self.model(concat)
    
        return torch.argmax(q_values, dim=1).item()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def update(self):
        if len(self.replay_buffer) < 64:
            return
        state, next_state, action, reward, reward_pos, wall_pos, done = self.replay_buffer.get_batch()
        state = state.to(device)
        next_state = next_state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        reward_pos = reward_pos.to(device)
        wall_pos = wall_pos.to(device)
        done = done.to(device)

        #reward_pos_batch = reward_pos.expand(state.shape[0], -1)  # (64, 2)
        concat = torch.cat((state, reward_pos, wall_pos), dim=1)
        concat_next = torch.cat((next_state, reward_pos, wall_pos), dim=1)
        qs = self.model(concat).gather(1, action.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            next_qs = self.target_model(concat_next).max(dim=1)[0]
            target = reward + (1 - done) * self.gamma * next_qs
        
        loss = self.criterion(qs, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
class ReplayBuffer:
    def __init__(self, batch_size, buffer_size = 10000):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, next_state, action, reward, reward_pos, wall_pos, done):
        data = (state, next_state, action, reward, reward_pos, wall_pos, done)
        self.buffer.append(data)


    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.array([x[0] for x in data])
        next_state = np.array([x[1] for x in data])
        action = np.array([x[2] for x in data])
        reward = np.array([x[3] for x in data])
        reward_pos = np.array([x[4] for x in data])
        wall_pos = np.array([x[5] for x in data])
        done = np.array([x[6] for x in data])
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        reward_pos = torch.FloatTensor(reward_pos)
        wall_pos = torch.FloatTensor(wall_pos)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done)
        return state, next_state, action, reward, reward_pos, wall_pos, done

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
            action = agent.choose_action(env.agent_pos, env.reward_pos, env.wall_pos)
            reward, done, old_pos = env.step(action, episode_ticks)

            agent.replay_buffer.add(old_pos, env.agent_pos, action, reward, env.reward_pos, env.wall_pos, done)
            agent.update()
            

        total_ticks += episode_ticks
        total_reward += reward
        
        if episode == 5000:
            agent.save_model("123.pth")
        if episode % printing == 0:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        if episode % 500 == 0:
            visualize_qvalues(env, agent)
        
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


def visualize_qvalues(env, agent):
    all_states = []  # 가능한 모든 state를 저장
    qvalues = []     # 모든 Q값을 저장
    max_q = []
    # 가능한 모든 상태를 반복 (예: gridworld 환경 기준)
    for x in range(5):  # 환경의 x축 크기
        for y in range(5):  # 환경의 y축 크기
            # 상태 및 보상 위치 저장
            state = [x, y]
            reward_pos = env.reward_pos
            wall_pos = env.wall_pos
            # 상태와 보상 위치를 모델에 입력하여 Q값 계산
            concat = state + reward_pos + wall_pos
            concat_tensor = torch.FloatTensor(concat).unsqueeze(0).to(device)
            with torch.no_grad():
                q_value = agent.model(concat_tensor)

            # 모든 상태와 Q값 저장
            all_states.append(state)
            qvalues.append(q_value.cpu().numpy())  # GPU에서 CPU로 이동하여 numpy 변환
            max_q_value = np.argmax(q_value.cpu().numpy())

            if max_q_value == 0:
                max_q.append("up")
            elif max_q_value == 1:
                max_q.append("down")
            elif max_q_value == 2:
                max_q.append("left")
            elif max_q_value == 3:
                max_q.append("right")
    # 각 상태와 Q값 출력
    for i, state in enumerate(all_states):
        arr = np.array(qvalues[i])
        np.set_printoptions(precision=2, suppress=True)
        print(f"State: {state}, Reward Position: {env.reward_pos}, Wall Position: {env.wall_pos}, Q-Values: {arr}, Max Q-Value: {max_q[i]}")