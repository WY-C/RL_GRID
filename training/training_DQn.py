import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_num, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_num, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x




# Self-Play Agent
class SelfPlayAgent:
    def __init__(self, env, state_num, action_size, agent_id):
        self.env = env
        self.state_num = state_num
        self.action_size = action_size
        self.lr = 0.01
        self.agent_id = agent_id  # 에이전트 ID

        self.model = DQN(state_num, action_size)  # DQN 모델
        self.target_model = DQN(state_num, action_size)  # 타겟 모델
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()  # 손실 함수
        self.q_table = np.zeros((env.grid_size, env.grid_size, 4))  # Q-테이블
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9  # 감쇠율
        self.alpha = 0.1  # 학습률
        self.epsilon = 0.9  # 탐험 확률

    def choose_action(self, position):
        # ε-그리디 정책
        if random.uniform(0, 1) < self.epsilon:  # 탐험
            return random.randint(0, 3)
        else:  # 활용
            qs = self.model(torch.FloatTensor(self.q_table[position[0], position[1]]))
            return torch.argmax(qs).item()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def update_q_value(self, old_pos, action, reward, new_pos):
        old_q = self.q_table[old_pos[0], old_pos[1], action]
        max_future_q = np.max(self.q_table[new_pos[0], new_pos[1]])
        self.q_table[old_pos[0], old_pos[1], action] = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)

    def store_experience(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train_from_memory(self):
        if len(self.memory) < 32:
            return 
        batch = random.sample(self.memory, 32)
        for state, action, reward, next_state in batch:
            self.update_q_value(state, action, reward, next_state)

def solo_play(env, agent1, episodes, test = False):
    if test == True:
        printing = 10
    else:
        printing = 1000
    
    total_ticks = 0  # 총 틱 수
    total_reward = 0
 
    for episode in range(episodes):
        env.reset()
        done = False
        episode_ticks = 0  # 에피소드 틱 수
        while not done:
            episode_ticks += 1  # 한 틱 증가

            # 에이전트1 행동
            action1 = agent1.choose_action(env.agent1_pos)
            old_pos1 = env.agent1_pos[:]
            env.move(env.agent1_pos, action1)
            if env.check_wall(env.agent1_pos):
                env.agent1_pos = old_pos1  # 벽에 부딪히면 원래 위치로 돌아감

            # 리워드 확인
            if env.check_reward(env.agent1_pos):
                reward1 = 1.5 - (episode_ticks / 100)
                if reward1 == 0:
                    reward1 = 1e-16  # 리워드가 0일 때 작은 값으로 설정
                #cnt_success += 1
                env.reward_pos = env.generate_reward_pos()
            else:
                reward1 = 0

            # Q-값 업데이트
            total_reward += reward1
            agent1.update_q_value(old_pos1, action1, reward1, env.agent1_pos)

            # 종료 조건
            if reward1 != 0: # or episode_ticks == 150:
                #total_reward += reward1
                total_ticks += episode_ticks
                if(episode != 0 and episode % printing == 0):
                    # 에피소드 틱 수를 총 틱 수에 추가
                    print("tick", total_ticks / printing)
                    print("reward", total_reward / printing)
                    #print("success", cnt_success / printing, "collision", cnt_collision / printing)
                    #x.append(episode)
                    #y.append(total_reward / printing)
                    total_ticks = 0  # 총 틱 수
                    total_reward = 0
                    agent1.epsilon *= 0.98  # 탐험 확률 감소
                done = True      
                
# Self-Play 실행
def self_play(env, agent1, agent2, episodes, test=False):
    if test == True:
        printing = 10
    else:
        printing = 1000
    
    total_ticks = 0  # 총 틱 수
    total_reward = 0

    cnt_success = 0  # 성공 횟수
    cnt_collision = 0  # 충돌 횟수

    for episode in range(episodes):
        env.reset()
        done = False
        episode_ticks = 0  # 에피소드 틱 수
        while not done:
            episode_ticks += 1  # 한 틱 증가

            # 에이전트1 행동
            action1 = agent1.choose_action(env.agent1_pos)
            old_pos1 = env.agent1_pos[:]
            env.move(env.agent1_pos, action1)

            # 에이전트2 행동
            action2 = agent2.choose_action(env.agent2_pos)
            old_pos2 = env.agent2_pos[:]
            env.move(env.agent2_pos, action2)

            # 리워드 및 충돌 확인
            if env.check_reward(env.agent1_pos) or env.check_reward(env.agent2_pos):
                reward1 = reward2 = 1.5 - (episode_ticks / 100)
                cnt_success += 1
                env.reward_pos = env.generate_reward_pos()
            elif env.check_collision():
                reward1 = reward2 = -1
                cnt_collision += 1
            else:
                reward1 = reward2 = 0

            # Q-값 업데이트
            total_reward += reward1
            agent1.update_q_value(old_pos1, action1, reward1, env.agent1_pos)
            agent2.update_q_value(old_pos2, action2, reward2, env.agent2_pos)

            # 종료 조건
            if reward1 != 0:
                #total_reward += reward1
                total_ticks += episode_ticks
                if(episode != 0 and episode % printing == 0):
                    # 에피소드 틱 수를 총 틱 수에 추가
                    print("tick", total_ticks / printing)
                    print("reward", total_reward / printing)
                    print("success", cnt_success / printing, "collision", cnt_collision / printing)
                    cnt_success = 0
                    cnt_collision = 0
                    total_ticks = 0  # 총 틱 수
                    total_reward = 0
                done = True

# Q-값 시각화
def visualize_q_values(agent, grid_size=5):
    actions = ['↑', '↓', '←', '→']
    fig, ax = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    for i in range(grid_size):
        for j in range(grid_size):
            q_values = agent.q_table[i, j]
            max_action = np.argmax(q_values)
            
            ax[i, j].imshow(np.zeros((1, 1)), cmap='gray', vmin=0, vmax=1)  # 빈 칸
            ax[i, j].set_title(f"{actions[max_action]}\n{q_values[max_action]:.2f}", fontsize=10)
            ax[i, j].axis('off')
    
    plt.suptitle("Learned Q-Values and Preferred Actions", fontsize=10)
    plt.tight_layout()
    plt.show()



 