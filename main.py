#todo
# reward 위치 고정하고, 벽 위치는 랜덤. #진행중
# DQN으로 변환
# 혼자 학습시키고, 멀티를 따로 학습하기. 
# environment에 done 추가했음.
# 
#
#
#
#
#
#
#
#탐험률을 줄여가면서 학습
import numpy as np
import matplotlib.pyplot as plt

from environment.environment import GridEnvironment_1player, GridEnvironment_2player
from training.training_Qlearning import SelfPlayAgent, solo_play, self_play, visualize_q_values 


# 시각화 코드
"""
env = GridEnvironment_1player()
agent1 = SelfPlayAgent(env, agent_id=0)
agent2 = SelfPlayAgent(env, agent_id=1)

agent1.q_table = np.load("agent1_q_table.npy")
agent2.q_table = np.load("agent2_q_table.npy")


# 좌표별 최댓값 및 방향 계산
max_values = np.max(agent1.q_table, axis=2)  # 각 좌표에서 최댓값
max_directions = np.argmax(agent1.q_table, axis=2)  # 각 좌표에서 최댓값의 행동 (0: 상, 1: 하, 2: 좌, 3: 우)

# 시각화를 위해 방향 이름 정의
directions = ['↑', '↓', '←', '→']

# 히트맵으로 최댓값 시각화
plt.figure(figsize=(8, 6))
plt.imshow(max_values, cmap='viridis', interpolation='nearest')
plt.colorbar(label="Max Q-Value")
plt.title("Maximum Q-Values and Directions")

# 각 셀에 방향 추가
for i in range(max_values.shape[0]):  # 행
    for j in range(max_values.shape[1]):  # 열
        plt.text(j, i, directions[max_directions[i, j]], ha='center', va='center', color='white', fontsize=8)

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
"""

#학습 코드

env1 = GridEnvironment_1player()
agent1 = SelfPlayAgent(env1, agent_id=0)
solo_play(env1, agent1, episodes=100000, test=False)
#np.save("agent1_q_table.npy", agent1.q_table)

#self_play(env, agent1, agent1, episodes=1000000)
#agent2 = SelfPlayAgent(env1, agent_id=1)
#solo_play(env1, agent2, episodes=100000, test=False)
#np.save("agent2_q_table.npy", agent2.q_table)
#print("학습 완료!")

#시각화 코드
#plt.plot(x, y, label='training', color='blue', linestyle='--', marker='o')
#plt.title("Line Graph Example")
#plt.xlabel("episode")
#plt.ylabel("reward")
#plt.legend()  # 범례 추가
#plt.grid(True)  # 격자 추가
#plt.show()

#visualize_q_values(agent1)
#visualize_q_values(agent2)