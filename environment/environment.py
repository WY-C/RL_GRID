import random
class GridEnvironment_1player:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.agent_pos = [0, 0]  # 에이전트1 초기 위치
        self.reward_pos = [0, 0]
        self.wall_pos = [0, 0]
        self.set_position()
        #print(self.reward_pos, self.wall_pos)



    def set_position(self):
        reward, wall = random.sample(range(1, 25), 2)
        self.reward_pos = [reward // 5, reward % 5]
        self.wall_pos = [wall // 5, wall % 5]

    def move(self, action):
        # 이동: 상(0), 하(1), 좌(2), 우(3)
        if action == 0 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.grid_size - 1:
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.grid_size - 1:
            self.agent_pos[1] += 1
    
    # 리워드 획득 여부 확인
    def check_reward(self):
        if self.reward_pos == self.agent_pos:
            return True
        
        return False
    
    def check_wall(self):
        # 벽 위치 확인
        return self.agent_pos == self.wall_pos 
    
    def step(self, action, episode_ticks):
        # 에이전트 이동
        old_pos = self.agent_pos.copy()
        self.move(action)

        #reward에 도착
        if self.check_reward():
            reward = 1.5 - (episode_ticks / 100) 
            if reward < 1e-16:
                reward = 1e-16  # 리워드가 0일 때 작은 값으로 설정
            return reward, 1, old_pos
        #벽에 부딪힘. 
        elif self.check_wall():
            self.agent_pos = old_pos 
            reward = -0.05
            return reward, 1, old_pos
        #대부분의 이동
        else:
            old_pos_distance = abs(old_pos[0] - self.reward_pos[0]) + abs(old_pos[1] - self.reward_pos[1])
            new_pos_distance = abs(self.agent_pos[0] - self.reward_pos[0]) + abs(self.agent_pos[1] - self.reward_pos[1])
            if new_pos_distance < old_pos_distance:
                reward = 0.1
            #reward와 거리가 좁혀지지 않음.
            else:
                reward = -0.1
            return reward, 0, old_pos

        
    def reset(self):
        self.agent_pos = [0, 0]  # 에이전트1 초기 위치
        self.reward_pos = [0, 0]
        self.wall_pos = [0, 0]
        self.set_position()

class GridEnvironment_2player:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.agent1_pos = [0, 0]  # 에이전트1 초기 위치
        self.agent1_done = False

        self.agent2_pos = [4, 4]
        self.agent2_done = False

        self.reward1_pos = [0, 0]
        self.reward2_pos = [0, 0]
        self.wall_pos = [0, 0]
        self.set_position()
        #print(self.reward_pos, self.wall_pos)



    def set_position(self):
        reward1, reward2, wall = random.sample(range(1, 24), 3)
        self.reward1_pos = [reward1 // 5, reward1 % 5]
        self.reward2_pos = [reward2 // 5, reward2 % 5]
        self.wall_pos = [wall // 5, wall % 5]

    def move(self, action, agent_num):
        if agent_num == 1:
            if action == 0 and self.agent1_pos[0] > 0:
                self.agent1_pos[0] -= 1
            elif action == 1 and self.agent1_pos[0] < self.grid_size - 1:
                self.agent1_pos[0] += 1
            elif action == 2 and self.agent1_pos[1] > 0:
                self.agent1_pos[1] -= 1
            elif action == 3 and self.agent1_pos[1] < self.grid_size - 1:
                self.agent1_pos[1] += 1
        elif agent_num == 2:
            if action == 0 and self.agent2_pos[0] > 0:
                self.agent2_pos[0] -= 1
            elif action == 1 and self.agent2_pos[0] < self.grid_size - 1:
                self.agent2_pos[0] += 1
            elif action == 2 and self.agent2_pos[1] > 0:
                self.agent2_pos[1] -= 1
            elif action == 3 and self.agent2_pos[1] < self.grid_size - 1:
                self.agent2_pos[1] += 1            
        # 이동: 상(0), 하(1), 좌(2), 우(3)

    
    # 리워드 획득 여부 확인
    def check_reward(self, agent_num):
        if agent_num == 1:
            if self.agent1_pos == self.reward1_pos:
                self.reward1_pos =[-99,-99]
                self.agent1_done = True
                return True
            elif self.agent1_pos == self.reward2_pos:
                self.reward2_pos =[-99,-99]
                self.agent1_done = True
                return True
            return False
        else:
            if self.agent2_pos == self.reward1_pos:
                self.reward1_pos =[-99,-99]
                self.agent2_done = True
                return True
            elif self.agent2_pos == self.reward2_pos:
                self.reward2_pos =[-99,-99]
                self.agent2_done = True
                return True
            return False
        
    def check_wall(self, agent_num):
        if agent_num == 1:
            return self.agent1_pos == self.wall_pos
        else:
            return self.agent2_pos == self.wall_pos
    
    def step(self, action, episode_ticks, agent_num):
        if agent_num == 1:
        # 에이전트 이동
            old_pos = self.agent1_pos.copy()
            self.move(action, agent_num)

            #reward에 도착
            if self.check_reward(agent_num):
                reward = 1.5 - (episode_ticks / 100) 
                if reward < 1e-16:
                    reward = 1e-16  # 리워드가 0일 때 작은 값으로 설정                
                return reward, 1, old_pos
            #벽에 부딪힘. 
            elif self.check_wall(agent_num):
                self.agent1_pos = old_pos 
                reward = -0.05
                return reward, 1, old_pos
            #대부분의 이동
            else:

                old_pos_distance_1 = abs(old_pos[0] - self.reward1_pos[0]) + abs(old_pos[1] - self.reward2_pos[1])
                old_pos_distance_2 = abs(old_pos[0] - self.reward2_pos[0]) + abs(old_pos[1] - self.reward2_pos[1])
                #2번이 더 가까움
                if old_pos_distance_1 > old_pos_distance_2:
                    old_pos_distance = old_pos_distance_2
                    new_pos_distance = abs(self.agent1_pos[0] - self.reward2_pos[0]) + abs(self.agent1_pos[1] - self.reward2_pos[1])
                #1번이 더 가까움.
                else:
                    old_pos_distance = old_pos_distance_1
                    new_pos_distance = abs(self.agent1_pos[0] - self.reward1_pos[0]) + abs(self.agent1_pos[1] - self.reward1_pos[1])
                if new_pos_distance < old_pos_distance:
                    reward = 0.1
                #reward와 거리가 좁혀지지 않음.
                else:
                    reward = -0.1
                return reward, 0, old_pos            

        elif agent_num == 2:
        # 에이전트 이동
            old_pos = self.agent2_pos.copy()
            self.move(action, agent_num)

            #reward에 도착
            if self.check_reward(agent_num):
                reward = 1.5 - (episode_ticks / 100) 
                if reward < 1e-16:
                    reward = 1e-16  # 리워드가 0일 때 작은 값으로 설정                
                return reward, 1, old_pos
            #벽에 부딪힘. 
            elif self.check_wall(agent_num):
                self.agent1_pos = old_pos 
                reward = -0.05
                return reward, 1, old_pos
            #대부분의 이동
            else:
                old_pos_distance_1 = abs(old_pos[0] - self.reward1_pos[0]) + abs(old_pos[1] - self.reward2_pos[1])
                old_pos_distance_2 = abs(old_pos[0] - self.reward2_pos[0]) + abs(old_pos[1] - self.reward2_pos[1])
                #1번이 더 가까움
                if old_pos_distance_2 > old_pos_distance_1:
                    old_pos_distance = old_pos_distance_1
                    new_pos_distance = abs(self.agent2_pos[0] - self.reward1_pos[0]) + abs(self.agent2_pos[1] - self.reward1_pos[1])
                #2번이 더 가까움.
                else:
                    old_pos_distance = old_pos_distance_2
                    new_pos_distance = abs(self.agent2_pos[0] - self.reward2_pos[0]) + abs(self.agent2_pos[1] - self.reward2_pos[1])
                if new_pos_distance < old_pos_distance:
                    reward = 0.1
                #reward와 거리가 좁혀지지 않음.
                else:
                    reward = -0.1
                return reward, 0, old_pos  

        
    def reset(self):
        self.agent1_pos = [0, 0]  # 에이전트1 초기 위치
        self.agent2_pos = [4, 4]
        self.agent1_done = False
        self.agent2_done = False
        self.set_position()
        self.set_position()


# 거리 계산 함수 (반복 제거용)
def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

