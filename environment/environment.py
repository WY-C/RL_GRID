import random
#아직 벽은 구현하지 않음.

class GridEnvironment_1player:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.agent_pos = [0, 0]  # 에이전트1 초기 위치
        self.reward_pos = self.generate_reward_pos() # 리워드 위치
        self.wall_pos = self.generate_wall_pos()  # 벽 (랜덤)
        
        if self.agent_pos == self.reward_pos or self.agent_pos == self.wall_pos or self.reward_pos == self.wall_pos:
            while self.agent_pos == self.reward_pos or self.agent_pos == self.wall_pos or self.reward_pos == self.wall_pos:
                self.agent_pos = self.generate_reward_pos()
                self.wall_pos = self.generate_wall_pos()



    def generate_wall_pos(self):
        # 벽 위치 생성 (예: 0~4 사이의 랜덤 위치)
        wall_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        return wall_pos
     
    def generate_reward_pos(self):
        reward_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        return reward_pos

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
            if reward == 0:
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
            else:
                reward = -0.1
            return reward, 0, old_pos

        
    def reset(self):
        # 환경 초기화
        self.agent_pos = [0, 0]
        self.reward_pos = self.generate_reward_pos()  # 리워드 위치 고정
        self.wall_pos = self.generate_wall_pos()  # 벽 위치 초기화

        if self.agent_pos == self.reward_pos or self.agent_pos == self.wall_pos or self.reward_pos == self.wall_pos:
            while self.agent_pos == self.reward_pos or self.agent_pos == self.wall_pos or self.reward_pos == self.wall_pos:
                self.agent_pos = self.generate_reward_pos()
                self.wall_pos = self.generate_wall_pos()


#벽 아직 없음
class GridEnvironment_2player:

    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.agent_pos = [0, 0]  # 에이전트1 초기 위치
        self.agent2_pos = [0, 1]  # 에이전트2 초기 위치
        self.reward_pos = self.generate_reward_pos()

    def generate_reward_pos(self):
        values = [0, 1, 2, 3, 4]
        weights_x = [0.2, 0.2, 0.9, 0.2, 0.2]  # 각 위치에 대한 가중치
        weights_y = [0.2, 0.2, 0.2, 0.9, 0.2]
        # 랜덤 리워드 위치 생성
        x = random.choices(values, weights_x)[0]
        y = random.choices(values, weights_y)[0]
        return [x, y]

    def move(self, entity, action):
        # 이동: 상(0), 하(1), 좌(2), 우(3)
        if action == 0 and entity[0] > 0:
            entity[0] -= 1
        elif action == 1 and entity[0] < self.grid_size - 1:
            entity[0] += 1
        elif action == 2 and entity[1] > 0:
            entity[1] -= 1
        elif action == 3 and entity[1] < self.grid_size - 1:
            entity[1] += 1

    def check_collision(self):
        # 에이전트1과 에이전트2가 동일한 위치에 있는지 확인
        return self.agent_pos == self.agent2_pos

    def check_reward(self):

        # 리워드 획득 여부 확인
        return self.agent_pos == self.reward_pos
    
    def reset(self):
        # 환경 초기화
        self.agent_pos = [0, 0]
        self.agent2_pos = [0, 1]
        self.reward_pos = self.generate_reward_pos()