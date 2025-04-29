import gymnasium as gym
import random
import numpy as np

class GridEnv(gym.Env):
    def __init__(self, grid_size): #goal position 제거 확인하기
        super(GridEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right
        self.state = np.zeros(4)
        self.ticks = 0

    def reset(self):
        self.ticks = 0
        
        self.state[0] = 0
        self.state[1] = 0

        goal = random.sample(range(1, self.grid_size*self.grid_size), 1)
        self.state[2], self.state[3] = goal[0] // self.grid_size, goal[0] % self.grid_size
        
        #self.state[2] = self.grid_size - 1
        #self.state[3] = self.grid_size - 1
        return self.state, {}

    def step(self, action):
        self.ticks += 1
        assert self.action_space.contains(action), f"{action!r} ({type(action)})는 유효하지 않은 동작입니다."
        
        x, y, gx, gy = self.state[0], self.state[1], self.state[2], self.state[3]
        
        old_distance = abs(x - gx) + abs(y - gy)




        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # Left
            y = max(0, y - 1)
        elif action == 3:  # Right
            y = min(self.grid_size - 1, y + 1)

        new_distance = abs(x - gx) + abs(y - gy)

        self.state[0] = x
        self.state[1] = y


 
        if new_distance < old_distance:
            reward = 0.2
        else:
            reward = -0.17

        if (x == gx) and (y == gy):
            terminated = True
        else:
            terminated = False         
               
        if terminated:
            reward = (1.5 - self.ticks / 100)
        #print(self.state, reward, terminated)
        return self.state, reward, terminated, False, {}
    
    def get_ticks(self):
        return self.ticks

    def render(self):
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        x, y = self.state[0], self.state[1]
        grid[x][y] = 'A'  # Agent position
        gx, gy = self.state[2], self.state[3]
        grid[gx][gy] = 'G'  # Goal position

        for row in grid:
            print(' '.join(row))
        print()

    def check_wall(self, action):
        x, y = self.state[0], self.state[1]
        if action == 0 and x == 0:
            return True
        elif action == 1 and x == self.grid_size - 1:
            return True
        elif action == 2 and y == 0:
            return True
        elif action == 3 and y == self.grid_size - 1:
            return True
        return False