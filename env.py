import gymnasium as gym
import random
import numpy as np
import tkinter as tk
import time
class GridEnv(gym.Env):


    def __init__(self, grid_size): #goal position 제거 확인하기
        super(GridEnv, self).__init__()
        self.grid_size = grid_size
        #self.observation_space = gym.spaces.Discrete(8)
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right
        self.state = np.zeros(8)
        self.ticks = 0
        self.reward = np.zeros(2)
        self.timestep = 1
        self.tot_reward = 0

        self.grid_GUI = GridEnvGUI(grid_size)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.timestep = 1
        self.tot_reward = 0
        #timestep
        
        self.state[0] = 0
        self.state[1] = 0
        self.state[2] = self.grid_size - 1
        self.state[3] = self.grid_size - 1

        goal = random.sample(range(1, self.grid_size*self.grid_size), 2)
        self.state[4], self.state[5] = goal[0] // self.grid_size, goal[0] % self.grid_size
        self.state[6], self.state[7] = goal[1] // self.grid_size, goal[1] % self.grid_size
        
        #self.state[2] = self.grid_size - 1
        #self.state[3] = self.grid_size - 1
        return self.state, {}

    def step(self, action):
        
        assert self.action_space.contains(action[0]), f"{action[0]!r} ({type(action[0])})는 유효하지 않은 동작입니다."
        assert self.action_space.contains(action[1]), f"{action[1]!r} ({type(action[1])})는 유효하지 않은 동작입니다."
        
        x1, y1, x2, y2 = self.state[0], self.state[1], self.state[2], self.state[3]
        gx1, gy1, gx2, gy2 = self.state[4], self.state[5], self.state[6], self.state[7]
        
        if(abs(x1 - gx1) + abs(y1 - gy1)) > (abs(x1 - gx2) + abs(y1 - gy2)):
            old_distance_1 = abs(x1 - gx2) + abs(y1 - gy2)
            Agent1_Goal = 2
        else:
            old_distance_1 = abs(x1 - gx1) + abs(y1 - gy1)
            Agent1_Goal = 1

        if(abs(x2 - gx1) + abs(y2 - gy1)) > (abs(x2 - gx2) + abs(y2 - gy2)):
            old_distance_2 = abs(x2 - gx2) + abs(y2 - gy2)
            Agent2_Goal = 2
        else:
            old_distance_2 = abs(x2 - gx1) + abs(y2 - gy1)
            Agent2_Goal = 1



        if action[0] == 0:  # Up
            x1 = max(0, x1 - 1)
        elif action[0] == 1:  # Down
            x1 = min(self.grid_size - 1, x1 + 1)
        elif action[0] == 2:  # Left
            y1 = max(0, y1 - 1)
        elif action[0] == 3:  # Right
            y1= min(self.grid_size - 1, y1 + 1)

        if action[1] == 0:  # Up
            x2 = max(0, x2 - 1)
        elif action[1] == 1:  # Down
            x2 = min(self.grid_size - 1, x2 + 1)
        elif action[1] == 2:  # Left
            y2 = max(0, y2 - 1)
        elif action[1] == 3:  # Right
            y2 = min(self.grid_size - 1, y2 + 1)

        if Agent1_Goal == 1:
            new_distance_1 = abs(x1 - gx1) + abs(y1 - gy1)
        else:
            new_distance_1 = abs(x1 - gx2) + abs(y1 - gy2)
        
        if Agent2_Goal == 1:
            new_distance_2 = abs(x2 - gx1) + abs(y2 - gy1)
        else:
            new_distance_2 = abs(x2 - gx2) + abs(y2 - gy2)

        self.state[0] = x1
        self.state[1] = y1
        self.state[2] = x2
        self.state[3] = y2

        #이상한 곳으로 갈수록 negative reward
        #더 이상할 수록 높은 reward를 준다.
        #positive는 goal에 도착할때만 준다.
        #render / GIF

        #goal을 먹으면 reward를 주긴 해야겠다.
        if new_distance_1 < old_distance_1:
            self.reward[0] = 0.15
        elif new_distance_1 == old_distance_1:
            self.reward[0] = -0.3
        else:
            self.reward[0] = -0.1

        if new_distance_2 < old_distance_2:
            self.reward[1] = 0.15
        elif new_distance_2 == old_distance_2:
            self.reward[1] = -0.3
        else:
            self.reward[1] = -0.1

        if (x1 == gx1 and y1 == gy1):
            self.reward[0] = 0.5
            self.reward[1] = 0.5
            gx1, self.state[4] = 99, 99
            gy1, self.state[5] = 99, 99
        elif (x2 == gx1 and y2 == gy1):
            self.reward[1] = 0.5        
            gx1, self.state[4] = 99, 99
            gy1, self.state[5] = 99, 99
        elif (x1 == gx2 and y1 == gy2):
            self.reward[0] = 0.5
            gx2, self.state[6] = 99, 99
            gy2, self.state[7] = 99, 99
        elif (x2 == gx2 and y2 == gy2):
            self.reward[1] = 0.5
            gx2, self.state[6] = 99, 99
            gy2, self.state[7] = 99, 99
               
        if gx1 == 99 and gx2 == 99 and gy1 == 99 and gy2 == 99:
            terminated = True
        else:
            terminated = False

        if terminated:
            self.reward[0] = (2 - self.timestep / 50)
            self.reward[1] = (2 - self.timestep / 50)
        #print(self.state, reward, terminated)
        self.timestep += 1
        self.tot_reward += (self.reward[0] + self.reward[1])
        return self.state, self.reward, terminated, False, {}
    
    def get_timestep(self):
        return self.timestep

    def render(self):
        self.grid_GUI.render(self.state[0:2], self.state[2:4], self.state[4:6], self.state[6:8], self.timestep, self.tot_reward)

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
    

class GridEnvGUI:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.cell_size = 50
        self.window = tk.Tk()
        self.canvas = tk.Canvas(self.window, width=(grid_size+1) * self.cell_size, height=(grid_size+1) * self.cell_size)
        self.canvas.pack()

    def draw_grid(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1 = i * self.cell_size
                y1 = j * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="black")

    def update_agent(self, agent_pos1, agent_pos2, goal_pos1, goal_pos2, color="blue"):
        Agent_X1, Agent_Y1 = agent_pos1
        Agent_X2, Agent_Y2 = agent_pos2
        Goal_X1, Goal_Y1 = goal_pos1
        Goal_X2, Goal_Y2 = goal_pos2
        x1 = Agent_X1 * self.cell_size
        y1 = Agent_Y1 * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)


        x1 = Agent_X2 * self.cell_size
        y1 = Agent_Y2 * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)



        x1 = Goal_X1 * self.cell_size
        y1 = Goal_Y1 * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)


        x1 = Goal_X2 * self.cell_size
        y1 = Goal_Y2 * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size        
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)
        time.sleep(0.05)

    def render(self, agent_pos1, agent_pos2, goal_pos1, goal_pos2, timestep, tot_reward, color="blue"):
        self.canvas.delete("all")
        self.draw_grid()
        Agent_X1, Agent_Y1 = agent_pos1
        Agent_X2, Agent_Y2 = agent_pos2
        Goal_X1, Goal_Y1 = goal_pos1
        Goal_X2, Goal_Y2 = goal_pos2
        x1 = Agent_X1 * self.cell_size
        y1 = Agent_Y1 * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, tag = "agent1")

        x1 = Agent_X2 * self.cell_size
        y1 = Agent_Y2 * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, tag = "agent2")

        x1 = Goal_X1 * self.cell_size
        y1 = Goal_Y1 * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="red",  tag = "goal1")

        x1 = Goal_X2 * self.cell_size
        y1 = Goal_Y2 * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size        
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="red", tag = "goal2")
        
        timestep_text = f"Timestep: {timestep}"
        rewards_text = f"tot_reward: {tot_reward:.2f}"
        self.canvas.create_text(
            self.grid_size * self.cell_size / 2, self.grid_size * self.cell_size + 20,
            text=timestep_text, fill="black", font=("Helvetica", 8)
        )
        self.canvas.create_text(
            self.grid_size * self.cell_size / 2, self.grid_size * self.cell_size + 40,
            text=rewards_text, fill="black", font=("Helvetica", 8)
        )

        
        
        
        self.window.update()

        time.sleep(0.1)