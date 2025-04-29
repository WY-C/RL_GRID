from env import GridEnv
from Agent import DQNAgent
from matplotlib import pyplot as plt


Agent1 = DQNAgent(state_size=8, action_size=4)
Agent2 = DQNAgent(state_size=8, action_size=4)
env = GridEnv(5)
tot_reward = 0
tot_ticks = 0
printing = 100
action = [0, 0]
reward = [0, 0]
for i in range(100000000):
    state, _ = env.reset()
    
    terminated = False
    while not terminated:
        action[0] = Agent1.choose_action(state)
        action[1] = Agent2.choose_action(state)
        next_state, reward, terminated, _, _ = env.step(action) #truncated 없음
        Agent1.replay_buffer.add(state, action[0], reward[0], next_state, terminated, 1.0)
        Agent2.replay_buffer.add(state, action[1], reward[1], next_state, terminated, 1.0)

        tot_reward += reward[0] #+ reward[1]
        
        state = next_state.copy()
        Agent1.update()
        Agent2.update()
        #PER도 update

    if terminated:
        #print(tot_reward, env.get_ticks())
        tot_ticks += env.get_ticks()

    if (i + 1) % 50 == 0:
        Agent1.update_target_model() 
    if terminated and (i + 1) % printing == 0:
        Agent.update_target_model()
        print(f"Episode {i+1} finished with reward: {tot_reward/printing:.3f}, Epsilon: {Agent.epsilon:.5f}")
        Agent.epsilon = max(Agent.epsilon_min, Agent.epsilon * Agent.epsilon_decay)
        tot_reward = 0
        
