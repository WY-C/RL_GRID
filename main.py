from env import GridEnv
from Agent import DQNAgent


Agent = DQNAgent(state_size=4, action_size=4)
env = GridEnv(5)
tot_reward = 0
tot_ticks = 0
printing = 100
for i in range(100000000):
    state, _ = env.reset()
    
    terminated = False
    while not terminated:
        action = Agent.choose_action(state)
        next_state, reward, terminated, _, _ = env.step(action) #truncated 없음
        Agent.replay_buffer.add(state, action, reward, next_state, terminated)

        tot_reward += reward
        state = next_state.copy()
        Agent.update()

    if terminated:
        tot_ticks += env.get_ticks()
    if (i + 1) % 50 == 0:
        Agent.update_target_model() 
    if terminated and (i + 1) % printing == 0:
        Agent.epsilon = max(Agent.epsilon_min, Agent.epsilon * Agent.epsilon_decay)  
        print(f"Episode {i+1} finished with reward: {tot_reward/printing:.2f}, ticks: {tot_ticks/printing:.3f}, Epsilon: {Agent.epsilon:.3f}")
        tot_ticks = 0
        tot_reward = 0