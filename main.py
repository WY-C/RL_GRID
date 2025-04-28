from env import GridEnv
from Agent import DQNAgent

Agent = DQNAgent(state_size=4, action_size=4)
env = GridEnv(5)
tot_reward = 0
printing = 100
for i in range(10000):
    state, _ = env.reset()
    #print(state)
    terminated = False
    while not terminated:
        action = Agent.choose_action(env.state)
        #print(1)
        next_state, reward, terminated, _, _ = env.step(action) #truncated 없음.
        #env.render()
        Agent.replay_buffer.add(state, action, reward, next_state, terminated)
        tot_reward += reward
        state = next_state
        Agent.update()

        
    if terminated and (i + 1) % printing == 0:
        Agent.update_target_model()
        print(f"Episode {i+1} finished with reward: {tot_reward/printing:.3f}, Epsilon: {Agent.epsilon:.5f}")
        Agent.epsilon = max(Agent.epsilon_min, Agent.epsilon * Agent.epsilon_decay)
        tot_reward = 0
        
