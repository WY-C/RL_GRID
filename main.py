from env import GridEnv, GridEnvGUI
from Agent import DQNAgent
from matplotlib import pyplot as plt
#from stable_baselines3 import DQN  # 또는 DQN, A2C 등

# env = GridEnv(5)
# model = DQN("MlpPolicy", env)
# model.learn(total_timesteps=10000)
# model.save("dqn_grid")

env = GridEnv(5)
Agent1 = DQNAgent(state_size=8, action_size=4)
Agent2 = DQNAgent(state_size=8, action_size=4)

tot_reward = 0
tot_timestep = 0
printing = 100
action = [0, 0]
reward = [0, 0]


#save, load시 target model, epsilon도 같이.
try:
    Agent1.load_model("model/Agent1_mk2_111100.pth")
    Agent2.load_model("model/Agent2_mk2_111100.pth")
    print("Model loaded successfully.")
except:
    print("Model not found, starting training from scratch.")
for i in range(100000000):
    state, _= env.reset()
    terminated = False
    while not terminated:
        #다른 observasation
        #reward 는 r1,r2를 더하고
        #Action은 [0] + [1] concat
        #sb3 vs custom model
        #다른 env와 DQN 비교하기
        #DQN, PPO, SAC, MPO

        #DDQN, DQN, PER, Replay 비교하기

        #CHI 2025 재밌는 논문 몇 개 읽어오기
        action[0] = Agent1.choose_action(state)
        action[1] = Agent2.choose_action(state)
        next_state, reward, terminated, _, _ = env.step(action) #truncated 없음
        Agent1.replay_buffer.add(state, action[0], reward[0], next_state, terminated, 1.0)
        Agent2.replay_buffer.add(state, action[1], reward[1], next_state, terminated, 1.0)

        #reward 연산 해결하기
        tot_reward += (reward[0] + reward[1])
        #epsilon 수정하기
        #env.render()
        state = next_state.copy()
        Agent1.update()
        Agent2.update()
        

    if terminated:
        #print(tot_reward, env.get_timestep())
        timestep = env.get_timestep()
        tot_timestep += timestep

    if (i + 1) % 100 == 0:
        Agent1.update_target_model() 
    if terminated and (i + 1) % printing == 0:
        DQNAgent.epsilon = max(DQNAgent.epsilon_min, DQNAgent.epsilon * DQNAgent.epsilon_decay)  
        print(f"Episode {i+1} finished with reward: {tot_reward/printing:.2f}, timestep: {tot_timestep/printing:.3f}, Epsilon: {DQNAgent.epsilon:.3f}")
        tot_timestep = 0
        tot_reward = 0

    if (i + 1) % 1000 == 0:
        Agent1.save_model(f"model/{i+1}_Agent1_mk2.pth")
        Agent2.save_model(f"model/{i+1}_Agent2_mk2.pth")
        print(f"Model saved at episode {i+1}")
    