from env import GridEnv, GridEnvGUI
from Agent import DQNAgent, ReplayAgent
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.callbacks import BaseCallback

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []  # 각 에피소드의 총 보상 저장
        self.episode_lengths = []  # 각 에피소드의 길이 저장

    def _on_step(self) -> bool:
        """
        매 스텝마다 호출됩니다. 에피소드 종료 시 보상 기록.
        """
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info.keys():
                    # 에피소드 종료 시 보상 및 길이 저장
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
                    if self.verbose > 0:
                        print(f"에피소드 종료 - 총 보상: {info['episode']['r']}, 길이: {info['episode']['l']}")
        return True  # 학습 계속 진행


# 환경 초기화
#env = DummyVecEnv(lambda: GridEnv(grid_size=5))
env = GridEnv(5)
model = DQN("MlpPolicy", env, verbose=1)
reward_logger = RewardLoggerCallback(verbose=1)
model.learn(total_timesteps=200000, callback=reward_logger)
print(reward_logger.episode_rewards)
model.save("dqn_grid")
# SB3 호환성 확인
#check_env(env, warn=True)

# 벡터화된 환경 생성
#vec_env = DummyVecEnv([lambda: env])

"""
Agent1 = ReplayAgent(state_size=8, action_size=4)
Agent2 = ReplayAgent(state_size=8, action_size=4)

tot_reward = 0
tot_timestep = 0
printing = 100
action = [0, 0]
reward = [0, 0]


#save, load시 target model, epsilon도 같이.
try:
    Agent1.load_model("model/10000000_Agent1_mk4.pth")
    Agent2.load_model("model/10000000_Agent2_mk4.pth")
    print("Model loaded successfully.")
except:
    print("Model not found, starting training from scratch.")
for i in range(100000000):
    state= env.reset()
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
        Agent1.replay_buffer.add(state, action[0], reward[0], next_state, terminated)#, 1.0) #replay는 안쓰고, per은 씀.
        Agent2.replay_buffer.add(state, action[1], reward[1], next_state, terminated)#, 1.0)

        #reward 연산 해결하기
        tot_reward += (reward[0] + reward[1])
        #epsilon 수정하기
        #env.render()
        #최대 tick 출력하기
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
        Agent1.save_model(f"model/{i+1}_Agent1_DDQN_Replay.pth")
        Agent2.save_model(f"model/{i+1}_Agent2_DDQN_Replay.pth")
        print(f"Model saved at episode {i+1}")
    
        

"""