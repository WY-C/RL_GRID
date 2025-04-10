from training import GridEnvironment, SelfPlayAgent
import numpy as np

def user_choose_action():
    key_map = {'w': 0, 's': 1, 'a': 2, 'd': 3}
    while True:
        key = input("사용자 방향 입력 (↑:w ↓:s ←:a →:d): ").strip().lower()
        if key in key_map:
            return key_map[key]
        else:
            print("❗잘못된 입력입니다. w/s/a/d 중 하나를 입력해주세요.")


def cooperative_play(env, agent1, episodes=5):
    agent1.epsilon = 0  # 학습된 에이전트는 활용만 함

    for episode in range(episodes):
        env.reset()
        print(f"\n▶ 에피소드 {episode+1} 시작")
        print(f"보상 위치: {env.reward_pos}")

        done = False
        step = 0

        while not done:
            print(f"\n🧭 스텝 {step}")
            print(f"에이전트1 위치: {env.agent1_pos}")
            print(f"에이전트2 위치: {env.agent2_pos}")

            # 에이전트1 자동 선택
            action1 = agent1.choose_action(env.agent1_pos)
            env.move(env.agent1_pos, action1)
            print(f"에이전트1 이동 방향: {['↑','↓','←','→'][action1]}")

            # 사용자 입력
            action2 = user_choose_action()
            env.move(env.agent2_pos, action2)

            # 결과 확인
            if env.check_reward(env.agent1_pos) or env.check_reward(env.agent2_pos):
                print("🎉 보상 획득!")
                done = True
            elif env.check_collision():
                print("💥 충돌 발생!")
                done = True

            step += 1

        print(f"에피소드 {episode+1} 종료.\n")


# 나중에 불러오기 (새 인스턴스를 만들어도 됨)
env = GridEnvironment()
agent1 = SelfPlayAgent(env, agent_id=0)
agent1.q_table = np.load("agent1_q_table.npy")  # 학습된 Q-table 불러오기

cooperative_play(env, agent1, episodes=3)
