import tkinter as tk
from env import GridEnv
import time

class GridEnvWithTk(GridEnv):
    def __init__(self, grid_size):
        super().__init__(grid_size)
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=grid_size * 50, height=grid_size * 50 + 50)  # 추가 공간 확보
        self.canvas.pack()
        self.cell_size = 50

    def render(self):
        self.canvas.delete("all")

        # Draw grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1, y1 = j * self.cell_size, i * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black")

        # Draw agents
        x1, y1 = self.state[0], self.state[1]
        x2, y2 = self.state[2], self.state[3]
        self.canvas.create_oval(
            y1 * self.cell_size + 10, x1 * self.cell_size + 10,
            (y1 + 1) * self.cell_size - 10, (x1 + 1) * self.cell_size - 10,
            fill="blue", tag="agent1"
        )
        self.canvas.create_oval(
            y2 * self.cell_size + 10, x2 * self.cell_size + 10,
            (y2 + 1) * self.cell_size - 10, (x2 + 1) * self.cell_size - 10,
            fill="red", tag="agent2"
        )

        # Draw goals
        gx1, gy1, gx2, gy2 = self.state[4], self.state[5], self.state[6], self.state[7]
        if gx1 < self.grid_size and gy1 < self.grid_size:
            self.canvas.create_rectangle(
                gy1 * self.cell_size + 5, gx1 * self.cell_size + 5,
                (gy1 + 1) * self.cell_size - 5, (gx1 + 1) * self.cell_size - 5,
                fill="green", tag="goal1"
            )
        if gx2 < self.grid_size and gy2 < self.grid_size:
            self.canvas.create_rectangle(
                gy2 * self.cell_size + 5, gx2 * self.cell_size + 5,
                (gy2 + 1) * self.cell_size - 5, (gx2 + 1) * self.cell_size - 5,
                fill="yellow", tag="goal2"
            )

        # Add text: timestep and rewards
        timestep_text = f"Timestep: {self.timestep}"
        rewards_text = f"Agent 1 Reward: {self.reward[0]:.2f}, Agent 2 Reward: {self.reward[1]:.2f}"

        # Display text below the grid
        self.canvas.create_text(
            self.grid_size * self.cell_size / 2, self.grid_size * self.cell_size + 20,
            text=timestep_text, fill="black", font=("Helvetica", 8)
        )
        self.canvas.create_text(
            self.grid_size * self.cell_size / 2, self.grid_size * self.cell_size + 40,
            text=rewards_text, fill="black", font=("Helvetica", 8)
        )

        self.root.update()

    def close(self):
        self.root.destroy()
def run_environment(env, max_steps=100, step_delay=0.5):
    """
    실행 환경을 초기화하고 지정된 스텝 수만큼 동작을 시뮬레이션합니다.

    Args:
        env (GridEnvWithTk): 환경 인스턴스.
        max_steps (int): 최대 스텝 수.
        step_delay (float): 각 스텝 간의 대기 시간(초).
    """
    try:
        # 환경 초기화
        state, _ = env.reset()
        print("Environment initialized. Starting simulation...")

        for step in range(max_steps):
            # 랜덤 액션 생성 (에이전트 두 명 각각)
            action = [env.action_space.sample(), env.action_space.sample()]

            # 환경 업데이트
            state, reward, terminated, _, _ = env.step(action)

            # 렌더링
            env.render()

            # 콘솔 출력
            print(f"Step: {step + 1}, State: {state}, Reward: {reward}")

            # 종료 조건 확인
            if terminated:
                print(f"Simulation ended at step {step + 1}. Goal reached!")
                break

            # 스텝 간 대기
            time.sleep(step_delay)

        print("Simulation finished.")
    finally:
        # GUI 리소스 정리
        env.close()

# 실행 테스트
if __name__ == "__main__":
    env = GridEnvWithTk(grid_size=5)
    run_environment(env, max_steps=20, step_delay=0.5)