

#Taxi 예제!! test!

#Y라는 승객을 태우고 B Goal까지 도달하는 Reinforcement Learning
#뭔가 Q-Learning 대신 DQN 알고리즘으로 전환하면 더 복잡한 문제를 해결하는데 도움을 줄 것 같음




# 필요한 라이브러리 임포트
import numpy as np
import gym
import random
from collections import defaultdict

# Taxi 환경 초기화
env = gym.make("Taxi-v3", render_mode="ansi")  # 텍스트 기반 렌더링
print("Action space:", env.action_space)  # 행동 공간 확인
print("State space:", env.observation_space)  # 상태 공간 확인

# Q-Learning 하이퍼파라미터
alpha = 0.1    # 학습률
gamma = 0.99   # 할인율
epsilon = 1.0  # 탐험률
epsilon_decay = 0.995
epsilon_min = 0.1
num_episodes = 5000
max_steps = 100  # 각 에피소드 최대 단계 수

# Q-테이블 초기화
q_table = defaultdict(lambda: np.zeros(env.action_space.n))

# Q-Learning 알고리즘
for episode in range(num_episodes):
    state, _ = env.reset()  # 환경 초기화 (state만 추출)
    total_reward = 0
    
    for step in range(max_steps):
        # Epsilon-Greedy 정책
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # 무작위 행동 선택
        else:
            action = np.argmax(q_table[state])  # Q-값이 가장 높은 행동 선택
        
        # 환경에 행동 적용
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Q-테이블 업데이트
        q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
        
        # 상태 업데이트
        state = next_state
        
        # 에피소드 종료 확인
        if done:
            break

    # Epsilon 감소 (탐험률 줄이기)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # 진행 상황 출력
    if (episode + 1) % 500 == 0:
        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward:.2f}")

# 학습 완료
print("Training finished!")

# 학습된 정책 테스트
state, _ = env.reset()  # 초기 상태
print(env.render())  # 텍스트 기반 렌더링 출력
done = False
total_reward = 0

while not done:
    action = np.argmax(q_table[state])  # 학습된 정책에 따라 행동 선택
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = next_state
    total_reward += reward
    print(env.render())  # 텍스트 기반 렌더링 출력

print(f"Total reward: {total_reward}")
env.close()
