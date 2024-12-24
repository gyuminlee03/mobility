# 필요한 라이브러리 임포트
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from torch.distributions import Categorical
import gym

from gym import envs
print([env_id for env_id in envs.registry.keys() if "Pong" in env_id])



# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Pong 환경 초기화
env = gym.make('ALE/Pong-v5')
print('Observation space:', env.observation_space)
print('Action space:', env.action_space)

# 정책 네트워크 정의
class Policy(nn.Module):
    def __init__(self, state_size=6400, action_size=6, hidden_size=256):
        super(Policy, self).__init__()
        # Flatten된 픽셀 데이터를 input으로 받음 (state_size: 80x80 흑백 이미지)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

# 환경 관측값 전처리
def preprocess(frame):
    # Convert to grayscale, crop, and downsample
    frame = frame[35:195]  # Crop to remove score and boundaries
    frame = frame[::2, ::2, 0]  # Downsample by factor of 2 and pick one channel
    frame[frame == 144] = 0  # Remove background type 1
    frame[frame == 109] = 0  # Remove background type 2
    frame[frame != 0] = 1  # Set paddles and ball to 1
    return frame.astype(np.float32).ravel()  # Flatten the array

# REINFORCE 알고리즘
def reinforce(policy, optimizer, n_episodes=1000, max_t=1000, gamma=0.99, print_every=10):
    scores_deque = deque(maxlen=100)
    scores = []
    for e in range(1, n_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = preprocess(env.reset()[0])
        episode_reward = 0

        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            next_state, reward, done, _, _ = env.step(action)
            state = preprocess(next_state)
            rewards.append(reward)
            episode_reward += reward
            if done:
                break

        scores_deque.append(episode_reward)
        scores.append(episode_reward)

        # Discounted rewards 계산
        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])
        
        # 정책 손실 계산
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        
        # 역전파 및 가중치 업데이트
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if e % print_every == 0:
            print(f"Episode {e}\tAverage Score: {np.mean(scores_deque):.2f}")
        if np.mean(scores_deque) >= 20.0:
            print(f"Environment solved in {e - 100} episodes!\tAverage Score: {np.mean(scores_deque):.2f}")
            break
    
    return scores

# 메인 실행
policy = Policy(state_size=6400, action_size=env.action_space.n).to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
scores = reinforce(policy, optimizer, n_episodes=1000, print_every=100)



# 에이전트 성능 시각화
import time
env = gym.make('Pong-v0', render_mode="human")
state = preprocess(env.reset()[0])
done = False

while not done:
    env.render()
    action, _ = policy.act(state)
    next_state, reward, done, _, _ = env.step(action)
    state = preprocess(next_state)
    time.sleep(0.1)

env.close()
