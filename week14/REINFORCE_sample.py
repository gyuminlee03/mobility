    

#초기과정
#$ pip install gym[classic_control] 해야함 터미널 창에서 = gym 설치!!
#$ pip install pygame
#$ python3 로 터미널에서 실행

# Import necessary libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque
from torch.distributions import Categorical
import gym

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda 설치, 사용 가능한 함수를 띄워주기
print(device)

env = gym.make('CartPole-v0') # gym environment를 초기화하는것
# gymlibrary.dev/environments/atari 사이트

print('observation space:', env.observation_space) # state 공간 한번 얼마나 되는지 출력해보기
print('action space:', env.action_space) # action space 출력해보기

class Policy(nn.Module): # 아까식에서 봤던 파이o세타
    def __init__(self, state_size=4, action_size=2, hidden_size=32):
        super(Policy, self).__init__()
        # state(4) -> hidden hidden(32) -> action(2)
        #보면 굉장히 작은 network임!!
        self.fc1 = nn.Linear(state_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        # to-do
        return F.softmax(x, dim=1)
    
    def act(self, state): 
        # to-do
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        model = Categorical(probs)
        action = model.sample()
        return action.item(), model.log_prob(action)


# Reinforce algorithm 부분
def reinforce(policy, optimizer, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []
    for e in range(1, n_episodes):
        saved_log_probs = []
        rewards = []
        state = env.reset()[0]

        # Collect trajectory(경로를 구하는 부분 우리 다 식에서 봤었음 초기 state가 주어지고 )
        for t in range(max_t):
            # Sample the action from current policy
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            if done:
                break

        # Calculate total expected reward
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))


        #이득 R값과 할인을 구하는 것
        # Recalculate the total reward applying discounted factor
        discounts = [gamma ** i for i in range(len(rewards) + 1)] # discount = gamma 값!
        R = sum([a * b for a,b in zip(discounts, rewards)])
        
        # Calculate the loss 
        # to-do
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)

        #After that, we concatenate while policy loss in 0th dimension
        policy_loss = torch.cat(policy_loss).sum()
    
        # Backpropagation
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if e % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e - 100, np.mean(scores_deque)))
            break
    return scores



#메인====
policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
scores = reinforce(policy, optimizer, n_episodes=2000) # REinforcement 부분 (학습시키는 부분) = 균형맞추기!

# Visualize the agent's performance
import time

env = gym.make("CartPole-v1", render_mode="human")
state = env.reset()[0]
done = False

while not done:
    env.render()
    action, _ = policy.act(state)
    next_state, reward, done, _, _ = env.step(action)
    state = next_state
    time.sleep(0.1)  # Add a delay to make the visualization easier to follow

env.close()