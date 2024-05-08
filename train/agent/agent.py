import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from agent.schedule import LinearSchedule
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class simpleagent(nn.Module):
    def __init__(self, input_size, output_size):
        super(simpleagent, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gym

# Define the neural network architecture
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the DQN agent and Epsilon scheduler
scheduler = LinearSchedule(10000, 0.01, 1.0)

class DQNAgent():
    def __init__(self, state_size, action_size, memLength = 2000, gamma = 0.9,
                 epsilon = 3.0, minimalE = 0.01, eDecay = 0.99, lr = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memLength)
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = minimalE
        self.epsilon_decay = eDecay
        self.model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return torch.tensor(random.randrange(self.action_size), device=device)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.model(state)
        return q_values.argmax().item()

    def replay(self, batch_size, t):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(state).squeeze(0).tolist()
            target_f[action] = target
            target_f = torch.FloatTensor(target_f).to(device)
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(state).squeeze(0), target_f)
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon = scheduler.value(t)

def createAgents(cfg):
    agents = []
    if(cfg["type"]=="DQN"):
        for i in range(cfg["number"]):
            a = DQNAgent(cfg["state_size"], cfg["action_size"], cfg["memLength"], cfg["gamma"],cfg["epsilon"], cfg["minimalE"], cfg["eDecay"], cfg["lr"])
            
            agents.append(a)
    #else: 
    #    for i in number:
    #        agents.append(agent(inputSize,outputSize))
    return agents
