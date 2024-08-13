# DQN_model.py
import random
import torch.nn as nn
import torch.nn.functional as F

class Qnet(nn.Module):
    def __init__(self, no_actions, no_states):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(no_states, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, no_actions)
        self.no_actions = no_actions  # Store number of actions

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, observation, epsilon):
        q_values = self.forward(observation)
        if random.random() < epsilon:
            return random.randint(0, self.no_actions - 1)
        else:
            return q_values.argmax().item()
