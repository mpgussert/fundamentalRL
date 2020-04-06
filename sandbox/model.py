import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

class Agent(torch.nn.Module):
    def __init__(self, numObs, numActions):
        super(Agent, self).__init__()
        self.critic_input  = nn.Linear(numObs, 32)
        self.critic_fc1    = nn.Linear(32, 32)
        self.critic_output = nn.Linear(32, 1)

        self.actor_input  = nn.Linear(numObs, 32)
        self.actor_fc1    = nn.Linear(32, 32)
        self.actor_output = nn.Linear(32, numActions)
        

    def forward(self, x):
        y = F.relu(self.actor_input(x))
        y = F.relu(self.actor_fc1(y))
        logits = self.actor_output(y)

        z = F.relu(self.critic_input(x))
        z = F.relu(self.critic_fc1(z))
        value = self.critic_output(z)

        return logits, value

class Actor(torch.nn.Module):
    def __init__(self, numObs, numActions):
        super(Actor, self).__init__()
        self.actor_input  = nn.Linear(numObs, 32)
        self.actor_fc1    = nn.Linear(32, 32)
        self.actor_output = nn.Linear(32, numActions)

    def forward(self, x):
        x = F.relu(self.actor_input(x))
        x = F.relu(self.actor_fc1(x))
        logits = self.actor_output(x)

        return logits

class Critic(torch.nn.Module):
    def __init__(self, numObs):
        super(Critic, self).__init__()
        self.critic_input  = nn.Linear(numObs, 32)
        self.critic_fc1    = nn.Linear(32, 32)
        self.critic_output = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.critic_input(x))
        x = F.relu(self.critic_fc1(x))
        value = self.critic_output(x)

        return value