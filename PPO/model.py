import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

class Agent(torch.nn.Module):
    def __init__(self, numObs, numActions, numHidden):
        super(Agent, self).__init__()
        self.critic_input  = nn.Linear(numObs, numHidden)
        self.critic_fc1    = nn.Linear(numHidden, numHidden)
        self.critic_output = nn.Linear(numHidden, 1)

        self.actor_input  = nn.Linear(numObs, numHidden)
        self.actor_fc1    = nn.Linear(numHidden, numHidden)
        self.actor_output = nn.Linear(numHidden, numActions)
        

    def forward(self, x):
        y = F.relu(self.actor_input(x))
        y = F.relu(self.actor_fc1(y))
        logits = self.actor_output(y)

        z = F.relu(self.critic_input(x))
        z = F.relu(self.critic_fc1(z))
        value = self.critic_output(z)

        return logits, value
