import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

class Agent(torch.nn.Module):
    """
    A simple Actor Critic agent.  note that 
    we do not apply any form of normalization 
    or rescaling to the output of the actor 
    (the action logits). 
    """
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

class SharedAgent(torch.nn.Module):
    """
    A simple two headed / chimera Actor Critic agent.
    The actor and critic share the body of the network.
    It is argued that this is because "good" actions 
    correlate to visiting states with "large" values, and
    so there should exist some form of shared information 
    between these two functions, thus motivating the shared 
    body.  However, I haven't seen a rigorous proof of this, 
    and training an AC model with a shared body usually just 
    leads to added complications in my experience.  If you
    know a good reference for a mathematical proof on why 
    this should be done please let me know!
    """
    def __init__(self, numObs, numActions, numHidden):
        super(SharedAgent, self).__init__()
        self.shared_input  = nn.Linear(numObs, numHidden)
        self.shared_fc1    = nn.Linear(numHidden, numHidden)
        self.shared_fc2    = nn.Linear(numHidden, 2*numHidden)

        self.actor_output  = nn.Linear(2*numHidden, numActions)
        self.critic_output = nn.Linear(2*numHidden, 1)

    def forward(self, x):
        x = F.relu(self.shared_input(x))
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))

        logits = self.actor_output(x)
        value = self.critic_output(x)

        return logits, value