import gym
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
import torch.optim as optim

import argparse

from model import Agent, Actor, Critic
from storage import Memory
from algorithms import A2C_step, A2C_step_separated


def fill_replay(env, agent, replay):
    obs = torch.from_numpy(np.array(env.env.state)).float()
    for i in range(replay._capacity):
        s0 = obs
        action_logits, value = agent(obs)
        action = Categorical(logits = action_logits).sample()
        obs, rews, dones, infos = env.step(action.numpy())
        obs = torch.from_numpy(obs).float()
        if(dones):
            rews = -1

        replay.remember(s0, 
                       action_logits, 
                       action,  
                       rews,
                       value, 
                       dones)

        if dones:
            obs = torch.from_numpy(env.reset()).float()
        env.render()
    _, final_value = agent(obs)
    replay.prep(final_value)

def fill_replay_separated(env, actor, critic, replay):
    obs = torch.from_numpy(np.array(env.env.state)).float()
    for i in range(replay._capacity):
        s0 = obs
        action_logits = actor(obs)
        value = critic(obs)
        action = Categorical(logits = action_logits).sample()
        obs, rews, dones, infos = env.step(action.numpy())
        obs = torch.from_numpy(obs).float()
        if(dones):
            rews = -1

        replay.remember(s0, 
                       action_logits, 
                       action,  
                       rews,
                       value, 
                       dones)

        if dones:
            obs = torch.from_numpy(env.reset()).float()
        env.render()
    if not dones:
     final_value = critic(obs)
    else:
        final_value = -1
    replay.prep(final_value)        


if __name__=="__main__":
    env = gym.make('CartPole-v0')
    #figure out how to extract this from spaces.  need to test for type...
    numObs = 4
    numActions = 2
    mem_length = 500
    replay = Memory(mem_length,4,2)

    obs = env.reset()
    env.render()

    actor = Actor(numObs, numActions)
    critic = Critic(numObs)

    actor_optim = optim.Adam(actor.parameters(), 1E-2)
    critic_optim = optim.Adam(critic.parameters(), 1E-2)

    while True:
        fill_replay_separated(env, actor, critic, replay)
        a, c = A2C_step_separated(actor_optim, actor, critic_optim, critic, replay)
        print(a, c)
    