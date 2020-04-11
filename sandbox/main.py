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


def fill_replay_separated(env, actor, critic, replay):
    obs = torch.from_numpy(env.reset()).float()
    total_reward = 0
    current_reward = 0
    num_episodes = 0
    for i in range(replay._capacity):
        s0 = obs
        action_logits = actor(obs)
        value = critic(obs)
        action = Categorical(logits = action_logits).sample()
        obs, rews, dones, infos = env.step(action.numpy())
        obs = torch.from_numpy(obs).float()
        total_reward += rews
        current_reward += rews
        replay.remember(s0, 
                       action_logits, 
                       action,  
                       rews,
                       value, 
                       dones)

        if dones:
            obs = torch.from_numpy(env.reset()).float()
            num_episodes += 1
            print(current_reward)
            current_reward=0
        env.render()

    final_value = critic(obs)
    replay.prep(final_value)  
    return total_reward/num_episodes      


if __name__=="__main__":
    env = gym.make('CartPole-v0')

    numObs = 4
    numActions = 2
    mem_length = 500
    replay = Memory(mem_length,4,2)

    obs = env.reset()
    env.render()

    actor = Actor(numObs, numActions)
    critic = Critic(numObs)

    lr = 1E-4
    lr_string = np.format_float_scientific(np.float(lr), precision=1)

    actor_optim = optim.Adam(actor.parameters(), lr)
    critic_optim = optim.Adam(critic.parameters(), lr)

    data = []

    for i in range(1000):
        mean_reward = fill_replay_separated(env, actor, critic, replay)
        a, c = A2C_step_separated(actor_optim, actor, critic_optim, critic, replay)
        print(i, mean_reward, a, c)
        data.append([i,mean_reward, a, c])
    
    np.savetxt("out_"+lr_string, np.array(data))
    