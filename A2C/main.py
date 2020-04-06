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
    obs = torch.from_numpy(env.reset()).float()
    total_reward = 0
    current_reward = 0
    num_episodes = 0
    for i in range(replay._capacity):
        s0 = obs
        action_logits, value = agent(obs)
        action = Categorical(logits = action_logits).sample()
        obs, rews, dones, infos = env.step(action.numpy())
        obs = torch.from_numpy(obs).float()
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
            total_reward += current_reward
            print("episode "+str(num_episodes)+":", current_reward)
            current_reward=0
        env.render()

    final_value = critic(obs)
    replay.compute_returns(final_value)  
    return total_reward/num_episodes    

if __name__=="__main__":
    """
    cartPole-v0 is the environment.  openai often registers multiple
    environments with slightly different parameters as different versions
    of that environment (in this case, v0 terminates after 200 steps, while
    v1 terminates after 500)
    """
    env = gym.make('CartPole-v0')

    """
    these can be exctracted from a general environment using 
    openai spaces. in a general algorithm you would have a 
    typecheck here for the space and then extract the 
    shape from that space.  
    """
    numObs = 4
    numActions = 2

    """
    the replay length in A2C is a hyperparameter.
    for simplicitly, I do not preform batching in this
    example but often times you will need to specify 
    a batch and minibatch size as well
    """
    memLength = 500
    hiddenSize = 32

    """
    A2C is an "offline" algorithm, which means
    we collect transitions and then we batch train
    our models on that dataset like we would any other
    """
    replay = Memory(memLength,4,2)
    obs = env.reset()
    env.render()
    
    """
    the optimizer is generally a hyperparameter of a 
    given RL algorithm. Adam is a common choice but 
    it has stability issues in more complex environments. 
    SGD is usually a safer bet.  1E-2 is an extremely 
    fast learning rate, but we can get away with it for
    this environment
    """
    agent = Agent(numObs, numActions, hiddenSize)
    optimizer = optim.Adam(agent.parameters(), 1E-2)

    for i in range(1000):
        """
        we are going to execute 1000 epochs of training, 
        though we really dont need to for cartpole.  in 
        each epoch, we collect our transitions by having 
        the agent interact with the environment, and then
        we update the parameters of the agent
        """
        mean_reward = fill_replay(env, agent, replay)
        actorLoss, criticLoss = A2C_step(optimizer, agent, replay)
        print("#####################################")
        print("epoch:      ", i)
        print("mean reward:", mean_reward)
        print("actor loss: ", actorLoss)
        print("critic loss ", criticLoss)

    