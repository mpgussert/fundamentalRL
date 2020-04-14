import gym
import numpy as np

import torch
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
import torch.optim as optim

from model import Agent
from storage import Memory
from algorithms import PPO_step_clipped, PPO_step_adaptive

def fill_replay(env, agent, replay):
    """
    This function uses our agent and our environemnt
    to fill our replay buffer.
    """

    #initialize the env and convert the obs to torch
    obs = torch.from_numpy(env.reset()).float()

    #book keeping variables
    total_reward = 0
    current_reward = 0
    num_episodes = 0

    for i in range(replay._capacity):
        #see storage.py for details on the Memory class
        s0 = obs
        action_logits, value = agent(obs)

        """
        for cartpole, we have two actions: left and right.
        the output  of our neural network is just a vector of two numbers.
        we need to represent those numbers as probabilities, so they 
        must be on [0,1] and sum to 1. 
        
        now, we COULD just shift and normalize the output, but this is 
        bad because we would like larger outputs to correspond to more 
        confidence.  if we just shift and norm, then only the relative 
        ratio between the outputs will affect the confidence.  to get around 
        this we normalize after an exponential transform, which is known as the 
        Softmax funciton: prob[i] = exp(outputs[i])/sum(exp(outputs)).

        once we have this vector of probabilities, we want to sample it 
        to get the action.  we could have used np.random.choice with a 
        specific probability array, but the torch categorical distribution
        handles all the softmax crap for us internally, so we use that instead
        """
        action = Categorical(logits = action_logits).sample()
        obs, rews, dones, infos = env.step(action.numpy())
        obs = torch.from_numpy(obs).float()
        current_reward += rews

        #now that we have our transition, we store it for later
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

    """
    the infinite horizon stochastic return is defined by a sum over an 
    infinate number of time steps.  we obviously cannot do this, so we bootstrap
    the calculation using our value funciton to approximate the sum of the terms
    from N to infinity.
    """
    _, final_value = agent(obs)
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
        actorLoss, criticLoss = PPO_step_adaptive(optimizer, agent, replay)
        print("#####################################")
        print("epoch:      ", i)
        print("mean reward:", mean_reward)
        print("actor loss: ", actorLoss)
        print("critic loss ", criticLoss)
        print("#####################################")

    