import torch
import torch.nn.functional as F
import numpy as np


def PPO_step(optimizer, agent, transitions, nSteps=3, batchSize=10, epsilon=0.2):
    for i in range(nSteps):
        batch = transitions.sample(batchSize)
        actor_terms=[]
        critic_terms=[]
        for (s0, logits, action, targetValue, s1, advantage, done) in batch:
            currentLogits, currentValue = agent(s0)
            r_t = currentLogits[action]/logits

            actor_terms.append(torch.min(r_t*advantage, torch.clamp(r_t, 1-epsilon, 1+epsilon)*advantage))
            critic_terms.append((current_value-targetValue).pow(2))

        Lactor = torch.mean(actor_terms)
        Lcritic = torch.mean(critic_terms)
        totalLoss = Lactor + Lcritic

        optimizer.zero_grad()
        optimizer.step()