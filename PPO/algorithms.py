import torch
import torch.nn.functional as F
import numpy as np


def PPO_step(optimizer, agent, transitions, nSteps=3, epsilon=0.2):
    t = transitions

    #the detach call removes a pytorch tensor from the computational 
    #graph, which prevents that tensor from contribution to any gradients 
    #on that graph.  the returns are only bootstrapped with the value function, 
    #and we are not trying to ptimize that calculation, so we detach it.
    Rt = t._returns.detach()

    for i in range(nSteps):
        optimizer.zero_grad()
        currentLogits, currentValues = agent(t._states)

        #the advantage function is the difference between the "true"
        #value of a given state, and the value predicted by our critic.
        #if it's positive, it means we got more reward than we expected, 
        #and if it's negative then we made some mistakes and we got less
        #reward then we expected.
        advantages = Rt - currentValues

        currentPolicyProb = F.softmax(currentLogits,dim=1).gather(1, transitions._actions)
        initialPolicyProb = F.softmax(t._logits, dim=1).gather(1, transitions._actions).detach()
        
        ratio = currentPolicyProb / initialPolicyProb
        clampedRatio = torch.clamp(ratio, 1-epsilon, 1+epsilon)

        ActorLoss = -torch.mean(torch.min(ratio,clampedRatio)*advantages.detach())
        CriticLoss = torch.mean(advantages).pow(2)

        ActorLoss.backward(retain_graph=True)
        CriticLoss.backward()

        optimizer.step()
    
    return ActorLoss.detach().item(), CriticLoss.detach().item()