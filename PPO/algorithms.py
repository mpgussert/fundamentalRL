import torch
import torch.nn.functional as F
import numpy as np


def PPO_step_clipped(optimizer, agent, transitions, nSteps=3, epsilon=0.2):
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


#this is bad coding practice, I know, but it's less obfuscated this way
beta = 1.0
delta = 0.01
lastPolicyProb = None
def PPO_step_adaptive(optimizer, agent, transitions, nSteps=3, epsilon=0.2):
    global beta
    global delta
    global lastPolicyProb
    t = transitions

    #the detach call removes a pytorch tensor from the computational 
    #graph, which prevents that tensor from contribution to any gradients 
    #on that graph.  the returns are only bootstrapped with the value function, 
    #and we are not trying to ptimize that calculation, so we detach it.
    Rt = t._returns.detach()
    KLD = torch.nn.KLDivLoss(reduction='none')

    for i in range(nSteps):
        optimizer.zero_grad()
        currentLogits, currentValues = agent(t._states)

        #the advantage function is the difference between the "true"
        #value of a given state, and the value predicted by our critic.
        #if it's positive, it means we got more reward than we expected, 
        #and if it's negative then we made some mistakes and we got less
        #reward then we expected.
        advantages = Rt - currentValues

        currentPolicyProb = F.softmax(currentLogits,dim=1)
        initialPolicyProb = F.softmax(t._logits, dim=1).detach()
        
        ratio = currentPolicyProb.gather(1, transitions._actions) / initialPolicyProb.gather(1, transitions._actions)
        KLDiv = torch.sum(KLD(initialPolicyProb, currentPolicyProb), dim=1)

        ActorLoss = -torch.mean(ratio*advantages.detach() + beta*KLDiv)
        CriticLoss = torch.mean(advantages).pow(2)

        ActorLoss.backward(retain_graph=True)
        CriticLoss.backward()

        optimizer.step()
    

    if lastPolicyProb == None:
        lastPolicyProb = currentPolicyProb.detach()
    else:
        adaptiveTrigger = torch.mean(torch.sum(KLD(currentPolicyProb.detach(), lastPolicyProb),dim=1)).item()
        if adaptiveTrigger >= 1.5*delta:
            beta = 2.0*beta
            print("beta updated to", beta)
        elif adaptiveTrigger <= delta/1.5:
            beta = beta/2.0
            print("beta updated to", beta)
        lastPolicyProb = currentPolicyProb.detach()

    return ActorLoss.detach().item(), CriticLoss.detach().item()