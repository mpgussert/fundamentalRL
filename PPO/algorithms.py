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

        # PPO is just TRPO but the trust region line search is rephrased as a lagrangian 
        # constraint problem subject to the KL divergence (the boundary of the trust region).
        # In the experiments section of the paper they perform this optimization but they find 
        # that a constnt lagrange multiplier isn't sufficient, and a non constant multiplier 
        # implies the need for second order methods, which we want to avoid.
        #
        # Instead, PPO optimizes over a surrogate objective based on the clipped ratio between the 
        # current (just updated) policy, and the initial policy for a given replay buffer.
        #
        # The line of reasoning here is that we want to make the actions that gave us more 
        # reward than we expected, more probable through optimization so first we get all the 
        # probabilities for the actions we took... 
        currentPolicyProb = F.softmax(currentLogits,dim=1).gather(1, transitions._actions)
        initialPolicyProb = F.softmax(t._logits, dim=1).gather(1, transitions._actions).detach()
        
        #and we compute and clip the ratio.
        ratio = currentPolicyProb / initialPolicyProb
        clampedRatio = torch.clamp(ratio, 1-epsilon, 1+epsilon)

        # if the action we took was benificial, we want to take it more often, so we 
        # scale by the advantage (be sure to detach the advantage here as we are updating 
        # that using the critic model)
        ActorLoss = -torch.mean(torch.min(ratio,clampedRatio)*advantages.detach())
        CriticLoss = torch.mean(advantages).pow(2)

        # we are going to make nSteps of these tiny updates every PPO step, so we
        # want to retain the compute
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

        # this is the KL adaptive step mentioned in the clipped implementation.  Basically you just 
        # examine the expectation the KL divergence over the buffer and depending on it's magnitude, you rescale 
        # the multiplier as required.
        currentPolicyProb = F.softmax(currentLogits,dim=1)
        initialPolicyProb = F.softmax(t._logits, dim=1).detach()
        
        # same idea here but the gather here is moved because we need the full policy to 
        # to compute our KL divergence.
        ratio = currentPolicyProb.gather(1, transitions._actions) / initialPolicyProb.gather(1, transitions._actions)
        KLDiv = torch.sum(KLD(initialPolicyProb, currentPolicyProb), dim=1)

        # beta is our lagrange multiplier.  As we update our model we are going to keep 
        # track of the KL divergence and updated it as required.
        ActorLoss = -torch.mean(ratio*advantages.detach() + beta*KLDiv)
        CriticLoss = torch.mean(advantages).pow(2)

        ActorLoss.backward(retain_graph=True)
        CriticLoss.backward()

        optimizer.step()
    

    # I don't know if I did this part right... need to double check
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