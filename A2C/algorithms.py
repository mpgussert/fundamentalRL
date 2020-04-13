import torch
import torch.nn.functional as F
import numpy as np

def A2C_step(optimizer, agent, transitions):
    """
    execute a single A2C update step
    """

    #first, we clear the gradients being tracked by the optimizer.
    #we can do this anywhere so long as it's before the backward call.
    optimizer.zero_grad()

    #the detach call removes a pytorch tensor from the computational 
    #graph, which prevents that tensor from contribution to any gradients 
    #on that graph.  the returns are only bootstrapped with the value function, 
    #and we are not trying to ptimize that calculation, so we detach it.
    Rt = transitions._returns.detach()

    #the advantage function is the difference between the "true"
    #value of a given state, and the value predicted by our critic.
    #if it's positive, it means we got more reward than we expected, 
    #and if it's negative then we made some mistakes and we got less
    #reward then we expected.
    Advantages = Rt - transitions._values

    #calculate the loss functions for both the actor and the critic. the 
    #advantage is detached here because the gradient must operate only 
    #on the policy.  log_softmax is just log(softmax(x)), and the gather call
    #is used to intelligently index the log probabilities based on the actions taken
    #(torch does not have intelligent indexing like numpy in certain cases).  Note the negative 
    #sign on the actor loss.  this is becuase we are trying to MAXIMIZE the actor "loss", and
    # our optimizers are all working to minimize whatever you call backward() on.
    ActorLoss = -torch.mean(Advantages.detach()*F.log_softmax(transitions._logits, dim=1).gather(1, transitions._actions))
    CriticLoss = torch.mean(Advantages.pow(2))

    #the backward call propogates the gradient backwards 
    #through the computational graph from the loss function to 
    #the non detached variable leaves.
    ActorLoss.backward()
    CriticLoss.backward()

    #the gradient clipping here is optional, but it helps with stability.
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)

    #step the optimizer once.  this adds the calculated gradients 
    #to the parameters in in the model based on the selected 
    #optimization algorithm
    optimizer.step()

    return ActorLoss.detach().item(), CriticLoss.detach().item()

def A2C_step_shared(optimizer, agent, transitions):
    """
    execute a single A2C step for a agent with shared parameters.
    I prefer to avoid chimera models, as there hasn't been anything
    other than handwaving arguments for why doing it might be 
    benificial.
    """
    alpha = 0.001
    beta  = 0.001
    optimizer.zero_grad()

    Rt = transitions._returns.detach()

    Advantages = Rt - transitions._values

    ActorLoss = -torch.mean(Advantages.detach()*F.log_softmax(transitions._logits, dim=1).gather(1, transitions._actions))
    EntropyLoss = -torch.mean(torch.sum(F.softmax(transitions._logits, dim=1)*F.log_softmax(transitions._logits, dim=1),dim=1))
    CriticLoss = beta*torch.mean(Advantages.pow(2))

    TotalActorLoss = ActorLoss + alpha*EntropyLoss
    
    TotalActorLoss.backward(retain_graph=True)
    CriticLoss.backward()

    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)

    optimizer.step()

    return ActorLoss.detach().item(), CriticLoss.detach().item()