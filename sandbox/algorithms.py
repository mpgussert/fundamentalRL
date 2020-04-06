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

def A2C_step(optimizer, agent, transitions):
    actor_terms = (transitions._returns - transitions._values).detach()*F.log_softmax(transitions._logits)[transitions._actions]
    critic_terms = (transitions._returns.detach() - transitions._values).pow(2)
    
    Lactor = torch.mean(actor_terms)
    Lcritic = torch.mean(critic_terms)
    Ltotal = Lactor + Lcritic
    
    optimizer.zero_grad()
    Ltotal.backward()
    #Lactor.backward()
    #Lcritic.backward()
    optimizer.step()

    actorLoss = Lactor.detach().item()
    criticLoss = Lcritic.detach().item()

    return actorLoss, criticLoss

def A2C_step_separated(actor_optim, actor, critic_optim, critic, transitions):

    actor_optim.zero_grad()
    critic_optim.zero_grad()

    Rt = transitions._returns.detach()
    Advantages = Rt - transitions._values

    ActorLoss = -torch.mean(Advantages.detach()*F.log_softmax(transitions._logits, dim=1).gather(1, transitions._actions))
    
    func = torch.nn.MSELoss()
    CriticLoss = func(Rt, transitions._values)
    #CriticLoss = Advantages.pow(2).mean()

    ActorLoss.backward()
    CriticLoss.backward()

    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)

    actor_optim.step()
    critic_optim.step()

    #return 0 , CriticLoss.detach().item()
    return ActorLoss.detach().item(), CriticLoss.detach().item()