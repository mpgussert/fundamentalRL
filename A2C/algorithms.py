import torch
import torch.nn.functional as F
import numpy as np

def A2C_step(optimizer, agent, transitions):
    optimizer.zero_grad()

    Rt = transitions._returns.detach()
    Advantages = Rt - transitions._values

    ActorLoss = -torch.mean(Advantages.detach()*F.log_softmax(transitions._logits, dim=1).gather(1, transitions._actions))
    CriticLoss = torch.mean(Advantages.pow(2))

    ActorLoss.backward()
    CriticLoss.backward()

    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)

    optimizer.step()

    return ActorLoss.detach().item(), CriticLoss.detach().item()