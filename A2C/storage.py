import numpy as np

import torch

import random
from collections import deque, namedtuple
from typing import NewType


class Memory:
    def __init__(self, capacity, numObs, numActions):
        super(Memory, self).__init__()
        self._capacity = capacity
        self._numObs = numObs
        self._numActions = numActions
        self._states = torch.zeros((self._capacity, numObs)).float()
        self._logits = torch.zeros((self._capacity, numActions)).float()
        self._actions = torch.zeros((self._capacity, 1)).long()
        self._rewards = torch.zeros((self._capacity, 1)).float()
        self._values  = torch.zeros((self._capacity, 1)).float()
        self._returns = torch.zeros((self._capacity, 1)).float()
        self._dones = torch.zeros((self._capacity, 1)).int()
        self._gamma = 0.99
        self._currentIndex = 0


    def remember(self, state, logits, action, reward, value, done):
        if self._currentIndex == self._capacity:
            print("memory is full!  cannot allocate")
            return
        if self._currentIndex == 0:
            self._states = torch.zeros((self._capacity, self._numObs)).float()
            self._logits = torch.zeros((self._capacity, self._numActions)).float()
            self._actions = torch.zeros((self._capacity, 1)).long()
            self._rewards = torch.zeros((self._capacity, 1)).float()
            self._values  = torch.zeros((self._capacity, 1)).float()
            self._returns = torch.zeros((self._capacity, 1)).float()
            self._dones = torch.zeros((self._capacity, 1)).int()

        self._states[self._currentIndex] = state
        self._logits[self._currentIndex] = logits
        self._actions[self._currentIndex] = action
        self._rewards[self._currentIndex] = reward
        self._values[self._currentIndex] = value
        self._dones[self._currentIndex]  = done
        self._currentIndex+=1

    def compute_returns(self, final_value):
        #calculate stochastic returns. advantage here too?
        self._returns[-1] = final_value
        for t in reversed(range(self._capacity-1)):
            r_0 = self._rewards[t]
            R_1 = self._returns[t+1]
            mask = 1 - self._dones[t]

            R_0 = r_0 + self._gamma*mask*R_1
            self._returns[t] = R_0
        self._currentIndex=0


