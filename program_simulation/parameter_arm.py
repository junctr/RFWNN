# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:52:20 2020

@author: satoj
"""

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from math import pi 

import matplotlib
from matplotlib import pyplot as plt 

q = torch.zeros([3,1], dtype=torch.float32)

q_dot = torch.zeros([3,1], dtype=torch.float32)

M = torch.zeros([3,3], dtype=torch.float32)

C = torch.zeros([3,3], dtype=torch.float32)

G = torch.zeros([3,1], dtype=torch.float32)


g = 10

p1 = 4

p2 = 3

p3 = 1.5

l1 = 0.4

l2 = 0.3

l3 = 0.2


M[0][0] = l1**2 * (p1 + p2) + p2 * l2**2 + 2 * p1 * p2 * torch.cos(q[1])

M[0][1] = p2 * l2**2 + p2 * l1 * l2 * torch.cos(q[1])

M[1][0] = M[0][1]

M[1][1] = p2 * l2**2

M[2][2] = p3

C[0][0] = -p2 * l1 * l2 * (2 * q_dot[0] * q_dot[1] + q_dot[1]**2) * torch.sin(q[1])

C[1][0] = p2 * l1 * l2 * q_dot[0]**2 * torch.sin(q[1])

G[0] = (p1 + p2) * g * l1 * torch.cos(q[0]) + p2 * g * l2 * torch.cos(q[0] + q[1])

G[1] = p2 * g * l2 * torch.cos(q[0] + q[1])

G[2] = -p3 * g

print(q)
print(q.dtype)
print(q.size())

print(q_dot)
print(q_dot.dtype)
print(q_dot.size())

print(M)
print(M.dtype)
print(M.size())

print(C)
print(C.dtype)
print(C.size())

print(G)
print(G.dtype)
print(G.size())