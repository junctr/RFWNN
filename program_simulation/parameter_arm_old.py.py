# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 17:42:23 2020

@author: satoj
"""
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from math import pi 

import matplotlib
from matplotlib import pyplot as plt 

q = torch.ones(3,1)

q_dot = torch.ones(3,1)

q1 = q[0]

q2 = q[1]

q3 = q[2]

q1_dot = q_dot[0]

q2_dot = q_dot[1]

q3_dot = q_dot[2]


'''
q1 = torch.tensor(pi)

q2 = torch.tensor(pi)

q3 = torch.tensor(pi)

q1_dot = torch.tensor(pi)

q2_dot = torch.tensor(pi)

q3_dot = torch.tensor(pi)
'''

g = 10

p1 = 4

p2 = 3

p3 = 1.5

l1 = 0.4

l2 = 0.3

l3 = 0.2


M11 = l1**2 * (p1 + p2) + p2 * l2**2 + 2 * p1 * p2 * torch.cos(q2)

M12 = p2 * l2**2 + p2 * l1 * l2 * torch.cos(q2)

M13 = M23 = M31 = M32 = 0

M21 = M12

M22 = p2 * l2**2

M33 = p3

C11 = -p2 * l1 * l2 * (2 * q1_dot * q2_dot + q2_dot**2) * torch.sin(q2)

C21 = p2 * l1 * l2 * q1_dot**2 * torch.sin(q2)

C12 = C13 = C22 = C23 = C31 = C32 = C33 = 0

g1 = (p1 + p2) * g * l1 * torch.cos(q1) + p2 * g * l2 * torch.cos(q1 + q2)

g2 = p2 * g * l2 * torch.cos(q1 + q2)

g3 = -p3 * g


M = torch.tensor([[M11, M12, M13],
                  [M21, M22, M23],
                  [M31, M32, M33]])

C = torch.tensor([[C11, C12, C13],
                  [C21, C22, C23],
                  [C31, C32, C33]])

G = torch.tensor([[g1],
                  [g2], 
                  [g3]])

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