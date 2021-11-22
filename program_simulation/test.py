# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 17:54:57 2020

@author: satoj
"""

import torch
from math import pi 

x = torch.tensor(pi, dtype=torch.float)

print(x)
print(x.dtype)
print(x.size())

a = torch.cos(x)

print(a)
print(a.dtype)
print(a.size())

b = -10

M13 = M23  = -b

print(M13, M23)

y = torch.zeros([1,3])

y[0][1] = 1

print(y)
print(y[0][1])
print(y.dtype)
print(y.size())