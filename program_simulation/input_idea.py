# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 17:14:10 2020

@author: satoj
"""
import numpy as np
import time
#import scipy.integrate as sp
from matplotlib import pyplot as plt


def M(t, q, p, l):

    M = np.zeros((3,3), dtype=np.float64)

    M[0][0] = l[0]**2 * (p[0] + p[1]) + p[1] * (l[1]**2 + 2 * l[0] * l[1] * np.cos(q[1][0]))
    M[0][1] = p[1] * l[1]**2 + p[1] * l[0] * l[1] * np.cos(q[1][0])
    M[1][0] = M[0][1]
    M[1][1] = p[1] * l[1]**2
    M[2][2] = p[2]

    return M

def C(t, q, p, l):

    C = np.zeros((3,3), dtype=np.float64)

    C[0][0] = -p[1] * l[0] * l[1] * (2 * q[0][1] * q[1][1] + q[1][1]**2) * np.sin(q[1][0])
    C[1][0] = p[1] * l[0] * l[1] * q[0][1]**2 * np.sin(q[1][0])

    return C

def G(t, q, p, l, g):
    
    G = np.array([
        [(p[0] + p[1]) * g * l[0] * np.cos(q[0][0]) + p[1] * g * l[1] * np.cos(q[0][0] + q[1][0])],
        [p[1] * g * l[1] * np.cos(q[0][0] + q[1][0])],
        [-p[2] * g]],
        dtype=np.float64
    )

    return G

def F(t, q):

    F = np.array([
        [5*q[0][1] + 0.2 * np.sign(q[0][1])],
        [5*q[1][1] + 0.2 * np.sign(q[1][1])],
        [5*q[2][1] + 0.2 * np.sign(q[2][1])]],
        dtype=np.float64
    )

    return F
    
def qd(t):

    qd = np.array([
        [0.5*np.sin(2*np.pi*t), np.pi*np.cos(2*np.pi*t)], 
        [0.5*np.sin(2*np.pi*t), np.pi*np.cos(2*np.pi*t)], 
        [0.5*np.sin(2*np.pi*t), np.pi*np.cos(2*np.pi*t)]],
        dtype=np.float64
    )

    return qd

def tau(t):
    
    """
    tau0 = np.array([
        [2*np.sin(np.pi*t)],
        [2*np.sin(np.pi*t)],
        [2*np.sin(np.pi*t)]],
        dtype=np.float64
    )
    """
    
    tau0 = np.array([
        [2*np.sin(10 *2*np.pi*t)],
        [2*np.sin(10 *2*np.pi*t)],
        [2*np.sin(10 *2*np.pi*t)]],
        dtype=np.float64
    )
    
    #tau0 = np.zeros((3,1),dtype=np.float64)

    qd_dotdot = np.array([
        [-2*(np.pi**2)*np.sin(2*np.pi*t)], 
        [-2*(np.pi**2)*np.sin(2*np.pi*t)], 
        [-2*(np.pi**2)*np.sin(2*np.pi*t)]],
        dtype=np.float64
    )

    tau = M(t,qd(t),p,l) @ qd_dotdot + np.dot(C(t,qd(t),p,l), qd(t)[:,[1]]) + G(t,qd(t),l,p,g) + F(t,qd(t)) + tau0
    #tau = np.dot(C(t,qd(t),p,l), qd(t)[:,[1]]) + G(t,qd(t),l,p,g) + F(t,qd(t)) + tau0
    #tau = M(t,qd(t),p,l) @ qd_dotdot + G(t,qd(t),l,p,g) + F(t,qd(t)) + tau0
    #tau = M(t,qd(t),p,l) @ qd_dotdot + np.dot(C(t,qd(t),p,l), qd(t)[:,[1]]) + F(t,qd(t)) + tau0
    #tau = M(t,qd(t),p,l) @ qd_dotdot + np.dot(C(t,qd(t),p,l), qd(t)[:,[1]]) + G(t,qd(t),l,p,g) + tau0
    #tau = M(t,qd(t),p,l) @ qd_dotdot + np.dot(C(t,qd(t),p,l), qd(t)[:,[1]]) + G(t,qd(t),l,p,g) + F(t,qd(t))
    return tau


p = np.array([4, 3, 1.5])
l = np.array([0.4, 0.3, 0.2])
g = 10


t = 0.0
end = 2
step = 0.0001
i = 0


e1_data = []
e2_data = []
e3_data = []
t_data = []


start = time.time()

while t < end:

    e1_data.append(tau(t)[0])
    e2_data.append(tau(t)[1])
    e3_data.append(tau(t)[2])
    t_data.append(t)

    t += step
    i += 1
    if i%1000 == 0:
        print(round(100*t/end, 1),"%",i,round(time.time()-start, 1),"s")



fig = plt.figure()

ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

ax1.plot(t_data, e1_data)
ax2.plot(t_data, e2_data)
ax3.plot(t_data, e3_data)

ax1.grid()
ax2.grid()
ax3.grid()

fig.tight_layout()

plt.title("input_idea")
plt.show()