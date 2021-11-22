# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 17:14:10 2020

@author: satoj
"""
import numpy as np
import scipy.integrate as sp
from matplotlib import pyplot as plt

t = 10
q = np.ones((3,2), dtype=np.float32)
#tau = np.ones((3,1), dtype=np.float32)

W = np.ones((5,3), dtype=np.float32)
odot = np.ones((5,3), dtype=np.float32)
co = np.ones((5,3), dtype=np.float32)
ro = np.ones((5,3), dtype=np.float32)



alpha_w = 50 * np.identity(5, dtype=np.float32)
alpha_odot = 20 * np.identity(45, dtype=np.float32)
alpha_co = 20 * np.identity(45, dtype=np.float32)
alpha_ro = 20 * np.identity(45, dtype=np.float32)
alpha_beta = 0.001 * np.identity(5, dtype=np.float32)
alpha_zeta = 0.1



p = np.array([4, 3, 1.5])
l = np.array([0.4, 0.3, 0.2])
g = 10

def system(t, q, p, l, g, tau):
    
    dq = np.empty((3, 2), dtype=np.float32)

    #dq = [[q1,q1_dot],[q2,q2_dot],[q3,q3_dot]]
    
    dq[:,[0]] = q[:,[1]]
    dq[:,[1]] = np.dot(np.linalg.inv(M(t,q,p,l)), tau - tau0(t) - np.dot(C(t,q,p,l), q[:,[1]]) - G(t,q,l,p,g) - F(t,q))

    dx = dq.T.reshape(1,-1)

    return dx

def simulation():

    return


def M(t, q, p, l):

    M = np.zeros((3,3), dtype=np.float32)

    M[0][0] = l[0]**2 * (p[0] + p[1]) + p[1] * l[1]**2 + 2 * p[0] * p[1] * np.cos(q[1][0])
    M[0][1] = p[1] * l[1]**2 + p[1] * l[0] * l[1] * np.cos(q[1][0])
    M[1][0] = M[0][1]
    M[1][1] = p[1] * l[1]**2
    M[2][2] = p[2]

    return M

def C(t, q, p, l):

    C = np.zeros((3,3), dtype=np.float32)

    C[0][0] = -p[1] * l[0] * l[1] * (2 * q[0][1] * q[1][1] + q[1][1]**2) * np.sin(q[1][0])
    C[1][0] = p[1] * l[0] * l[1] * q[0][1]**2 * np.sin(q[1][0])

    return C

def G(t, q, p, l, g):
    
    G = np.array([
        [(p[0] + p[1]) * g * l[0] * np.cos(q[0][0]) + p[1] * g * l[1] * np.cos(q[0][0] + q[1][0])],
        [p[1] * g * l[1] * np.cos(q[0][0] + q[1][0])],
        [-p[2] * g]],
        dtype=np.float32
    )

    return G

def F(t, q):

    F = np.array([
        [5*q[0][1] + 0.2 * np.sign(q[0][1])],
        [5*q[1][1] + 0.2 * np.sign(q[1][1])],
        [5*q[2][1] + 0.2 * np.sign(q[2][1])]],
        dtype=np.float32
    )

    return F
    
def qd(t):

    qd = np.array([
        [0.5*np.sin(2*np.pi*t), np.pi*np.cos(2*np.pi*t)], 
        [0.5*np.sin(2*np.pi*t), np.pi*np.cos(2*np.pi*t)], 
        [0.5*np.sin(2*np.pi*t), np.pi*np.cos(2*np.pi*t)]],
        dtype=np.float32
    )

    return qd

def tau0(t):

    tau0 = np.array([
        [2*np.sin(np.pi*t)],
        [2*np.sin(np.pi*t)],
        [2*np.sin(np.pi*t)]],
        dtype=np.float32
    )

    return tau0

def e_f(t, q):

    e = np.empty((3,2), dtype=np.float32)

    e = qd(t) - q

    return e

def s(t, q):

    s = np.empty((3,1), dtype=np.float32)

    s = e(t,q)[:,[1]] + np.dot(5 * np.identity(3, dtype=np.float32), e(t,q)[:,[0]])

    return s

def x_f(t, q, s):

    x = np.empty((15,1), dtype=np.float32)

    x = np.concatenate([q.T.reshape(-1,1), qd(t).T.reshape(-1,1), s])

    return x

def xji_f(t, x, xold, odot, co, ro):

    xji = np.empty((15,5), dtype=np.float32)

    xji = x + odot * np.exp(A_f(t,xold,co,ro))

    return xji

def A_f(t, xji, co, ro):

    A = np.empty((15,5), dtype=np.float32)

    A = -(co**2) * ((xji - ro)**2)

    return A

def mu(A):

    mu = np.empty((1,5), dtype=np.float32)

    mu = np.prod(1 + A * np.exp(A), axis=0)

    return mu

def y(A, W):

    y = np.empty((3,1), dtype=np.float32)

    y = np.dot(W.T, mu(A).T)

    return y

def taus(s, beta, zeta, omega):

    taus = ((np.dot(beta.T, omega))**2 / (np.linalg.norm(s) * np.dot(beta.T, omega) + zeta)) * s

    return taus

def tau(s, A, beta, zeta, omega):
    
    tau = np.empty((3,1), dtype=np.float32)

    K = 5 * np.identity(3, dtype=np.float32)

    tau = taus(s,beta,zeta,omega) + np.dot(K, s) + y(A,W)

    return tau

def omega(odot, co, ro, W):

    omega = np.array([
        [1],
        [np.linalg.norm(odot)],
        [np.linalg.norm(co)],
        [np.linalg.norm(ro)],
        [np.linalg.norm(W)]],
        dtype=np.float32
    )

    return omega







print("M:")
print(M(t,q,p,l))
print(M(t,q,p,l).shape)
print(M(t,q,p,l).dtype)

print("C:")
print(C(t,q,p,l))
print(C(t,q,p,l).shape)
print(C(t,q,p,l).dtype)

print("G:")
print(G(t,q,p,l,g))
print(G(t,q,p,l,g).shape)
print(G(t,q,p,l,g).dtype)

print("F:")
print(F(t,q))
print(F(t,q).shape)
print(F(t,q).dtype)

print("qd:")
print(qd(t))
print(qd(t).shape)
print(qd(t).dtype)

print("tau0:")
print(tau0(t))
print(tau0(t).shape)
print(tau0(t).dtype)

print("system:")
print(system(t,q,p,l,g,tau))
print(system(t,q,p,l,g,tau).shape)
print(system(t,q,p,l,g,tau).dtype)
"""
print(":")
print()
print(.shape)
print(.dtype)
"""








    
    
    
    