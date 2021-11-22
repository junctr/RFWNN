# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 17:14:10 2020

@author: satoj
"""
import numpy as np
import time
#import scipy.integrate as sp
from matplotlib import pyplot as plt


def system(t, q, p, l, g, tau):
    
    dq = np.empty((3, 2), dtype=np.float64)

    #dq = [[q1,q1_dot],[q2,q2_dot],[q3,q3_dot]]
    
    dq[:,[0]] = q[:,[1]]
    dq[:,[1]] = np.linalg.inv(M(t,q,p,l)) @ (tau - tau0(t) - np.dot(C(t,q,p,l), q[:,[1]]) - G(t,q,l,p,g) - F(t,q))

    return dq

def M(t, q, p, l):

    M = np.zeros((3,3), dtype=np.float64)
    
    M[0][0] = l[0]**2 * (p[0] + p[1]) + p[1] * (l[1]**2 + 2 * l[0] * l[1] * np.cos(q[1][0]))
    #M[0][0] = l[0]**2 * (p[0] + p[1]) + p[1] * l[1]**2 + 2 * p[0] * p[1] * np.cos(q[1][0])
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

def tau0(t):

    tau0 = np.array([
        [2*np.sin(10 *2*np.pi*t)],
        [2*np.sin(10 *2*np.pi*t)],
        [2*np.sin(10 *2*np.pi*t)]],
        dtype=np.float64
    )
    """
    tau0 = np.array([
        [2*np.sin(np.pi*t)],
        [2*np.sin(np.pi*t)],
        [2*np.sin(np.pi*t)]],
        dtype=np.float64
    )
    
    tau0 = np.array([
        [2*np.sin(np.pi*t) + 20*np.sin(20*t)],
        [2*np.sin(np.pi*t) + 20*np.sin(20*t)],
        [2*np.sin(np.pi*t) + 20*np.sin(20*t)]],
        dtype=np.float64
    )
    """
    return tau0

def e_f(t, q):

    e = np.empty((3,2), dtype=np.float64)

    e = qd(t) - q

    return e

def s_f(e):

    s = np.empty((3,1), dtype=np.float64)

    s = e[:,[1]] + ((5 * np.identity(3, dtype=np.float64)) @ e[:,[0]])

    return s

def x_f(t, q, s):

    x = np.empty((15,1), dtype=np.float64)

    x = np.concatenate([q.T.reshape(-1,1), qd(t).T.reshape(-1,1), s])

    return x

def xji_f(t, x, xold, odot, co, ro):

    xji = np.empty((15,5), dtype=np.float64)

    xji = x + odot * np.exp(A_f(t,xold,co,ro))

    return xji

def A_f(t, xji, co, ro):

    A = np.empty((15,5), dtype=np.float64)

    A = -(co**2) * ((xji - ro)**2)

    return A

def mu_f(A):

    mu = np.empty((1,5), dtype=np.float64)

    mu = np.prod((1 + A) * np.exp(A), axis=0)

    #muji = (1 + A) * np.exp(A)

    return mu

def muji_f(A):

    muji = (1 + A) * np.exp(A)

    return muji

def y(A, W):

    #y = np.empty((3,1), dtype=np.float64)

    y = W.T @ mu_f(A).reshape(5,1)

    return y

def omega_f(odot, co, ro, W):

    omega = np.array([
        [1],
        [np.linalg.norm(odot)],
        [np.linalg.norm(co)],
        [np.linalg.norm(ro)],
        [np.linalg.norm(W)]],
        dtype=np.float64
    )

    return omega

def taus(s, beta, zeta, omega):

    taus = ((beta.T @ omega)**2 / (np.linalg.norm(s) * beta.T @ omega + zeta)) * s

    return taus

def tau_f(s, A, beta, zeta, omega):
    
    tau = np.empty((3,1), dtype=np.float64)

    K = 100 * np.identity(3, dtype=np.float64)

    tau = taus(s,beta,zeta,omega) + K @ s + y(A,W)
    #tau = y(A,W)
    #tau = taus(s,beta,zeta,omega) + K @ s

    return tau    

def B_f(x, Aold, odot, ro):

    B = x + odot * np.exp(Aold) - ro

    return B    

def bk_f(mu, muji, A, Aold, B, co):

    bk = np.empty((5,75), dtype=np.float64)

    dmuji =(2 + A) * np.exp(A) *(-2 * co**2 * np.exp(Aold) * B)

    #x = mu * dmuji / muji
    x = mu * np.divide(dmuji, muji, out=np.zeros_like(dmuji), where=muji!=0)

    zeros0 = np.zeros((15,5), dtype=np.float64) 
    zeros1 = np.zeros((15,5), dtype=np.float64) 
    zeros2 = np.zeros((15,5), dtype=np.float64) 
    zeros3 = np.zeros((15,5), dtype=np.float64) 
    zeros4 = np.zeros((15,5), dtype=np.float64) 

    zeros0[:,[0]] = x[:,[0]]
    zeros1[:,[1]] = x[:,[1]]
    zeros2[:,[2]] = x[:,[2]]
    zeros3[:,[3]] = x[:,[3]]
    zeros4[:,[4]] = x[:,[4]]

    bk[0] = zeros0.T.reshape(1,-1)
    bk[1] = zeros1.T.reshape(1,-1)
    bk[2] = zeros2.T.reshape(1,-1)
    bk[3] = zeros3.T.reshape(1,-1)
    bk[4] = zeros4.T.reshape(1,-1)

    return bk.T

def ek_f(mu, muji, A, Aold, B ,odot, co, ro, xold):

    ek = np.empty((5,75), dtype=np.float64)

    dmuji =(2 + A) * np.exp(A) *(-2 * co * B**2 -2 * co**2 * B *(-2 * odot * co * (xold - ro)**2 ) * np.exp(Aold))

    #x = mu * dmuji / muji
    x = mu * np.divide(dmuji, muji, out=np.zeros_like(dmuji), where=muji!=0)

    zeros0 = np.zeros((15,5), dtype=np.float64) 
    zeros1 = np.zeros((15,5), dtype=np.float64) 
    zeros2 = np.zeros((15,5), dtype=np.float64) 
    zeros3 = np.zeros((15,5), dtype=np.float64) 
    zeros4 = np.zeros((15,5), dtype=np.float64) 

    zeros0[:,[0]] = x[:,[0]]
    zeros1[:,[1]] = x[:,[1]]
    zeros2[:,[2]] = x[:,[2]]
    zeros3[:,[3]] = x[:,[3]]
    zeros4[:,[4]] = x[:,[4]]

    ek[0] = zeros0.T.reshape(1,-1)
    ek[1] = zeros1.T.reshape(1,-1)
    ek[2] = zeros2.T.reshape(1,-1)
    ek[3] = zeros3.T.reshape(1,-1)
    ek[4] = zeros4.T.reshape(1,-1)

    return ek.T

def gk_f(mu, muji, A, Aold, B ,odot, co, ro):

    gk = np.empty((5,75), dtype=np.float64)

    dmuji =(2 + A) * np.exp(A) *(-2 * co**2 * B * (-1 -2 * odot * co**2 * ro * np.exp(Aold)))

    #x = mu * dmuji / muji
    x = mu * np.divide(dmuji, muji, out=np.zeros_like(dmuji), where=muji!=0)

    zeros0 = np.zeros((15,5), dtype=np.float64) 
    zeros1 = np.zeros((15,5), dtype=np.float64) 
    zeros2 = np.zeros((15,5), dtype=np.float64) 
    zeros3 = np.zeros((15,5), dtype=np.float64) 
    zeros4 = np.zeros((15,5), dtype=np.float64) 

    zeros0[:,[0]] = x[:,[0]]
    zeros1[:,[1]] = x[:,[1]]
    zeros2[:,[2]] = x[:,[2]]
    zeros3[:,[3]] = x[:,[3]]
    zeros4[:,[4]] = x[:,[4]]

    gk[0] = zeros0.T.reshape(1,-1)
    gk[1] = zeros1.T.reshape(1,-1)
    gk[2] = zeros2.T.reshape(1,-1)
    gk[3] = zeros3.T.reshape(1,-1)
    gk[4] = zeros4.T.reshape(1,-1)

    return gk.T

def rungekutta_2(t, q, xold, p, l, g, odot, co, ro, W, beta, zeta):
    
    e = e_f(t,q)
    s = s_f(e)
    x = x_f(t,q,s)
    xji = xji_f(t,x,xold,odot,co,ro)
    A = A_f(t,xji,co,ro)
    Aold = A_f(t,xold, co,ro)
    B = B_f(x,Aold,odot,ro)
    mu = mu_f(A)
    muji = muji_f(A)
    omega = omega_f(odot,co,ro,W)
    tau = tau_f(s,A,beta,zeta,omega)
    bk = bk_f(mu,muji,A,Aold,B,co)
    ek = ek_f(mu,muji,A,Aold,B,odot,co,ro,xold)
    gk = gk_f(mu,muji,A,Aold,B,odot,co,ro)

    #xold = x
    #qold = q

    if np.linalg.norm(beta) < 0.25:

        k_q = system(t,q,p,l,g,tau)
        k_W = alpha_w @ (mu.reshape(5,1) - bk.T @ odot.T.reshape(-1,1) - ek.T @ co.T.reshape(-1,1) - gk.T @ ro.T.reshape(-1,1)) @ s.T
        k_odot = (alpha_odot @ bk @ W @ s).reshape(5,15).T
        k_co = (alpha_co @ ek @ W @ s).reshape(5,15).T
        k_ro = (alpha_ro @ gk @ W @ s).reshape(5,15).T
        k_beta = np.linalg.norm(s) * alpha_beta @ omega
        k_zeta = -alpha_zeta * zeta
    
    else :
        k_q = system(t,q,p,l,g,tau)
        k_W = 0.0
        k_odot = 0.0
        k_co = 0.0
        k_ro = 0.0
        k_beta = 0.0
        k_zeta = 0.0
    
    return e, x, k_q, k_odot, k_co, k_ro, k_W, k_beta, k_zeta

n_seed = 13
np.random.seed(n_seed)
print(n_seed)

alpha_w = 50 * np.identity(5, dtype=np.float64)
alpha_odot = 20 * np.identity(75, dtype=np.float64)
alpha_co = 20 * np.identity(75, dtype=np.float64)
alpha_ro = 20 * np.identity(75, dtype=np.float64)
alpha_beta = 0.001 * np.identity(5, dtype=np.float64)
alpha_zeta = 0.1

zeta = 1
omega = np.ones((5,1), dtype=np.float64)

p = np.array([4, 3, 1.5])
l = np.array([0.4, 0.3, 0.2])
g = 10

beta = 0.1 * np.array([
    [1],
    [1],
    [1],
    [1],
    [1]],
    dtype=np.float64
)

t = 0.0
end = 20
step = 0.0001
i = 0

m = -0.01
n = 1.01
q = np.array([
    [m * 0.5, n * np.pi],
    [m * 0.5, n * np.pi],
    [m * 0.5, n * np.pi]],
    dtype=np.float64
)
xold = []
xold.append(np.array([[m*0.5,m*0.5,m*0.5,n*np.pi,n*np.pi,n*np.pi,0,0,0,np.pi,np.pi,np.pi,(1-n)*np.pi-m*2.5,(1-n)*np.pi-m*2.5,(1-n)*np.pi-m*2.5]], dtype=np.float64).reshape(-1,1))

W = 50 * 2 * (np.random.rand(5,3) - 0.5)
j_q = 1.0 * 0.5
j_dq = 1.0 * np.pi
j_s = 0.1 * 1.0 * np.pi * np.sqrt(2)
j = np.array([[j_q,j_q,j_q,j_dq,j_dq,j_dq,j_q,j_q,j_q,j_dq,j_dq,j_dq,j_s,j_s,j_s]]).T
odot = j * 0.1 * 2 * (np.random.rand(15,5) - 0.5)
co = (1/j) * 0.5 * 2 * (np.random.rand(15,5) - 0.5)
ro = j * 1 * 2 * (np.random.rand(15,5) - 0.5)

print("W")
print(np.round(W,4))
print("odot")
print(np.round(odot,4))
print("co")
print(np.round(co,4))
print("ro")
print(np.round(ro,4))
print("beta")
print(beta)
print("zeta")
print(zeta)
"""
print("W")
print(W.dtype)
print("odot")
print(odot.dtype)
print("co")
print(co.dtype)
print("ro")
print(ro.dtype)
print("beta")
print(beta.dtype)
#print("zeta")
#print(zeta.dtype)
"""

#beta = np.random.randn(5,1)

#q_data = np.empty((1,6), dtype=np.float64)                         
#e_data = np.empty((1,6), dtype=np.float64)
e1_data = []
e2_data = []
e3_data = []
e4_data = []
t_data = []

start = time.time()

while t < end:

    e_k1,x_k1,k1_q,k1_odot,k1_co,k1_ro,k1_W,k1_beta,k1_zeta = rungekutta_2(t,q,xold[-1],p,l,g,odot,co,ro,W,beta,zeta)
    e_k2,x_k2,k2_q,k2_odot,k2_co,k2_ro,k2_W,k2_beta,k2_zeta = rungekutta_2(t+step/2,q+(step/2)*k1_q,(x_k1+xold[-1])/2,p,l,g,odot+(step/2)*k1_odot,co+(step/2)*k1_co,ro+(step/2)*k1_ro,W+(step/2)*k1_W,beta+(step/2)*k1_beta,zeta+(step/2)*k1_zeta)
    e_k3,x_k3,k3_q,k3_odot,k3_co,k3_ro,k3_W,k3_beta,k3_zeta = rungekutta_2(t+step/2,q+(step/2)*k2_q,(x_k1+xold[-1])/2,p,l,g,odot+(step/2)*k2_odot,co+(step/2)*k2_co,ro+(step/2)*k2_ro,W+(step/2)*k2_W,beta+(step/2)*k2_beta,zeta+(step/2)*k2_zeta)
    e_k4,x_k4,k4_q,k4_odot,k4_co,k4_ro,k4_W,k4_beta,k4_zeta = rungekutta_2(t+step,q+step*k3_q,x_k1,p,l,g,odot+step*k3_odot,co+step*k3_co,ro+step*k3_ro,W+step*k3_W,beta+step*k3_beta,zeta+step*k3_zeta)

    qdt = qd(t)
    e = e_f(t,q)
    s = s_f(e)
    x = x_f(t,q,s)
    xji = xji_f(t,x,xold[-1],odot,co,ro)
    A = A_f(t,xji,co,ro)
    Aold = A_f(t,xold[-1], co,ro)
    B = B_f(x,Aold,odot,ro)
    mu = mu_f(A)
    muji = muji_f(A)
    omega = omega_f(odot,co,ro,W)
    tau = tau_f(s,A,beta,zeta,omega)

    xold.append(x_k1.copy())

    e1_data.append(e[0][0])
    e2_data.append(e[1][0])
    e3_data.append(e[2][0])
    e4_data.append(np.linalg.norm(s))
    t_data.append(t)

    q += (step / 6) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
    W += (step / 6) * (k1_W + 2 * k2_W + 2 * k3_W + k4_W)
    odot += (step / 6) * (k1_odot + 2 * k2_odot + 2 * k3_odot + k4_odot)
    co += (step / 6) * (k1_co + 2 * k2_co + 2 * k3_co + k4_co)
    ro += (step / 6) * (k1_ro + 2 * k2_ro + 2 * k3_ro + k4_ro)
    beta += (step / 6) * (k1_beta + 2 * k2_beta + 2 * k3_beta + k4_beta)
    zeta += (step / 6) * (k1_zeta + 2 * k2_zeta + 2 * k3_zeta + k4_zeta)

    t += step
    i += 1
    if i%1000 == 0:
        print(round(100*t/end, 1),"%",i,round(time.time()-start, 1),"s")
        """
        print("x")
        print(x)
        print("xji")
        print(xji)
        print("A")
        print(A)
        print("mu")
        print(mu)
        print("muji")
        print(muji)
        """

print("odot_j")
print(np.round(odot/j,4))
print("co_j")
print(np.round(co*j,4))
print("ro_j")
print(np.round(ro/j,4))

print("W")
print(np.round(W,4))
print("odot")
print(np.round(odot,4))
print("co")
print(np.round(co,4))
print("ro")
print(np.round(ro,4))
print("beta")
print(beta)
print("zeta")
print(zeta)
"""
print("W")
print(W.dtype)
print("odot")
print(odot.dtype)
print("co")
print(co.dtype)
print("ro")
print(ro.dtype)
print("beta")
print(beta.dtype)
#print("zeta")
#print(zeta.dtype)
"""
"""
print("beta")
print(np.linalg.norm(W.T @ bk.T))
print(np.linalg.norm(W.T @ ek.T))
print(np.linalg.norm(W.T @ gk.T))
print(np.linalg.norm(bk.T @ odot.T.reshape(-1,1) + ek.T @ co.T.reshape(-1,1) + gk.T @ ro.T.reshape(-1,1)))
"""

fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

ax1.plot(t_data, e1_data)
ax2.plot(t_data, e2_data)
ax3.plot(t_data, e3_data)
ax4.plot(t_data, e4_data)

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()

fig.tight_layout()

plt.grid

plt.title(f"s_s{n_seed}")
plt.show()

np.savetxt(f"z_s{n_seed}_e1.csv",e1_data)
np.savetxt(f"z_s{n_seed}_e2.csv",e2_data)
np.savetxt(f"z_s{n_seed}_e3.csv",e3_data)
np.savetxt(f"z_s{n_seed}_s.csv",e4_data)