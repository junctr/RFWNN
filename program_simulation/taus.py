import numpy as np
import time
#import scipy.integrate as sp
from matplotlib import pyplot as plt

t = 0.0
end = 200
step = 0.0001
i = 0

zeta = 1
beta_omega = 20
s = 0.005

z_data = []
taus_data = []
t_data = []

start = time.time()


while t < end:

    taus = (s * beta_omega**2) / (s * beta_omega + zeta)

    z_data.append(zeta)
    taus_data.append(taus)
    t_data.append(t)

    zeta += -step * 0.1 * zeta
    t += step
    i += 1
    if i%1000 == 0:
        print(round(100*t/end, 1),"%",i,round(time.time()-start, 1),"s")

plt.plot(t_data,z_data)
plt.plot(t_data,taus_data)
plt.grid()
plt.title("beta_omega_zeta")
plt.show()