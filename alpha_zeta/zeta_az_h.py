import numpy as np
from matplotlib import pyplot as plt

zeta1 = 1
zeta2 = 1
zeta3 = 1
zeta4 = 1
zeta5 = 1

alpha_zeta1 = 0.025
alpha_zeta2 = 0.05
alpha_zeta3 = 0.1
alpha_zeta4 = 0.2
alpha_zeta5 = 0.4

t = 0.0
end = 100
step = 0.0001

t_data = []
zeta1_data = []
zeta2_data = []
zeta3_data = []
zeta4_data = []
zeta5_data = []
"""
while zeta > 0.9:

    t_data.append(t)
    zeta_data.append(zeta)

    zeta += -step * alpha_zeta * zeta

    t += step
"""
while t < end:

    t_data.append(t)
    zeta1_data.append(zeta1)
    zeta2_data.append(zeta2)
    zeta3_data.append(zeta3)
    zeta4_data.append(zeta4)
    zeta5_data.append(zeta5)

    zeta1 += -step * alpha_zeta1 * zeta1
    zeta2 += -step * alpha_zeta2 * zeta2
    zeta3 += -step * alpha_zeta3 * zeta3
    zeta4 += -step * alpha_zeta4 * zeta4
    zeta5 += -step * alpha_zeta5 * zeta5

    t += step

#print(zeta)
#print(t)


plt.plot(t_data, zeta1_data)
plt.plot(t_data, zeta2_data)
plt.plot(t_data, zeta3_data)
plt.plot(t_data, zeta4_data)
plt.plot(t_data, zeta5_data)

plt.grid()

plt.show()