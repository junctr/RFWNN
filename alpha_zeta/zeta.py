import numpy as np
from matplotlib import pyplot as plt

zeta = 1
alpha_zeta = 0.1

t = 0.0
end = 100
step = 0.0001

t_data = []
zeta_data = []
"""
while zeta > 0.9:

    t_data.append(t)
    zeta_data.append(zeta)

    zeta += -step * alpha_zeta * zeta

    t += step
"""
while t < end:

    t_data.append(t)
    zeta_data.append(zeta)

    zeta += -step * alpha_zeta * zeta

    t += step

print(zeta)
print(t)


plt.plot(t_data, zeta_data)

plt.grid()

plt.show()