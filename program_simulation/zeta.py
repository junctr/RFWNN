import numpy as np
from matplotlib import pyplot as plt

zeta1 = 1
zeta2 = 1
zeta3 = 0.1
alpha_zeta = 0.1

t = 0.0
end = 80

step = 0.0001

t_data = []
zeta1_data = []
zeta2_data = []
zeta3_data = []
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

    zeta1 += -step * alpha_zeta * zeta1

    if zeta2 > 0.1:
        zeta2 += -step * alpha_zeta * zeta2

    t += step


print(zeta1)
print(zeta2)
print(t)


plt.plot(t_data, zeta2_data, color="tab:red", linestyle = "-", linewidth = 4.0, label = "Proposed")
plt.plot(t_data, zeta1_data, color="tab:green", linestyle = "-.", linewidth = 2.0, label = "Conventional")
plt.plot(t_data, zeta3_data, color="tab:orange", linestyle = "--", linewidth = 2.0, label = r"$\zeta_{min}$")



plt.xlabel("Time [s]")
plt.ylabel(r"$\zeta$")

plt.legend()

plt.grid()

plt.show()