import csv
import numpy as np
from matplotlib import pyplot as plt

s = 7
m = 0.1

t_data = np.loadtxt("time20.csv")
e1_m_data = np.loadtxt(f"s{s}_m{m}_e1.csv")
e2_m_data = np.loadtxt(f"s{s}_m{m}_e2.csv")
e3_m_data = np.loadtxt(f"s{s}_m{m}_e3.csv")
e1_nasi_data = np.loadtxt(f"s{s}_m0.0_e1.csv")
e2_nasi_data = np.loadtxt(f"s{s}_m0.0_e2.csv")
e3_nasi_data = np.loadtxt(f"s{s}_m0.0_e3.csv")

fig = plt.figure()

ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

ax1.plot(t_data, e1_nasi_data, label = "Conventional")
ax2.plot(t_data, e2_nasi_data, label = "Conventional")
ax3.plot(t_data, e3_nasi_data, label = "Conventional")

ax1.plot(t_data, e1_m_data, label = f"Proposed{m}")
ax2.plot(t_data, e2_m_data, label = f"Proposed{m}")
ax3.plot(t_data, e3_m_data, label = f"Proposed{m}")

#ax1.xlim(0,20)
ax1.legend()
ax1.grid()
#ax2.xlim(0,20)
ax2.legend()
ax2.grid()
#ax3.xlim(0,20)
ax3.legend()
ax3.grid()

plt.show()